"""Audio and Vision I/O Service for Anubis."""

import time
import threading
import asyncio
import io
from PIL import ImageGrab
from pynput import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import edge_tts
import pygame
import structlog

logger = structlog.get_logger(__name__)

class AudioVisionService:
    """Orchestrates synchronized screen and audio capture for multimodal processing."""
    
    def __init__(self, hotkey='f14'):
        self.hotkey = hotkey
        self.sample_rate = 44100
        self.channels = 1
        
        self.recording = False
        self.stream = None
        self.audio_chunks = []
        
        self.vision_buffer_path = "anubis_vision_context.png"
        self.audio_buffer_path = "anubis_audio_context.wav"
        
        self.on_input_ready = None
        
        try:
            pygame.mixer.init()
            logger.info("pygame_mixer_initialized")
        except Exception as e:
            logger.error("pygame_mixer_failed", error=str(e))

    def _buffer_audio_stream(self, indata, frames, time, status):
        """Append incoming audio data to the internal buffer during active recording."""
        if self.recording:
            self.audio_chunks.append(indata.copy())

    def start_multimodal_capture(self):
        """Initiate concurrent screen and audio capture."""
        if not self.recording:
            logger.info("hotkey_pressed", action="starting_recording_and_vision")
            self.recording = True
            self.audio_chunks = []
            
            try:
                screenshot = ImageGrab.grab()
                screenshot.save(self.vision_buffer_path)
            except Exception as exc:
                logger.error("vision_capture_failed", error=str(exc))
            
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate, 
                    channels=self.channels, 
                    callback=self._buffer_audio_stream
                )
                self.stream.start()
            except Exception as exc:
                logger.error("audio_capture_failed", error=str(exc))
                self.recording = False

    def stop_multimodal_capture(self):
        """Stop recording, finalize audio file, and trigger processing callback."""
        if self.recording:
            self.recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                
            logger.info("hotkey_released", action="processing_input")
            
            if self.audio_chunks:
                final_audio = np.concatenate(self.audio_chunks, axis=0)
                sf.write(self.audio_buffer_path, final_audio, self.sample_rate)
                
                if self.on_input_ready:
                    threading.Thread(
                        target=self.on_input_ready, 
                        args=(self.audio_buffer_path, self.vision_buffer_path),
                        daemon=True
                    ).start()
            else:
                logger.warning("no_audio_recorded")

    def listen_in_background(self, callback) -> None:
        """Set up a global keyboard listener to trigger capture on hotkey press."""
        self.on_input_ready = callback
        
        def on_press(key):
            try:
                if hasattr(key, 'name') and key.name == self.hotkey:
                    self.start_multimodal_capture()
                elif str(key).replace("'", "") == self.hotkey:
                    self.start_multimodal_capture()
            except Exception:
                pass

        def on_release(key):
            try:
                if hasattr(key, 'name') and key.name == self.hotkey:
                    self.stop_multimodal_capture()
                elif str(key).replace("'", "") == self.hotkey:
                    self.stop_multimodal_capture()
            except Exception:
                pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        logger.info("listening_for_hotkey", key=self.hotkey, provider="pynput")

    LANGUAGE_VOICE_MAPPING = {
        "de": "de-DE-KillianNeural",
        "en": "en-US-ChristopherNeural",
        "fr": "fr-FR-HenriNeural",
        "es": "es-ES-AlvaroNeural",
        "it": "it-IT-DiegoNeural",
        "pt": "pt-BR-AntonioNeural",
        "nl": "nl-NL-MaartenNeural",
        "pl": "pl-PL-MarekNeural",
        "ja": "ja-JP-KeitaNeural",
        "ko": "ko-KR-InJoonNeural",
        "zh": "zh-CN-YunxiNeural",
        "ru": "ru-RU-DmitryNeural",
    }

    @staticmethod
    def identify_language_from_text(text: str) -> str:
        """Detect the primary language of the text using character distribution and keyword heuristics."""
        latin_extended = 0
        cjk = 0
        cyrillic = 0
        german_chars = 0
        
        for ch in text:
            cp = ord(ch)
            if ch in "äöüÄÖÜß":
                german_chars += 1
            elif 0x0400 <= cp <= 0x04FF:
                cyrillic += 1
            elif 0x4E00 <= cp <= 0x9FFF:
                cjk += 1
            elif 0x3040 <= cp <= 0x30FF:
                cjk += 1
        
        if cyrillic > 5:
            return "ru"
        if cjk > 5:
            return "zh"
        if german_chars > 2:
            return "de"
        
        lower = text.lower()
        if any(w in lower for w in [" ist ", " und ", " das ", " ein ", " ich ", " nicht ", " auch ", " dir ", " du "]):
            return "de"
        if any(w in lower for w in [" est ", " les ", " des ", " une ", " vous ", " avec "]):
            return "fr"
        if any(w in lower for w in [" está ", " los ", " las ", " una ", " pero ", " como "]):
            return "es"
        if any(w in lower for w in [" sono ", " questo ", " della ", " anche ", " come "]):
            return "it"
        
        return "en"

    @staticmethod
    def remove_markdown_for_tts(text: str) -> str:
        """Strip markdown syntax to prepare text for clear speech synthesis."""
        import re
        text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    async def speak(self, text: str):
        """Convert text to speech and play back using pygame mixer."""
        clean_text = self.remove_markdown_for_tts(text)
        
        lang = self.identify_language_from_text(clean_text)
        voice = self.LANGUAGE_VOICE_MAPPING.get(lang, "en-US-ChristopherNeural")
        
        logger.info("generating_tts", text_length=len(clean_text), language=lang, voice=voice)
        audio_file = "anubis_response.mp3"
        
        try:
            communicate = edge_tts.Communicate(clean_text, voice)
            await communicate.save(audio_file)
            
            logger.info("playing_tts")
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
            pygame.mixer.music.unload()
            logger.info("tts_finished")
        except Exception as e:
            logger.error("tts_failed", error=str(e))
