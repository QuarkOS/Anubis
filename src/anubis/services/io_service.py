"""Audio and Vision I/O Service for Anubis with VAD and Multi-Monitor Support."""

import time
import threading
import asyncio
import io
import os
import re
from PIL import Image
import mss
from pynput import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import edge_tts
import pygame
import structlog
import torch

logger = structlog.get_logger(__name__)

class AudioVisionService:
    """Orchestrates synchronized screen and audio capture for multimodal processing."""
    
    def __init__(self, hotkey='f14', use_vad=True):
        self.hotkey = hotkey
        self.use_vad = use_vad
        self.sample_rate = 16000  # Silero VAD prefers 16kHz
        self.channels = 1
        
        self.recording = False
        self.stream = None
        self.audio_chunks = []
        
        self.vision_buffer_path = "anubis_vision_context.png"
        self.audio_buffer_path = "anubis_audio_context.wav"
        
        self.on_input_ready = None
        self.mss = mss.mss()
        
        # VAD State
        self.vad_model = None
        self.vad_utils = None
        self.is_speech_active = False
        self.silence_threshold_ms = 1000
        self.last_speech_time = 0
        self.vad_buffer = []
        
        try:
            pygame.mixer.init()
            logger.info("pygame_mixer_initialized")
        except Exception as e:
            logger.error("pygame_mixer_failed", error=str(e))
            
        if self.use_vad:
            self._init_vad()

    def _init_vad(self):
        """Initialize Silero VAD model."""
        try:
            logger.info("loading_vad_model")
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        force_reload=False,
                                        trust_repo=True)
            self.vad_model = model
            self.vad_utils = utils
            logger.info("vad_model_loaded")
        except Exception as e:
            logger.error("vad_init_failed", error=str(e))
            self.use_vad = False

    def _buffer_audio_stream(self, indata, frames, time_info, status):
        """Append incoming audio data and perform VAD if enabled."""
        audio_data = indata.copy()
        self.audio_chunks.append(audio_data)
        
        if self.use_vad and self.vad_model:
            # Convert to torch tensor for VAD
            tensor_data = torch.from_numpy(audio_data.flatten())
            speech_prob = self.vad_model(tensor_data, self.sample_rate).item()
            
            if speech_prob > 0.5:
                if not self.is_speech_active:
                    logger.info("speech_started")
                    self.is_speech_active = True
                    # If this is the first speech in a VAD session, capture vision
                    self._capture_vision()
                self.last_speech_time = time.time()
            else:
                if self.is_speech_active:
                    elapsed_silence = (time.time() - self.last_speech_time) * 1000
                    if elapsed_silence > self.silence_threshold_ms:
                        logger.info("speech_ended_by_vad")
                        self.is_speech_active = False
                        # Trigger processing callback in a thread to not block stream
                        threading.Thread(target=self.stop_multimodal_capture, daemon=True).start()

    def _capture_vision(self):
        """Capture all monitors and stitch them together."""
        try:
            # mss mon 0 is all monitors
            sct_img = self.mss.grab(self.mss.monitors[0])
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.save(self.vision_buffer_path)
            logger.info("vision_captured", monitors=len(self.mss.monitors)-1)
        except Exception as exc:
            logger.error("vision_capture_failed", error=str(exc))

    def start_multimodal_capture(self):
        """Initiate concurrent screen and audio capture."""
        if not self.recording:
            logger.info("capture_started", mode="vad" if self.use_vad else "ptt")
            self.recording = True
            self.audio_chunks = []
            self.is_speech_active = False
            
            if not self.use_vad:
                self._capture_vision()
            
            try:
                # We always open the stream, but VAD logic will handle the "trigger"
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate, 
                    channels=self.channels, 
                    callback=self._buffer_audio_stream,
                    blocksize=512 # required for Silero VAD
                )
                self.stream.start()
            except Exception as exc:
                logger.error("audio_capture_failed", error=str(exc))
                self.recording = False

    def stop_multimodal_capture(self):
        """Stop recording, finalize audio file, and trigger processing callback."""
        if self.recording:
            # If using VAD, we don't close the stream immediately unless it's a global shutdown
            # But for the sake of the existing architecture, we'll process the current buffer
            
            if self.audio_chunks:
                final_audio = np.concatenate(self.audio_chunks, axis=0)
                sf.write(self.audio_buffer_path, final_audio, self.sample_rate)
                
                # Reset buffers for next VAD cycle if we want to keep it always on
                # For now, we'll stop the stream to match the "request-response" pattern
                if not self.use_vad:
                   self.recording = False
                   if self.stream:
                       self.stream.stop()
                       self.stream.close()
                       self.stream = None
                else:
                    # In VAD mode, we clear chunks so the next interaction is fresh
                    self.audio_chunks = []
                
                logger.info("processing_input")
                if self.on_input_ready:
                    self.on_input_ready(self.audio_buffer_path, self.vision_buffer_path)
            else:
                logger.warning("no_audio_recorded")

    def listen_in_background(self, callback) -> None:
        """Set up triggers for capture. PTT if VAD is off, otherwise background VAD."""
        self.on_input_ready = callback
        
        if self.use_vad:
            logger.info("vad_mode_active", instructions="Anubis is listening...")
            self.start_multimodal_capture()
        else:
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
            logger.info("listening_for_hotkey", key=self.hotkey)

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
        """Detect the primary language of the text using character distribution."""
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
        """Strip markdown syntax."""
        text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    async def speak(self, text: str):
        """Convert text to speech and play back."""
        clean_text = self.remove_markdown_for_tts(text)
        lang = self.identify_language_from_text(clean_text)
        voice = self.LANGUAGE_VOICE_MAPPING.get(lang, "en-US-ChristopherNeural")
        
        logger.info("generating_tts", language=lang, voice=voice)
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
