"""Audio and Vision I/O Service for Anubis with VAD, Multi-Monitor, and Wake-Word Support."""

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
from faster_whisper import WhisperModel

logger = structlog.get_logger(__name__)

class AudioVisionService:
    """Orchestrates synchronized screen and audio capture with Wake-Word activation."""
    
    def __init__(self, hotkey='f14', use_vad=True, wake_word="anubis"):
        self.hotkey = hotkey
        self.use_vad = use_vad
        self.wake_word = wake_word.lower()
        self.sample_rate = 16000  # Optimized for VAD and Whisper
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
        self.is_speech_active = False
        self.silence_threshold_ms = 800
        self.last_speech_time = 0
        
        # Wake Word State
        self.whisper_model = None
        self.is_waiting_for_command = False # True after wake word is detected
        
        try:
            pygame.mixer.init()
        except Exception:
            pass
            
        if self.use_vad:
            self._init_models()

    def _init_models(self):
        """Initialize VAD and local Whisper for wake-word detection."""
        try:
            # VAD
            self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                            model='silero_vad',
                                            trust_repo=True)
            # Whisper Tiny (extremely fast for keyword detection)
            logger.info("loading_whisper_tiny")
            self.whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
            logger.info("models_loaded")
        except Exception as e:
            logger.error("model_init_failed", error=str(e))
            self.use_vad = False

    def _buffer_audio_stream(self, indata, frames, time_info, status):
        """Buffer audio and run VAD/Wake-word logic."""
        audio_data = indata.copy()
        self.audio_chunks.append(audio_data)
        
        if self.use_vad and self.vad_model:
            tensor_data = torch.from_numpy(audio_data.flatten())
            speech_prob = self.vad_model(tensor_data, self.sample_rate).item()
            
            if speech_prob > 0.5:
                if not self.is_speech_active:
                    self.is_speech_active = True
                    logger.info("speech_detected")
                self.last_speech_time = time.time()
            else:
                if self.is_speech_active:
                    elapsed_silence = (time.time() - self.last_speech_time) * 1000
                    if elapsed_silence > self.silence_threshold_ms:
                        self.is_speech_active = False
                        # Trigger local processing to check for wake-word
                        threading.Thread(target=self._process_voice_trigger, daemon=True).start()

    def _process_voice_trigger(self):
        """Check if the recorded audio contains the wake word."""
        if not self.audio_chunks:
            return

        # Prepare audio for Whisper
        audio_full = np.concatenate(self.audio_chunks, axis=0).flatten()
        
        # We check for the wake word in the current buffer
        segments, _ = self.whisper_model.transcribe(audio_full, beam_size=1)
        transcript = " ".join([seg.text for segments in segments for seg in [segments]]).lower()
        
        logger.info("transcribed_locally", text=transcript)

        if self.wake_word in transcript:
            logger.info("wake_word_detected", word=self.wake_word)
            # 1. Capture Vision
            self._capture_vision()
            
            # 2. Save Audio
            sf.write(self.audio_buffer_path, audio_full, self.sample_rate)
            
            # 3. Trigger Gemini
            if self.on_input_ready:
                self.on_input_ready(self.audio_buffer_path, self.vision_buffer_path)
        
        # Reset chunks for the next listening cycle
        self.audio_chunks = []

    def _capture_vision(self):
        """Capture all monitors."""
        try:
            sct_img = self.mss.grab(self.mss.monitors[0])
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.save(self.vision_buffer_path)
            logger.info("vision_captured", monitors=len(self.mss.monitors)-1)
        except Exception as exc:
            logger.error("vision_capture_failed", error=str(exc))

    def start_multimodal_capture(self):
        """Start the background listening stream."""
        if not self.recording:
            logger.info("anubis_listening_background", mode="wake_word")
            self.recording = True
            self.audio_chunks = []
            
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate, 
                    channels=self.channels, 
                    callback=self._buffer_audio_stream,
                    blocksize=1024 
                )
                self.stream.start()
            except Exception as exc:
                logger.error("stream_failed", error=str(exc))
                self.recording = False

    def listen_in_background(self, callback) -> None:
        """Entry point for always-on mode."""
        self.on_input_ready = callback
        self.start_multimodal_capture()

    async def speak(self, text: str):
        """Convert text to speech and play back."""
        # Stop listening while speaking to avoid feedback loops
        if self.stream:
            self.stream.stop()
            
        clean_text = re.sub(r'[*`#]', '', text).strip()
        lang = "en" # Simplified for now
        voice = "en-US-ChristopherNeural"
        
        logger.info("generating_tts", voice=voice)
        audio_file = "anubis_response.mp3"
        
        try:
            communicate = edge_tts.Communicate(clean_text, voice)
            await communicate.save(audio_file)
            
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
            pygame.mixer.music.unload()
        except Exception as e:
            logger.error("tts_failed", error=str(e))
        finally:
            # Resume listening
            if self.stream:
                self.stream.start()
                self.audio_chunks = [] # Clear any feedback audio
