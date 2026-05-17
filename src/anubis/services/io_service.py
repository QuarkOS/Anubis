"""
Audio and Vision I/O Service for Anubis with Master-Level Architecture.
Utilizes continuous parallel streaming, OpenWakeWord for instant (<50ms) triggers,
and decoupled VAD + Faster-Whisper for high-accuracy command processing.
"""

import os
import re
import queue
import threading
import collections
import time
import numpy as np
from PIL import Image
import mss
from pynput import keyboard
import sounddevice as sd
import soundfile as sf
import pygame
import structlog
import torch

from faster_whisper import WhisperModel
from openwakeword.model import Model as OWWModel
from kokoro_onnx import Kokoro

logger = structlog.get_logger(__name__)

class AudioVisionService:
    """
    Master-Level orchestration of multimodal I/O.
    Features an asynchronous ring buffer and decoupled state machine for instant wake-word feedback.
    """
    
    def __init__(self, hotkey='f14', use_vad=True):
        self.hotkey = hotkey
        self.use_vad = use_vad
        
        # Audio Configuration (16kHz is required by Whisper, Silero, and OpenWakeWord)
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1280 # 80ms chunk for OpenWakeWord optimal performance
        
        # Concurrency & Buffering
        self.audio_stream_queue = queue.Queue()
        self.is_running = False
        
        # The Ring Buffer keeps the last 3 seconds of audio to catch context 
        # immediately preceding the VAD silence cutoff.
        self.ring_buffer_chunks = int((16000 * 3) / self.chunk_size)
        self.ring_buffer = collections.deque(maxlen=self.ring_buffer_chunks)
        
        # State Machine Flags
        self.spooling_command = False
        self.command_buffer = []
        self.is_speech_active = False
        self.last_speech_time = 0
        self.silence_threshold_ms = 1000
        
        # Callbacks
        self.on_input_ready = None
        self.on_wake_word_detected = None
        
        # Paths
        self.vision_buffer_path = "anubis_vision_context.png"
        self.audio_buffer_path = "anubis_audio_context.wav"
        
        self.mss = mss.mss()
        self.key_listener = None
        self.stream = None
        
        # Models
        self.oww_model = None
        self.vad_model = None
        self.whisper_model = None
        self.kokoro = None
        
        try:
            pygame.mixer.init()
        except Exception:
            pass
            
        if self.use_vad:
            self._init_models()

    def _init_models(self):
        """Warm up all local inference models across the parallel pipeline."""
        try:
            logger.info("loading_openwakeword")
            # Using 'hey_jarvis' as a placeholder to prove the Master-Level Architecture.
            # Replace with anubis.onnx once custom trained.
            self.oww_model = OWWModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
            
            logger.info("loading_silero_vad")
            self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                            model='silero_vad',
                                            trust_repo=True)
                                            
            logger.info("loading_whisper_base")
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            
            logger.info("loading_kokoro_tts")
            model_path = os.path.join("kokoro_models", "kokoro-v1.0.onnx")
            voices_path = os.path.join("kokoro_models", "voices.bin")
            self.kokoro = Kokoro(model_path, voices_path)
            
            logger.info("models_loaded_successfully")
        except Exception as e:
            logger.error("model_init_failed", error=str(e))
            self.use_vad = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Low-latency callback pushing raw 32/80ms chunks into the processing queue."""
        if status:
            logger.warning("audio_stream_status", status=status)
        self.audio_stream_queue.put(indata.copy())

    def _inference_worker(self):
        """
        The core state machine running on a dedicated thread.
        Consumes the audio queue, runs OpenWakeWord instantly, and spools commands via VAD.
        """
        while self.is_running:
            try:
                # Block until audio is available
                chunk = self.audio_stream_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Update the 3-second ring buffer
            self.ring_buffer.append(chunk)

            if not self.spooling_command:
                # STATE: LISTENING FOR WAKE WORD
                # 1. Instantaneous Inference (<50ms latency)
                try:
                    prediction = self.oww_model.predict(chunk.flatten())
                    # Using the hey_jarvis model as our proof-of-concept trigger
                    if prediction.get('hey_jarvis_v0.1', 0.0) > 0.5:
                        logger.info("openwakeword_detected", score=prediction['hey_jarvis_v0.1'])
                        
                        if self.on_wake_word_detected:
                            self.on_wake_word_detected() # Instantly trigger hardware UI
                            
                        self.spooling_command = True
                        self.is_speech_active = True
                        self.last_speech_time = time.time()
                        
                        # Pre-fill the command buffer with the ring buffer to catch anything
                        # said immediately before or during the wake-word detection.
                        self.command_buffer = list(self.ring_buffer)
                except Exception as e:
                    logger.error("oww_inference_error", error=str(e))
            
            else:
                # STATE: SPOOLING COMMAND
                self.command_buffer.append(chunk)
                
                # Check VAD for silence
                tensor_data = torch.from_numpy(chunk.flatten())
                speech_prob = self.vad_model(tensor_data, self.sample_rate).item()
                
                if speech_prob > 0.5:
                    self.is_speech_active = True
                    self.last_speech_time = time.time()
                else:
                    if self.is_speech_active:
                        elapsed_silence = (time.time() - self.last_speech_time) * 1000
                        if elapsed_silence > self.silence_threshold_ms:
                            # End of command detected
                            self.is_speech_active = False
                            self.spooling_command = False
                            
                            # Process the spooled command in a separate thread to unblock inference
                            threading.Thread(
                                target=self._process_spooled_command, 
                                args=(list(self.command_buffer),),
                                daemon=True
                            ).start()
                            
                            self.command_buffer.clear()
                            # Reset ring buffer to avoid accidental double-triggers
                            self.ring_buffer.clear()

    def _process_spooled_command(self, audio_chunks):
        """Run Faster-Whisper on perfectly segmented audio chunks."""
        audio_full = np.concatenate(audio_chunks, axis=0).flatten()
        
        # Audio Normalization
        max_vol = np.max(np.abs(audio_full))
        if max_vol < 0.01:
            return
        audio_full = audio_full / max_vol * 0.9
        
        logger.info("transcribing_spooled_command")
        segments, _ = self.whisper_model.transcribe(
            audio_full, 
            beam_size=5
        )
        
        transcript = "".join([s.text for s in segments]).strip().lower()
        if not transcript:
            return
            
        logger.info("transcribed_locally", text=transcript)
        
        self._capture_vision()
        sf.write(self.audio_buffer_path, audio_full, self.sample_rate)
        
        if self.on_input_ready:
            self.on_input_ready(self.audio_buffer_path, self.vision_buffer_path)

    def _trigger_manual_capture(self):
        """Hotkey override to immediately spool the last 3 seconds and trigger."""
        if self.on_wake_word_detected:
            self.on_wake_word_detected()
            
        audio_full = np.concatenate(list(self.ring_buffer), axis=0).flatten() if self.ring_buffer else np.zeros(self.sample_rate, dtype=np.float32)
        
        max_vol = np.max(np.abs(audio_full))
        if max_vol > 0.01:
            audio_full = audio_full / max_vol * 0.9
            
        self._capture_vision()
        sf.write(self.audio_buffer_path, audio_full, self.sample_rate)
        
        if self.on_input_ready:
            self.on_input_ready(self.audio_buffer_path, self.vision_buffer_path)

    def _capture_vision(self):
        """Capture a composite image of all connected monitors using mss."""
        try:
            sct_img = self.mss.grab(self.mss.monitors[0])
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.save(self.vision_buffer_path)
            logger.info("vision_captured", monitors=len(self.mss.monitors)-1)
        except Exception as exc:
            logger.error("vision_capture_failed", error=str(exc))

    def listen_in_background(self, callback, on_wake_word=None) -> None:
        """Initialize the Parallel Stream Architecture."""
        if self.is_running:
            return
            
        self.on_input_ready = callback
        self.on_wake_word_detected = on_wake_word
        self.is_running = True
        
        logger.info("starting_master_level_architecture", hotkey=self.hotkey)
        
        # 1. Start Inference Consumer Thread
        threading.Thread(target=self._inference_worker, daemon=True).start()
        
        # 2. Start Audio Producer Stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels, 
                callback=self._audio_callback,
                blocksize=self.chunk_size
            )
            self.stream.start()
        except Exception as exc:
            logger.error("stream_failed", error=str(exc))
            self.is_running = False

        # 3. Start Hotkey Listener
        def on_press(key):
            try:
                key_name = key.name if hasattr(key, 'name') else str(key)
                if key_name == self.hotkey:
                    logger.info("hotkey_pressed", key=self.hotkey)
                    self._trigger_manual_capture()
            except Exception:
                pass
                
        self.key_listener = keyboard.Listener(on_press=on_press)
        self.key_listener.start()

    async def speak(self, text: str):
        """Gapless TTS generation and playback loop."""
        if not pygame.mixer.get_init():
            logger.warning("pygame_mixer_not_initialized_skipping_tts", text=text)
            return

        if self.stream:
            self.stream.stop()
            
        clean_text = re.sub(r'[*`#]', '', text).strip()
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', clean_text) if s.strip()]
        if not sentences:
            return

        logger.info("generating_kokoro_tts_gapless", voice="bf_isabella", sentence_count=len(sentences))
        audio_queue = asyncio.Queue()
        
        async def producer():
            try:
                for sentence in sentences:
                    samples, sample_rate = await asyncio.to_thread(
                        self.kokoro.create,
                        sentence,
                        voice="bf_isabella",
                        speed=1.0,
                        lang="en-gb"
                    )
                    await audio_queue.put((samples, sample_rate))
                await audio_queue.put(None)
            except Exception as e:
                logger.error("tts_producer_failed", error=str(e))
                await audio_queue.put(None)

        producer_task = asyncio.create_task(producer())

        try:
            seg_idx = 0
            while True:
                item = await audio_queue.get()
                if item is None:
                    break
                    
                samples, sample_rate = item
                audio_file = f"anubis_response_seg_{seg_idx % 2}.wav"
                sf.write(audio_file, samples, sample_rate)
                
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.01)
                
                seg_idx += 1
                
            pygame.mixer.music.unload()
        except Exception as e:
            logger.error("kokoro_tts_failed", error=str(e))
        finally:
            await producer_task
            if self.stream:
                # Flush queues to prevent old audio from ghost-triggering
                while not self.audio_stream_queue.empty():
                    self.audio_stream_queue.get()
                self.stream.start()
