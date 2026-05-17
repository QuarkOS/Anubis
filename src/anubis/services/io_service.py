"""Audio and Vision I/O Service for Anubis with VAD, Multi-Monitor, Wake-Word, and Kokoro TTS."""

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
import pygame
import structlog
import torch
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from rapidfuzz import fuzz

logger = structlog.get_logger(__name__)

class AudioVisionService:
    """Orchestrates synchronized screen and audio capture with Wake-Word activation and local TTS."""
    
    def __init__(self, hotkey='f14', use_vad=True, wake_word="anubis"):
        """Initialize the service and set up local VAD, Whisper, and Kokoro models."""
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
        self.on_wake_word_detected = None
        self.mss = mss.mss()
        
        # VAD State
        self.vad_model = None
        self.is_speech_active = False
        self.silence_threshold_ms = 1000 # Increased for natural pauses in long requests
        self.last_speech_time = 0
        
        # Wake Word & TTS Models
        self.whisper_model = None
        self.kokoro = None
        self.wake_word_alternatives = [
            "anubis", "a knew this", "i know this", "and boost", 
            "her now bist", "hey anubis", "hey a knew this", 
            "annubis", "a newbie", "hi anubis", "hey a noticed",
            "hey i know this", "here and do this", "here i know this",
            "heja nugus", "janugus", "now bist", "hey anubus", "hey a nubus",
            "hainubis", "hanubis", "hey a new bus", "hey an ubis", "hey a nubes",
            "her nubis", "nubis", "nubus", "ubis", "newbus", "a nubis", "hey a nubis"
        ]
        
        try:
            pygame.mixer.init()
        except Exception:
            pass
            
        if self.use_vad:
            self._init_models()

    def _init_models(self):
        """Warm up local inference models for speech detection, transcription, and synthesis."""
        try:
            # VAD
            self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                            model='silero_vad',
                                            trust_repo=True)
            # Whisper Base
            logger.info("loading_whisper_base")
            self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            
            # Kokoro TTS
            logger.info("loading_kokoro_tts")
            model_path = os.path.join("kokoro_models", "kokoro-v1.0.onnx")
            voices_path = os.path.join("kokoro_models", "voices.bin")
            self.kokoro = Kokoro(model_path, voices_path)
            
            logger.info("models_loaded")
        except Exception as e:
            logger.error("model_init_failed", error=str(e))
            self.use_vad = False

    def _buffer_audio_stream(self, indata, frames, time_info, status):
        """
        Stream callback for managing the rolling audio buffer and triggering VAD logic.
        
        Maintains a 15s-30s buffer that intelligently preserves context during active speech.
        """
        audio_data = indata.copy()
        self.audio_chunks.append(audio_data)
        
        # Buffer capacity logic:
        # 1. We keep a rolling 15s window by default.
        # 2. If speech is active, we NEVER truncate the buffer, allowing long commands to be captured fully.
        # 3. We implement a 30s 'safety cap' to prevent unbounded memory growth in noisy environments.
        
        rolling_window_chunks = int(15 * self.sample_rate / 512)
        hard_limit_chunks = int(30 * self.sample_rate / 512)
        
        if len(self.audio_chunks) > hard_limit_chunks:
            self.audio_chunks.pop(0)
        elif len(self.audio_chunks) > rolling_window_chunks and not self.is_speech_active:
            self.audio_chunks.pop(0)

        if self.use_vad and self.vad_model:
            tensor_data = torch.from_numpy(audio_data.flatten())
            speech_prob = self.vad_model(tensor_data, self.sample_rate).item()
            
            if speech_prob > 0.5: # Lowered back slightly for better detection
                if not self.is_speech_active:
                    self.is_speech_active = True
                    logger.info("speech_detected", probability=round(speech_prob, 2))
                self.last_speech_time = time.time()
            else:
                if self.is_speech_active:
                    elapsed_silence = (time.time() - self.last_speech_time) * 1000
                    if elapsed_silence > self.silence_threshold_ms:
                        self.is_speech_active = False
                        # Trigger local processing
                        threading.Thread(target=self._process_voice_trigger, daemon=True).start()

    def _process_voice_trigger(self):
        """
        Transcribe the current buffer and evaluate wake-word activation.
        
        Uses a combination of exact phonetic matches and fuzzy similarity scores
        to ensure robust activation even with variable Whisper output.
        """
        if not self.audio_chunks:
            return

        # Snapshot current buffer
        current_audio = self.audio_chunks[:]
        audio_full = np.concatenate(current_audio, axis=0).flatten()
        
        # Audio Normalization: ensures consistent signal for Whisper
        max_vol = np.max(np.abs(audio_full))
        if max_vol < 0.01:
            return
        audio_full = audio_full / max_vol * 0.9
        
        # Higher beam_size (5) for better phonetic accuracy
        segments, _ = self.whisper_model.transcribe(
            audio_full, 
            beam_size=5, 
            initial_prompt="Anubis, Hey Anubis."
        )
        
        transcript = "".join([s.text for s in segments]).strip().lower()
        
        if not transcript:
            return

        # Simple repetition filter
        words = transcript.split()
        if len(words) > 8 and any(words.count(w) > (len(words) // 2) for w in set(words)):
            return

        logger.info("transcribed_locally", text=transcript)

        # 1. Exact/Substring match (Fast)
        matched = any(word in transcript for word in self.wake_word_alternatives)

        # 2. Fuzzy match (Robust)
        # We check the first 20 characters for a fuzzy match against 'hey anubis'
        if not matched and len(transcript) > 3:
            start_snippet = transcript[:20]
            score = fuzz.partial_ratio("hey anubis", start_snippet)
            if score > 80: # High confidence fuzzy match
                logger.info("fuzzy_wake_word_detected", score=score, snippet=start_snippet)
                matched = True

        if matched:
            logger.info("wake_word_detected", matched_transcript=transcript)
            
            # 1. Immediate visual feedback
            if self.on_wake_word_detected:
                self.on_wake_word_detected()
                
            # 2. Proceed with full capture
            self._capture_vision()
            sf.write(self.audio_buffer_path, audio_full, self.sample_rate)
            
            # Clear buffer ONLY on successful detection
            self.audio_chunks = []
            
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

    def start_multimodal_capture(self):
        """Initialize background audio monitoring and the F14 hotkey listener."""
        if not self.recording:
            logger.info("anubis_listening_background", mode="wake_word", hotkey=self.hotkey)
            self.recording = True
            self.audio_chunks = []

            # 1. Start Audio Stream (for Wake-Word/VAD)
            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate, 
                    channels=self.channels, 
                    callback=self._buffer_audio_stream,
                    blocksize=512 
                )
                self.stream.start()
            except Exception as exc:
                logger.error("stream_failed", error=str(exc))
                self.recording = False

            # 2. Start Keyboard Listener (for F14 manual trigger)
            def on_press(key):
                try:
                    # Handle both special keys and character keys
                    key_name = key.name if hasattr(key, 'name') else str(key)
                    if key_name == self.hotkey:
                        logger.info("hotkey_pressed", key=self.hotkey)
                        self._trigger_manual_capture()
                except Exception:
                    pass

            self.key_listener = keyboard.Listener(on_press=on_press)
            self.key_listener.start()

    def _trigger_manual_capture(self):
        """Execute a capture turn immediately, regardless of wake-word state."""
        if not self.audio_chunks:
            return

        current_audio = self.audio_chunks[:]
        audio_full = np.concatenate(current_audio, axis=0).flatten()

        # Normalize and save
        max_vol = np.max(np.abs(audio_full))
        if max_vol > 0.01:
            audio_full = audio_full / max_vol * 0.9

        if self.on_wake_word_detected:
            self.on_wake_word_detected()

        self._capture_vision()
        sf.write(self.audio_buffer_path, audio_full, self.sample_rate)

        self.audio_chunks = []
        if self.on_input_ready:
            self.on_input_ready(self.audio_buffer_path, self.vision_buffer_path)


    def listen_in_background(self, callback, on_wake_word=None) -> None:
        """Entry point for the always-on multimodal listening loop."""
        self.on_input_ready = callback
        self.on_wake_word_detected = on_wake_word
        self.start_multimodal_capture()

    async def speak(self, text: str):
        """
        Synthesize and play audio for the given text using Kokoro TTS.
        
        Employs a gapless sentence-by-sentence streaming producer-consumer loop
        to minimize playback latency while synthesis is in progress.
        """
        if not pygame.mixer.get_init():
            logger.warning("pygame_mixer_not_initialized_skipping_tts", text=text)
            return

        if self.stream:
            self.stream.stop()
            
        # Clean text
        clean_text = re.sub(r'[*`#]', '', text).strip()
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', clean_text) if s.strip()]
        if not sentences:
            return

        logger.info("generating_kokoro_tts_gapless", voice="bf_isabella", sentence_count=len(sentences))
        
        # We use a Queue to pass generated audio samples to the playback loop
        audio_queue = asyncio.Queue()
        
        async def producer():
            """Generates audio samples in the background."""
            try:
                for sentence in sentences:
                    # Generate samples in a thread to not block the event loop
                    samples, sample_rate = await asyncio.to_thread(
                        self.kokoro.create,
                        sentence,
                        voice="bf_isabella",
                        speed=1.0,
                        lang="en-gb"
                    )
                    await audio_queue.put((samples, sample_rate))
                # Signal end of generation
                await audio_queue.put(None)
            except Exception as e:
                logger.error("tts_producer_failed", error=str(e))
                await audio_queue.put(None)

        # Start the background producer
        producer_task = asyncio.create_task(producer())

        try:
            seg_idx = 0
            while True:
                # Get next generated audio set
                item = await audio_queue.get()
                if item is None:
                    break
                    
                samples, sample_rate = item
                audio_file = f"anubis_response_seg_{seg_idx % 2}.wav"
                sf.write(audio_file, samples, sample_rate)
                
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # While this segment is playing, the producer is already working 
                # on the next item and putting it into the queue.
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.01) # High frequency polling for minimal gap
                
                seg_idx += 1
                
            pygame.mixer.music.unload()
        except Exception as e:
            logger.error("kokoro_tts_failed", error=str(e))
        finally:
            await producer_task
            if self.stream:
                self.stream.start()
                self.audio_chunks = []
