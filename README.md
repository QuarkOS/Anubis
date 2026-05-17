# Anubis: A Multimodal Desktop Assistant Platform

Anubis is an end-to-end multimodal platform for desktop-integrated intelligence. It provides a comprehensive orchestration framework that enables Gemini 3.0 Flash Preview to interface directly with local system state—specifically screen context, audio input, and real-time telemetry—allowing developers to build and deploy highly contextual AI-powered tools.

## Key Features

*   **Hands-Free Wake-Word:** Always-on background listening with robust fuzzy matching for "Hey Anubis".
*   **Manual Hotkey (F14):** Immediate multimodal capture using the `F14` key for precise control in any environment.
*   **System Situational Awareness:** Real-time monitoring of CPU, GPU (NVIDIA), RAM, Network throughput, and Clipboard content.
*   **Local Multimodal Pipeline:** Utilizes Silero VAD, Faster-Whisper, and Kokoro TTS for low-latency local processing.
*   **Multi-Monitor Vision:** Captures all connected displays to provide full visual context to the model.

## Install

The project utilizes `uv` for high-performance dependency management.

```bash
uv sync
```

Ensure you have a Google Gemini API key configured in your environment variables or a `.env` file:

```bash
ANUBIS_LLM_API_KEY=your_api_key_here
```

## Usage

### Start Anubis

```bash
uv run python main_anubis.py
```

### Interactions

Anubis supports two primary modes of interaction:

1.  **Voice Activation:** Simply say **"Hey Anubis"** followed by your request. Anubis uses fuzzy phonetic matching to ensure it hears you even in noisy environments.
2.  **Manual Trigger (F14):** Press the **`F14`** key to immediately capture your current screen and the last 15 seconds of audio. This is ideal for quick "What's this?" questions or when you prefer not to use the wake word.

Anubis will respond with a concise voice reply through the Kokoro TTS engine.

### Training a Custom Wake Word ("Hey Anubis")

If you want to retrain or update the wake word system, Anubis includes a fully automated training utility script (`train_wakeword.py`). This script dynamically resolves absolute paths, configures the environment (handling Windows encoding quirks), downloads required voice synthesis dependencies (Piper VITS), synthesizes audio datasets, trains the neural classifier, deploys the finalized model, and cleans up the workspace automatically.

To train the custom model:
```bash
uv run python train_wakeword.py
```

This single command handles the entire 6-step machine learning pipeline (approx. 3-5 minutes on CPU). Once finished, your custom model is loaded instantly on the next run!

## Project Structure

*   **src/anubis/**: Core orchestration layer handling LLM interop, VAD/Wake-word logic, and TTS.
*   **src/anubis/repositories/system_windows.py**: Windows-specific telemetry probe for hardware and user context.
*   **tests/**: Comprehensive test suite for system awareness and agent logic.
*   **main_anubis.py**: Entry point for the multimodal assistant.

## Engineering Standards

This project follows the senior-level guidelines documented in [COMMENTING_STANDARDS.md](./COMMENTING_STANDARDS.md), prioritizing naming clarity and concise, high-signal documentation.

## License

MIT License
