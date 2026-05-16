# Anubis - Voice-First Multimodal AI Assistant

Anubis is a high-performance desktop assistant designed for real-time interaction. It captures your voice and screen context simultaneously to provide intelligent, context-aware assistance during gaming or productivity.

## 🚀 Features

- **Push-to-Talk (PTT):** Triggers only when you want (mapped to `F14`).
- **Multimodal Intelligence:** Uses **Gemini 3 Flash** to "see" your screen and "hear" your voice.
- **Natural Voice Response:** High-quality TTS using **Edge-TTS** with dynamic language detection (German/English).
- **Conversational Memory:** Remembers past context while keeping latency low.

## 🛠️ Installation

1. Install [uv](https://github.com/astral-sh/uv).
2. Clone the repository.
3. Setup environment variables:
   ```bash
   cp .env.example .env
   ```
4. Add your Gemini API Key to `.env`.
5. Run Anubis:
   ```bash
   uv run python main_anubis.py
   ```

## ⌨️ Controls

- **F14 (Hold):** Record voice and capture screen.
- **Release:** Send to Anubis for processing.
