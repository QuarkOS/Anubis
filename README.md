# Anubis

A multimodal desktop assistant that combines a Python/FastAPI backend with a Godot-based transparent overlay. 

I built this because I wanted a way to interact with an LLM that could see my screen and hear me without having to alt-tab or use a browser. It uses Gemini Flash 2.0 for the heavy lifting (vision/voice reasoning) and Edge-TTS for the voice output.

The visual component is handled via a Win32 overlay powered by Godot, which lets me experiment with custom shaders for the avatar/UI feedback.

## 🛠️ How it works

- **PTT Trigger:** Mapped to `F14`. Holding it down triggers a screen capture and audio recording.
- **Multimodal Context:** On release, the captured png and wav are sent to the Gemini 2.0 Flash model.
- **Backend:** FastAPI handles the orchestration between recording, LLM calls, and TTS.
- **Overlay:** A Godot project (`shader_test`) provides a transparent window with various shader experiments for visual feedback.

## 🚀 Setup

1. **Install Dependencies:**
   This project uses [uv](https://github.com/astral-sh/uv) for Python management.
   ```bash
   cd ai-assistant
   uv sync
   ```

2. **Configuration:**
   Copy the example env file and add your Gemini API key:
   ```bash
   cp .env.example .env
   ```

3. **Run:**
   ```bash
   uv run python main_anubis.py
   ```

## ⌨️ Controls

- **F14 (Hold):** Record voice + capture screen.
- **Release:** Send to Anubis.

---

*Note: This is a personal project and very much a work-in-progress. The Godot shaders in `shader_test` are experiments for a future UI/avatar integration.*
