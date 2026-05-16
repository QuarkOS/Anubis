# Anubis: A Multimodal Desktop Assistant Platform

Anubis is an end-to-end multimodal platform for desktop-integrated intelligence. It provides a comprehensive orchestration framework that enables Gemini 3.0 Flash Preview to interface directly with local system state—specifically screen context and audio input—allowing developers to build and deploy highly contextual AI-powered tools.

Anubis was developed to bridge the gap between high-reasoning LLMs and the local desktop environment. By utilizing a hybrid architecture of a Python-based backend and a Godot-powered Win32 overlay, the platform is versatile enough to be used for productivity assistance, real-time gaming analysis, and UI/UX research.

The platform provides stable Python APIs for multimodal orchestration and an experimental Godot-based visualization layer for real-time feedback.

## Install

See the Anubis install guide for detailed environment setup. The project utilizes `uv` for high-performance dependency management.

To install the current release and its dependencies:

```bash
uv sync
```

Ensure you have a Google Gemini API key configured in your environment variables or a `.env` file:

```bash
ANUBIS_LLM_API_KEY=your_api_key_here
```

## Try your first Anubis session

```bash
uv run python main_anubis.py
```

Once initialized, press and hold the `F14` key to capture your current screen and audio. Release the key to send the multimodal context to the model and receive a concise voice response.

## Contribution guidelines

If you want to contribute to Anubis, be sure to review the Contribution Guidelines. This project adheres to a professional Code of Conduct. By participating, you are expected to uphold this code.

We use GitHub Issues for tracking requests and bugs. The Anubis project strives to abide by generally accepted best practices in open-source software development.

## Project Structure

*   **src/anubis/**: The core Python orchestration layer handling LLM interop, PTT logic, and TTS.
*   **shader_test/**: A Godot 4.x project providing a transparent Win32 overlay and experimental shaders.
*   **main_anubis.py**: Entry point for the multimodal assistant.

## Resources

*   Google GenAI Documentation
*   Godot Engine Documentation
*   Edge-TTS Repository

## License

Apache License 2.0
