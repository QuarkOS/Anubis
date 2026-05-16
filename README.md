# Anubis: A Multimodal Desktop Assistant Platform

Anubis is an end-to-end multimodal platform for desktop-integrated intelligence. It provides a comprehensive orchestration framework that enables Gemini 3.0 Flash Preview to interface directly with local system state—specifically screen context and audio input—allowing developers to build and deploy highly contextual AI-powered tools.

Anubis was developed to bridge the gap between high-reasoning LLMs and the local desktop environment. By utilizing a high-performance Python orchestration layer, the platform is versatile enough to be used for productivity assistance, real-time gaming analysis, and general desktop automation.

The platform provides stable Python APIs for multimodal orchestration. Experimental UI layers using Godot are available in the `experimental` branch.

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

*   **src/anubis/**: Core orchestration layer handling LLM interop, PTT logic, and TTS.
*   **tests/**: Comprehensive test suite for the domain and service layers.
*   **main_anubis.py**: Entry point for the multimodal assistant.

## Resources

*   Google GenAI Documentation
*   Godot Engine Documentation
*   Edge-TTS Repository

## License

MIT License
