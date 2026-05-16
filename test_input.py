import time
import io
from PIL import ImageGrab

try:
    import keyboard
except ImportError:
    print("Please run: uv pip install keyboard mss sounddevice soundfile numpy")
    exit()

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    print("Please run: uv pip install sounddevice soundfile numpy")
    exit()


# Configuration
HOTKEY = 'alt+space'
SAMPLE_RATE = 44100
CHANNELS = 1

print(f"==================================================")
print(f"🎙️ Anubis Input Test")
print(f"Hold down '{HOTKEY}' to record. Release to stop.")
print(f"Press 'esc' to exit this test script.")
print(f"==================================================\n")

audio_data = []
recording = False
stream = None

def callback(indata, frames, time, status):
    """This is called for each audio block by sounddevice."""
    if status:
        print(status)
    if recording:
        audio_data.append(indata.copy())

def start_recording(e):
    global recording, stream, audio_data
    if not recording:
        print("\n[REC] Recording started... (Taking screenshot)")
        recording = True
        audio_data = []
        
        # Take a screenshot right when the user starts asking the question
        screenshot = ImageGrab.grab()
        screenshot.save("anubis_vision_context.png")
        print(f"[IMG] Screenshot saved to 'anubis_vision_context.png'")
        
        # Start Audio Stream
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback)
        stream.start()

def stop_recording(e):
    global recording, stream, audio_data
    if recording:
        recording = False
        if stream:
            stream.stop()
            stream.close()
            stream = None
            
        print("[REC] Recording stopped. Processing audio...")
        
        if audio_data:
            # Concatenate all recorded chunks
            final_audio = np.concatenate(audio_data, axis=0)
            
            # Save to a temporary WAV file
            filename = "anubis_audio_context.wav"
            sf.write(filename, final_audio, SAMPLE_RATE)
            print(f"[AUD] Audio saved to '{filename}' (Duration: {len(final_audio) / SAMPLE_RATE:.2f} seconds)")
            print(f"\n=> You can now send these files to Gemini! Ready for next recording. (Hold {HOTKEY})")
        else:
            print("[AUD] No audio recorded.")

# Register Hotkey Callbacks
keyboard.on_press_key('space', lambda e: start_recording(e) if keyboard.is_pressed('alt') else None)
keyboard.on_release_key('space', lambda e: stop_recording(e))

# Keep script running until 'esc' is pressed
keyboard.wait('esc')
print("Exiting test script.")
