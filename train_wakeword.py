"""
Automation script to setup and train custom wake word models for Anubis.
Handles cross-platform absolute path resolution, Windows UTF-8 encoding requirements,
VITS model setup/download, training pipeline execution, and model deployment.
"""

import os
import sys
import shutil
import subprocess
import yaml

def main():
    print("==================================================")
    print("*      ANUBIS WAKE WORD TRAINING AUTOMATION      *")
    print("==================================================")

    # 1. Resolve absolute paths dynamically to prevent Windows relative path bugs
    project_root = os.path.abspath(os.path.dirname(__file__) or ".")
    output_dir = os.path.join(project_root, "output")
    data_dir = os.path.join(project_root, "data")
    wakeword_yaml_path = os.path.join(project_root, "wakeword.yaml")
    
    print(f"Project Root: {project_root}")
    print(f"Data Dir:     {data_dir}")
    print(f"Output Dir:   {output_dir}")

    # 2. Write/Update wakeword.yaml with dynamically resolved absolute paths
    config = {
        "model_name": "hey_anubis",
        "target_phrases": ["hey anubis"],
        "n_samples": 500,
        "n_samples_val": 100,
        "steps": 5000,
        "output_dir": output_dir.replace("\\", "/"),
        "data_dir": data_dir.replace("\\", "/")
    }
    
    print("\n[1/5] Generating/updating wakeword.yaml configuration...")
    try:
        with open(wakeword_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print("[SUCCESS] Created wakeword.yaml with dynamic absolute paths.")
    except Exception as e:
        print(f"[ERROR] Failed to create config file: {e}")
        sys.exit(1)

    # 3. Enable UTF-8 encoding for Python globally to prevent Windows terminal crashes
    os.environ["PYTHONUTF8"] = "1"

    # 4. Run setup command to download the Piper VITS voice model if missing
    print("\n[2/5] Setting up synthetic voice models (Piper VITS)...")
    try:
        # Check if the model is already downloaded to skip setup
        piper_model_path = os.path.join(data_dir, "piper", "en-us-libritts-high.pt")
        if os.path.exists(piper_model_path):
            print("[SUCCESS] Piper VITS model already exists. Skipping download.")
        else:
            print("Downloading voice synthesis models. This may take a minute...")
            subprocess.run(
                ["uv", "run", "livekit-wakeword", "setup", "--config", "wakeword.yaml"],
                check=True
            )
            print("[SUCCESS] Setup complete.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Setup failed during voice model download: {e}")
        sys.exit(1)

    # 5. Run the full training pipeline
    print("\n[3/5] Starting full training pipeline...")
    print("This will synthesize audio clips, augment them, and train the neural classifier.")
    print("This may take 3-5 minutes on a CPU. Please wait...")
    try:
        subprocess.run(
            ["uv", "run", "livekit-wakeword", "run", "wakeword.yaml"],
            check=True
        )
        print("[SUCCESS] Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed. Check logs above: {e}")
        sys.exit(1)

    # 6. Deploy the trained ONNX model
    print("\n[4/5] Deploying trained model to Anubis wake words repository...")
    src_onnx = os.path.join(output_dir, "hey_anubis", "hey_anubis.onnx")
    dest_dir = os.path.join(project_root, "wakewords")
    dest_onnx = os.path.join(dest_dir, "hey_anubis.onnx")
    
    try:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_onnx, dest_onnx)
        print(f"[SUCCESS] Model successfully deployed to: {dest_onnx}")
    except Exception as e:
        print(f"[ERROR] Failed to deploy model: {e}")
        sys.exit(1)

    # 7. Clean up training artifacts while preserving setup assets
    print("\n[5/5] Performing cleanup of intermediate training data...")
    try:
        # We delete output/ and sub-directories of data/ except for data/piper/
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            # Preserve the setup voice assets so we don't have to redownload them next time
            if item == "piper":
                continue
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
                
        # Also clean up the temporary wakeword.yaml since it's generated dynamically
        if os.path.exists(wakeword_yaml_path):
            os.remove(wakeword_yaml_path)
            
        print("[SUCCESS] Intermediate folders and features cleaned.")
    except Exception as e:
        print(f"[WARNING] Cleanup warning: {e}")

    print("\n==================================================")
    print("*  SUCCESS: YOUR CUSTOM WAKE WORD IS DEPLOYED!   *")
    print("*  Run Anubis: uv run python main_anubis.py     *")
    print("==================================================")

if __name__ == "__main__":
    main()
