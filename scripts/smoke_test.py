import os
import sys
import logging
from pathlib import Path

# Append src to path for absolute module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

try:
    from src.image_inference import ImageModelInference
    from src.audio_inference import AudioModelInference
except ImportError as e:
    logging.error(f"[CRITICAL FAILURE] Could not import inference modules: {e}")
    sys.exit(1)

def run_smoke_tests():
    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir / "assets" / "data" / "sample_images"
    aud_dir = base_dir / "assets" / "data" / "sample_audio"

    print("\n==============================================")
    print("      DEEPSHIELD SMOKE TEST PROTOCOL")
    print("==============================================")

    # Step 1: Image Model Validation
    print("\n>>> [1/4] Validating Image Inference Module...")
    try:
        img_infer = ImageModelInference()
        img_infer.load_model()
        print("[STATUS: OK] EfficientNetB0 architecture loaded into memory.")
    except Exception as e:
        print(f"[STATUS: FAILED] Image model failed to initialize: {e}")

    # Step 2: Audio Model Validation
    print("\n>>> [2/4] Validating Audio Inference Module...")
    try:
        aud_infer = AudioModelInference()
        aud_infer.load_model()
        print("[STATUS: OK] MobileNetV2 architecture loaded into memory.")
    except Exception as e:
        print(f"[STATUS: FAILED] Audio model failed to initialize: {e}")

    # Step 3: Image Inference Execution
    print("\n>>> [3/4] Executing Image Inference...")
    sample_images = list(img_dir.glob("*.[jp][pn][g]")) if img_dir.exists() else []
    if not sample_images:
        print(f"[WARN] No sample images (.jpg/.png) found in assets/data/sample_images/.")
        print("ACTION: Download 5-10 test images from Colab to this folder to test live prediction.")
    else:
        for img in sample_images[:3]:  # Cap at 3 to prevent RAM exhaustion
            res = img_infer.predict(str(img))
            print(f" -> Inference on {img.name}: {res}")

    # Step 4: Audio Inference Execution
    print("\n>>> [4/4] Executing Audio Inference...")
    sample_audios = list(aud_dir.glob("*.wav")) if aud_dir.exists() else []
    if not sample_audios:
        print(f"[WARN] No sample audio (.wav) found in assets/data/sample_audio/.")
        print("ACTION: Download 5-10 test audio clips from Colab to this folder to test live prediction.")
    else:
        for aud in sample_audios[:3]:  # Cap at 3 to prevent RAM exhaustion
            res = aud_infer.predict(str(aud))
            print(f" -> Inference on {aud.name}: {res}")

    print("\n==============================================")
    print("      SMOKE TEST PROTOCOL COMPLETE")
    print("==============================================\n")

if __name__ == "__main__":
    run_smoke_tests()