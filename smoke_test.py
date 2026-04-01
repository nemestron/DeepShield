import os
import sys
import warnings
from pathlib import Path

# Suppress TF logging noise for clean output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath("src"))

from image_inference import ImageDeepfakeDetector
from audio_inference import AudioDeepfakeDetector

def run_smoke_tests():
    print("\n--------------------------------------------------")
    print("[SYSTEM] Initiating Patched Inference Smoke Tests")
    print("--------------------------------------------------")
    
    img_dir = Path("assets/sample_images")
    audio_dir = Path("assets/sample_audio")
    
    sample_img = next(img_dir.glob("*.*"), None)
    sample_audio = next(audio_dir.glob("*.*"), None)

    try:
        print(f"[TEST] Booting Image Pipeline targeting: {sample_img.name}")
        img_detector = ImageDeepfakeDetector()
        img_result = img_detector.predict(str(sample_img))
        print(f"[SUCCESS] Image Prediction: {img_result}")
    except Exception as e:
        print(f"[FATAL] Image Pipeline crashed. Exception: {e}")

    try:
        print(f"\n[TEST] Booting Audio Pipeline targeting: {sample_audio.name}")
        audio_detector = AudioDeepfakeDetector()
        audio_result = audio_detector.predict(str(sample_audio))
        print(f"[SUCCESS] Audio Prediction: {audio_result}")
    except Exception as e:
        print(f"[FATAL] Audio Pipeline crashed. Exception: {e}")

if __name__ == "__main__":
    run_smoke_tests()