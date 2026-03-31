import os
import librosa
import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress TensorFlow logging for a clean terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class DeepShieldAudioValidator:
    def __init__(self, model_path):
        print(f"[INFO] Loading DeepShield Audio Model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please check the path.")
        self.model = tf.keras.models.load_model(model_path)
        print("[SUCCESS] Model loaded successfully.\n")

    def preprocess_audio(self, file_path):
        """Replicates the Colab Mel Spectrogram extraction perfectly."""
        sample_rate = 16000
        duration = 3
        max_pad_len = sample_rate * duration

        # 1. Load Audio
        audio, sr = librosa.load(file_path, sr=sample_rate)
        
        # 2. Pad or Truncate
        if len(audio) > max_pad_len:
            audio = audio[:max_pad_len]
        else:
            padding = max_pad_len - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
            
        # 3. Generate Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 4. Normalize
        mel_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-9)
        
        # 5. Resize to 128x128 using Pillow (compliant with requirements.txt)
        mel_img = Image.fromarray((mel_normalized * 255).astype(np.uint8))
        mel_resized = mel_img.resize((128, 128), Image.Resampling.LANCZOS)
        mel_array = np.array(mel_resized) / 255.0
        
        # 6. Stack to 3 Channels and add Batch Dimension
        mel_3ch = np.stack((mel_array,)*3, axis=-1)
        mel_input = np.expand_dims(mel_3ch, axis=0)
        
        return mel_input

    def predict(self, file_path):
        print(f"[INFO] Analyzing audio file: {os.path.basename(file_path)}")
        processed_audio = self.preprocess_audio(file_path)
        
        # Get raw prediction score
        prediction = self.model.predict(processed_audio, verbose=0)[0][0]
        
        # Binary Classification Mapping
        confidence = prediction if prediction > 0.5 else (1 - prediction)
        label = "FAKE (AI Generated)" if prediction > 0.5 else "REAL (Human Voice)"
        
        print(f"--- DEEPSHIELD VERDICT ---")
        print(f"Result:     {label}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Raw Score:  {prediction:.4f}")
        print("-" * 26)

if __name__ == "__main__":
    MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'audio_model.h5')
    TEST_AUDIO_FILE = 'sample_test.wav' 
    AUDIO_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_audio', TEST_AUDIO_FILE)
    
    try:
        validator = DeepShieldAudioValidator(MODEL_PATH)
        validator.predict(AUDIO_PATH)
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")