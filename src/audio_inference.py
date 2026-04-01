import os
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioModelInference:
    def __init__(self, model_name="audio_model.h5"):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = self.base_dir / "models" / model_name
        self.model = None
        self.target_shape = (128, 128)
        self.duration = 3.0
        self.sample_rate = 16000

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Critical: Model not found at {self.model_path}")
        
        logging.info(f"Loading audio model from {self.model_path}...")
        self.model = tf.keras.models.load_model(str(self.model_path))
        logging.info("Audio model loaded successfully.")

    def preprocess_audio(self, audio_path):
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate, duration=self.duration)
        target_length = int(self.sample_rate * self.duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        mel_spec_resized = tf.image.resize(mel_spec_db, self.target_shape).numpy()
        mel_spec_rgb = np.repeat(mel_spec_resized, 3, axis=-1)
        processed = tf.keras.applications.mobilenet_v2.preprocess_input(mel_spec_rgb)
        return np.expand_dims(processed, axis=0)

    def predict(self, audio_path):
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Input audio not found: {audio_path_obj}")
        
        if self.model is None:
            self.load_model()
            
        try:
            processed_audio = self.preprocess_audio(audio_path_obj)
            prediction = self.model.predict(processed_audio, verbose=0)
            confidence = float(prediction[0][0])
            label = "Fake" if confidence > 0.5 else "Real"
            
            return {"status": "success", "prediction": label, "confidence": confidence}
        except Exception as e:
            logging.error(f"Inference failed for {audio_path}: {str(e)}")
            return {"status": "error", "message": str(e)}