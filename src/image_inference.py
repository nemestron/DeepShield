import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageModelInference:
    def __init__(self, model_name="image_model.h5"):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = self.base_dir / "models" / model_name
        self.model = None
        self.target_size = (224, 224)

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Critical: Model not found at {self.model_path}")
        
        logging.info(f"Loading image model from {self.model_path}...")
        self.model = tf.keras.models.load_model(str(self.model_path))
        logging.info("Image model loaded successfully.")

    def predict(self, image_path):
        img_path_obj = Path(image_path)
        if not img_path_obj.exists():
            raise FileNotFoundError(f"Input image not found: {img_path_obj}")
        
        if self.model is None:
            self.load_model()
            
        try:
            img = tf.keras.utils.load_img(str(img_path_obj), target_size=self.target_size)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array)
            prediction = self.model.predict(processed_img, verbose=0)
            confidence = float(prediction[0][0])
            label = "Fake" if confidence > 0.5 else "Real"
            
            return {"status": "success", "prediction": label, "confidence": confidence}
        except Exception as e:
            logging.error(f"Inference failed for {image_path}: {str(e)}")
            return {"status": "error", "message": str(e)}