from pathlib import Path
try:
    import tensorflow as tf
    load_model = tf.keras.models.load_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

class emb:
    def __init__(self):
        model_path = Path('PreTrained_model/facenet_keras.h5')
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pre-trained model not found: {model_path.absolute()}\n"
                "Please ensure facenet_keras.h5 exists in PreTrained_model/ directory."
            )
        try:
            self.model = load_model(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-trained model: {e}")
        
    def calculate(self, img):
        try:
            return self.model.predict(img, verbose=0)[0]
        except Exception as e:
            raise RuntimeError(f"Error during embedding calculation: {e}")
