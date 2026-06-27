from pathlib import Path

import cv2
import numpy as np
try:
    import tensorflow as tf
    Adam = tf.keras.optimizers.Adam
    to_categorical = tf.keras.utils.to_categorical
except ImportError:
    try:
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        from keras.optimizers import Adam
        from keras.utils import to_categorical

from Model_architecture.modelArch import DenseArchs
from embedding import emb

import warnings

warnings.filterwarnings("ignore")

PEOPLE_DIR = Path("people")


def _list_people_dirs():
    return [d for d in sorted(PEOPLE_DIR.iterdir()) if d.is_dir()]


def Model_Training():
    """Train the face recognition model"""
    print("=" * 60)
    print("Starting Model Training")
    print("=" * 60)
    
    # Check if people directory exists
    if not PEOPLE_DIR.exists():
        raise RuntimeError(f"Directory '{PEOPLE_DIR}' does not exist. Please enroll students first.")
    
    # List people directories
    people_dirs = _list_people_dirs()
    if not people_dirs:
        raise RuntimeError("No student datasets found in the 'people' directory.\nPlease capture images for at least one student first.")
    
    print(f"Found {len(people_dirs)} student(s) to train:")
    for idx, person_dir in enumerate(people_dirs):
        image_count = len(list(person_dir.glob("*.jpg")))
        print(f"  {idx + 1}. {person_dir.name} ({image_count} images)")
    
    # Check for pre-trained model
    pretrained_path = Path("PreTrained_model/facenet_keras.h5")
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Pre-trained model not found: {pretrained_path}\n"
            "Please ensure the facenet_keras.h5 file exists in PreTrained_model/ directory."
        )
    
    print(f"\nLoading pre-trained embedding model from {pretrained_path}...")
    try:
        embedding_model = emb()
        print("✓ Embedding model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")
    
    print(f"\nCreating model architecture for {len(people_dirs)} classes...")
    try:
        arc = DenseArchs(len(people_dirs))
        face_model = arc.arch()
        print("✓ Model architecture created")
    except Exception as e:
        raise RuntimeError(f"Failed to create model architecture: {e}")

    print("\nProcessing images and extracting embeddings...")
    x_data = []
    y_data = []
    total_images = 0

    for label, person_dir in enumerate(people_dirs):
        person_images = 0
        for image_path in person_dir.glob("*.jpg"):
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"  Warning: Could not read {image_path.name}")
                    continue
                
                img = cv2.resize(img, (160, 160)).astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)
                
                embs = embedding_model.calculate(img)
                x_data.append(embs)
                y_data.append(label)
                person_images += 1
                total_images += 1
                
            except Exception as e:
                print(f"  Warning: Error processing {image_path.name}: {e}")
                continue
        
        print(f"  Processed {person_images} images for {person_dir.name}")

    if not x_data:
        raise RuntimeError(
            "No valid images found for training.\n"
            "Please ensure:\n"
            "• Images are in .jpg format\n"
            "• Images are readable\n"
            "• At least one student has captured images"
        )

    print(f"\n✓ Total images processed: {total_images}")
    print(f"✓ Total embeddings extracted: {len(x_data)}")

    # Prepare training data
    print("\nPreparing training data...")
    X = np.array(x_data, dtype="float32")
    y = to_categorical(y_data, num_classes=len(people_dirs))
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Compile model
    learning_rate = 0.001
    epochs = 120
    batch_size = 16
    
    print(f"\nCompiling model...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    try:
        optimizer = Adam(learning_rate=learning_rate)
        face_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
        print("✓ Model compiled successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to compile model: {e}")

    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print("This may take several minutes. Please wait...\n")
    
    try:
        face_model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=1)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}\n\nPlease check:\n• Sufficient memory available\n• Images are valid\n• TensorFlow/Keras are properly installed")
    
    # Save model
    model_dir = Path("Model")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "Face_recognition.MODEL"
    
    print(f"\nSaving model to {model_path}...")
    try:
        face_model.save(str(model_path))
        print(f"✓ Model saved successfully")
        print(f"  Location: {model_path.absolute()}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")
    
    print("\n" + "=" * 60)
    print("Model Training Complete!")
    print("=" * 60)
    return

