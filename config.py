"""
Configuración global del proyecto de detección de enfermedades en plantas.
"""

import os

# ─── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# ─── Dataset ─────────────────────────────────────────────────────────────────
# Proporción train / validación / test
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Número de clases del dataset PlantVillage (38 clases)
NUM_CLASSES = 38

# Tamaño de imagen de entrada (píxeles)
IMAGE_SIZE = 224

# ─── Entrenamiento ───────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Paciencia para Early Stopping
EARLY_STOPPING_PATIENCE = 5

# Semilla para reproducibilidad
RANDOM_SEED = 42

# Usar GPU si está disponible
DEVICE = "cuda"  # se detecta automáticamente en train.py / evaluate.py

# ─── Nombres de los modelos ───────────────────────────────────────────────────
MODEL_CNN = "cnn_custom"
MODEL_RESNET = "resnet50_transfer"

# ─── Clases PlantVillage ─────────────────────────────────────────────────────
PLANT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
