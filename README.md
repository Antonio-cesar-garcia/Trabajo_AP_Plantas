# Comparación de Modelos de Redes Neuronales para Detección de Enfermedades en Plantas

Este proyecto implementa y compara dos arquitecturas de redes neuronales profundas para la clasificación automática de enfermedades en plantas utilizando el dataset [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease).

## Modelos comparados

| Modelo | Archivo | Descripción |
|--------|---------|-------------|
| **CNN Personalizada** | `models/cnn_model.py` | Red convolucional diseñada desde cero con 4 bloques Conv-BN-ReLU-MaxPool |
| **ResNet-50 (Transfer Learning)** | `models/resnet_model.py` | ResNet-50 pre-entrenada en ImageNet con ajuste fino por fases |

## Dataset — PlantVillage

- **54,306 imágenes** de hojas de plantas
- **38 clases** (26 enfermedades + 12 clases sanas)
- **14 especies** de plantas (tomate, patata, manzano, vid, maíz, etc.)

## Estructura del proyecto

```
Trabajo_AP_Plantas/
├── config.py                  # Configuración global (rutas, hiperparámetros, clases)
├── train.py                   # Script de entrenamiento
├── evaluate.py                # Script de evaluación y comparación
├── requirements.txt           # Dependencias Python
├── models/
│   ├── cnn_model.py           # CNN personalizada
│   └── resnet_model.py        # ResNet-50 con transfer learning
├── utils/
│   ├── data_loader.py         # Carga y preprocesamiento del dataset
│   └── visualization.py      # Gráficas de curvas, métricas y matrices de confusión
├── notebooks/
│   └── comparacion_modelos.ipynb  # Notebook de análisis y comparación
├── tests/
│   └── test_models.py         # Tests unitarios (26 tests)
└── results/                   # Checkpoints, historial e imágenes generadas
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### 1. Descargar el dataset

```bash
# Con Kaggle CLI
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d data/
```

### 2. Entrenar los modelos

```bash
# Entrenar ambos modelos
python train.py --data_dir data/PlantVillage --model all --epochs 30

# Solo CNN personalizada
python train.py --data_dir data/PlantVillage --model cnn

# Solo ResNet-50
python train.py --data_dir data/PlantVillage --model resnet
```

### 3. Evaluar y comparar

```bash
python evaluate.py --data_dir data/PlantVillage --results_dir results/
```

### 4. Ejecutar tests

```bash
python -m pytest tests/ -v
```

### 5. Explorar el notebook

```bash
jupyter notebook notebooks/comparacion_modelos.ipynb
```

## Resultados esperados

| Métrica | CNN Personalizada | ResNet-50 (Transfer) |
|---------|:-----------------:|:--------------------:|
| Exactitud | ~87% | ~95% |
| Precisión (macro) | ~87% | ~95% |
| Recall (macro) | ~86% | ~94% |
| F1-Score (macro) | ~86% | ~94% |

> Los valores anteriores son representativos. Los resultados exactos dependen del hardware y la semilla aleatoria.

## Estrategia de entrenamiento

### CNN Personalizada
- Todos los parámetros se aprenden desde cero
- Data augmentation: flip, rotación, jitter de color
- Optimizador: Adam (lr=0.001, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau

### ResNet-50 — Aprendizaje por Transferencia
- **Fase 1** (épocas 1-5): Solo se entrena la capa FC, el backbone está congelado
- **Fase 2** (épocas 6+): Se descongela `layer4` para fine-tuning profundo
- Tasas de aprendizaje diferenciadas: 1e-4 (backbone) / 1e-3 (FC)

## Dependencias principales

- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- scikit-learn ≥ 1.3
- matplotlib, seaborn, pandas, numpy
