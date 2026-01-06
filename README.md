# Titan Cyclist Detection & Classification Training

Sistema de entrenamiento de modelos de IA para detección y clasificación automática de ciclistas en competencias de downhill.

## 📋 Descripción

Este proyecto entrena dos modelos complementarios:

1. **Modelo de Detección (YOLOv11)**: Detecta objetos en imágenes de ciclistas (bicicleta, casco, ropa, números de competidor, textos)
2. **Modelo de Clasificación Multi-label**: Clasifica atributos como colores, marcas y números de competidor

## 🏗️ Estructura del Proyecto

```
titan_training/
├── configs/
│   ├── detection_config.yaml    # Configuración del modelo de detección
│   └── classification_config.yaml # Configuración del modelo de clasificación
├── src/
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── train.py             # Entrenamiento YOLOv11
│   │   ├── validate.py          # Validación del modelo
│   │   └── inference.py         # Inferencia con modelo entrenado
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset multi-label
│   │   ├── model.py             # Arquitectura del modelo
│   │   ├── train.py             # Entrenamiento
│   │   └── inference.py         # Inferencia
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py     # Visualización de resultados
│       └── metrics.py           # Métricas de evaluación
├── notebooks/
│   └── titan_training.ipynb     # Notebook principal para Colab
├── requirements.txt
└── README.md
```

## 🚀 Uso en Google Colab

1. Subir los datasets a Google Drive en:
   - `MyDrive/titan_project/models_datasets/titan_detection.yolov11/`
   - `MyDrive/titan_project/models_datasets/titan_labels.multiclass/`

2. Abrir `notebooks/titan_training.ipynb` en Google Colab

3. Ejecutar las celdas en orden

## 📊 Datasets

### Detección (YOLOv11)
- **Imágenes**: 222 (con augmentation)
- **Clases**: 9 (bicycle, bicycle_text, clothes_text, competidor_number, cyclist, cyclist_clothes, cyclist_with_bike, helmet, helmet_text)
- **Formato**: Polígonos en formato YOLO

### Clasificación Multi-label
- **Imágenes**: 240 (con augmentation)
- **Atributos**: 117 (colores de casco, bicicleta, ropa, números, textos de marcas)
- **Formato**: CSV con one-hot encoding

## 🔧 Requisitos

- Python 3.10+
- PyTorch 2.0+
- Ultralytics (YOLOv11)
- torchvision
- pandas
- opencv-python
- matplotlib
- scikit-learn

## 📈 Métricas Esperadas

### Detección
- mAP@0.5: Target > 0.7
- mAP@0.5:0.95: Target > 0.5

### Clasificación
- F1-Score (macro): Target > 0.6
- Hamming Loss: Target < 0.15

## 👥 Equipo

Proyecto desarrollado para automatizar la búsqueda de fotos de ciclistas en eventos de downhill.

## 📝 Licencia

MIT License
