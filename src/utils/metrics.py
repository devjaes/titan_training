"""
Módulo de métricas para evaluación de modelos.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    hamming_loss, accuracy_score, classification_report,
    multilabel_confusion_matrix
)
from typing import Dict, List, Tuple
import torch


def calculate_multilabel_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    threshold: float = 0.5,
    class_names: List[str] = None
) -> Dict[str, float]:
    """
    Calcula métricas para clasificación multi-label.
    
    Args:
        y_true: Labels verdaderos (N, num_classes)
        y_pred: Predicciones (probabilidades) (N, num_classes)
        threshold: Umbral para binarizar predicciones
        class_names: Nombres de las clases
        
    Returns:
        Diccionario con métricas
    """
    # Binarizar predicciones
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    metrics = {
        # Métricas globales
        'hamming_loss': hamming_loss(y_true, y_pred_binary),
        'exact_match_ratio': accuracy_score(y_true, y_pred_binary),
        
        # F1 scores
        'f1_micro': f1_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred_binary, average='weighted', zero_division=0),
        'f1_samples': f1_score(y_true, y_pred_binary, average='samples', zero_division=0),
        
        # Precision y Recall
        'precision_micro': precision_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred_binary, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred_binary, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_binary, average='macro', zero_division=0),
    }
    
    # Métricas por clase (si hay nombres)
    if class_names is not None:
        f1_per_class = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
        for i, name in enumerate(class_names):
            if i < len(f1_per_class):
                metrics[f'f1_{name}'] = f1_per_class[i]
    
    return metrics


def calculate_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """
    Calcula pesos para clases desbalanceadas en multi-label.
    
    Args:
        y_train: Labels de entrenamiento (N, num_classes)
        
    Returns:
        Tensor con pesos positivos para BCEWithLogitsLoss
    """
    # Contar positivos y negativos por clase
    pos_counts = y_train.sum(axis=0)
    neg_counts = y_train.shape[0] - pos_counts
    
    # Calcular pos_weight = neg_count / pos_count
    # Evitar división por cero
    pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    
    # Limitar pesos extremos
    pos_weights = np.clip(pos_weights, 0.1, 10.0)
    
    return torch.tensor(pos_weights, dtype=torch.float32)


def get_predictions_above_threshold(
    predictions: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5
) -> List[List[Tuple[str, float]]]:
    """
    Obtiene las predicciones por encima del umbral con sus probabilidades.
    
    Args:
        predictions: Predicciones (N, num_classes)
        class_names: Nombres de las clases
        threshold: Umbral mínimo
        
    Returns:
        Lista de listas con tuplas (nombre_clase, probabilidad)
    """
    results = []
    for pred in predictions:
        sample_results = []
        for i, (name, prob) in enumerate(zip(class_names, pred)):
            if prob >= threshold:
                sample_results.append((name, float(prob)))
        # Ordenar por probabilidad descendente
        sample_results.sort(key=lambda x: x[1], reverse=True)
        results.append(sample_results)
    return results


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5
) -> str:
    """
    Genera un reporte de clasificación detallado.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Filtrar clases que tienen al menos una muestra positiva
    valid_indices = []
    valid_names = []
    for i, name in enumerate(class_names):
        if y_true[:, i].sum() > 0 or y_pred_binary[:, i].sum() > 0:
            valid_indices.append(i)
            valid_names.append(name)
    
    if len(valid_indices) == 0:
        return "No hay clases con muestras positivas para reportar."
    
    y_true_filtered = y_true[:, valid_indices]
    y_pred_filtered = y_pred_binary[:, valid_indices]
    
    return classification_report(
        y_true_filtered, 
        y_pred_filtered, 
        target_names=valid_names,
        zero_division=0
    )
