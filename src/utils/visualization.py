"""
Módulo de visualización para resultados de entrenamiento y predicciones.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
from PIL import Image


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> None:
    """
    Grafica el historial de entrenamiento.
    
    Args:
        history: Diccionario con métricas por época
        save_path: Ruta para guardar la figura
        figsize: Tamaño de la figura
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # F1 Score
    if 'val_f1_macro' in history:
        axes[1].plot(history['val_f1_macro'], label='F1 Macro', linewidth=2, color='green')
        if 'val_f1_micro' in history:
            axes[1].plot(history['val_f1_micro'], label='F1 Micro', linewidth=2, color='blue')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Validation F1 Scores')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'learning_rate' in history:
        axes[2].plot(history['learning_rate'], linewidth=2, color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    top_k: int = 30,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Grafica la distribución de clases en el dataset.
    
    Args:
        labels: Array de labels (N, num_classes)
        class_names: Nombres de las clases
        top_k: Mostrar solo las top K clases más frecuentes
        save_path: Ruta para guardar
        figsize: Tamaño de la figura
    """
    # Contar frecuencias
    class_counts = labels.sum(axis=0)
    
    # Crear DataFrame y ordenar
    df = pd.DataFrame({
        'class': class_names,
        'count': class_counts
    }).sort_values('count', ascending=False)
    
    # Top K
    df_top = df.head(top_k)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_top)))
    bars = ax.barh(df_top['class'], df_top['count'], color=colors)
    
    ax.set_xlabel('Número de muestras')
    ax.set_ylabel('Clase')
    ax.set_title(f'Top {top_k} Clases más Frecuentes (Total: {len(class_names)} clases)')
    ax.invert_yaxis()
    
    # Añadir valores en las barras
    for bar, count in zip(bars, df_top['count']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{int(count)}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Mostrar estadísticas
    print(f"\n📊 Estadísticas del Dataset:")
    print(f"   Total de clases: {len(class_names)}")
    print(f"   Clases con 0 muestras: {(class_counts == 0).sum()}")
    print(f"   Clase más frecuente: {df.iloc[0]['class']} ({int(df.iloc[0]['count'])} muestras)")
    print(f"   Clase menos frecuente (>0): {df[df['count'] > 0].iloc[-1]['class']} ({int(df[df['count'] > 0].iloc[-1]['count'])} muestras)")
    print(f"   Promedio de labels por imagen: {labels.sum(axis=1).mean():.2f}")


def plot_confusion_matrix_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    top_k: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Grafica matrices de confusión para las clases más importantes.
    """
    from sklearn.metrics import multilabel_confusion_matrix
    
    # Binarizar predicciones
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # Obtener clases más frecuentes
    class_counts = y_true.sum(axis=0)
    top_indices = np.argsort(class_counts)[-top_k:][::-1]
    
    # Calcular métricas por clase
    mcm = multilabel_confusion_matrix(y_true, y_pred_binary)
    
    # Crear subplots
    n_cols = 5
    n_rows = (top_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, class_idx in enumerate(top_indices):
        if idx >= len(axes):
            break
            
        cm = mcm[class_idx]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Pred 0', 'Pred 1'],
                   yticklabels=['True 0', 'True 1'])
        axes[idx].set_title(class_names[class_idx][:15], fontsize=8)
    
    # Ocultar axes vacíos
    for idx in range(len(top_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Matrices de Confusión - Top Clases', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def visualize_detection_results(
    image_path: str,
    results,
    class_names: List[str],
    conf_threshold: float = 0.25,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualiza resultados de detección YOLO en una imagen.
    
    Args:
        image_path: Ruta a la imagen
        results: Resultados de YOLO
        class_names: Nombres de las clases
        conf_threshold: Umbral de confianza
        save_path: Ruta para guardar
        
    Returns:
        Imagen con anotaciones
    """
    # Leer imagen
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Colores para cada clase
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    colors = (colors[:, :3] * 255).astype(int)
    
    # Dibujar detecciones
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                if box.conf >= conf_threshold:
                    # Coordenadas
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls)
                    conf = float(box.conf)
                    
                    # Color
                    color = tuple(map(int, colors[cls % len(colors)]))
                    
                    # Dibujar rectángulo
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Label
                    label = f"{class_names[cls]}: {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Mostrar
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Detección de Objetos')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return img


def visualize_classification_results(
    image_path: str,
    predictions: np.ndarray,
    class_names: List[str],
    threshold: float = 0.5,
    top_k: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza resultados de clasificación multi-label.
    """
    # Leer imagen
    img = Image.open(image_path)
    
    # Obtener predicciones por encima del umbral
    pred_indices = np.where(predictions >= threshold)[0]
    pred_probs = predictions[pred_indices]
    pred_names = [class_names[i] for i in pred_indices]
    
    # Ordenar por probabilidad
    sorted_idx = np.argsort(pred_probs)[::-1][:top_k]
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Imagen
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Imagen Original')
    
    # Predicciones
    if len(sorted_idx) > 0:
        names = [pred_names[i] for i in sorted_idx]
        probs = [pred_probs[i] for i in sorted_idx]
        
        colors = plt.cm.RdYlGn(np.array(probs))
        ax2.barh(names, probs, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Probabilidad')
        ax2.set_title(f'Top {len(names)} Predicciones (umbral={threshold})')
        ax2.invert_yaxis()
        
        for i, (name, prob) in enumerate(zip(names, probs)):
            ax2.text(prob + 0.02, i, f'{prob:.2f}', va='center')
    else:
        ax2.text(0.5, 0.5, 'No hay predicciones\npor encima del umbral',
                ha='center', va='center', fontsize=12)
        ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
