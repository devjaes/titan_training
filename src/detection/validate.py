"""
Script de validación para modelo de detección YOLOv11.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics no está instalado. Ejecute: pip install ultralytics")


class DetectionValidator:
    """
    Clase para validar modelos de detección YOLOv11.
    """
    
    def __init__(self, model_path: str):
        """
        Inicializa el validador.
        
        Args:
            model_path: Ruta al modelo entrenado (.pt)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.results = None
        
    def validate(
        self,
        data_path: str,
        split: str = 'test',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        batch_size: int = 16,
        save_json: bool = True,
        save_plots: bool = True,
        project: str = 'runs/validation',
        name: str = 'titan_validation'
    ) -> Dict[str, Any]:
        """
        Ejecuta validación en un split del dataset.
        
        Args:
            data_path: Ruta al directorio del dataset
            split: Split a validar ('train', 'valid', 'test')
            conf_threshold: Umbral de confianza
            iou_threshold: Umbral de IoU para NMS
            imgsz: Tamaño de imagen
            batch_size: Tamaño de batch
            save_json: Guardar resultados en JSON (para COCO eval)
            save_plots: Guardar gráficos de métricas
            project: Directorio del proyecto
            name: Nombre del experimento
            
        Returns:
            Diccionario con métricas de validación
        """
        data_yaml = Path(data_path) / 'data.yaml'
        
        if not data_yaml.exists():
            raise FileNotFoundError(f"No se encontró data.yaml en {data_path}")
        
        print("\n" + "="*60)
        print(f"🔍 VALIDANDO MODELO EN SPLIT: {split.upper()}")
        print("="*60)
        print(f"📊 Dataset: {data_path}")
        print(f"🎯 Umbral confianza: {conf_threshold}")
        print(f"📐 Tamaño imagen: {imgsz}")
        print("="*60 + "\n")
        
        # Ejecutar validación
        self.results = self.model.val(
            data=str(data_yaml),
            split=split,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            batch=batch_size,
            save_json=save_json,
            plots=save_plots,
            project=project,
            name=name,
            exist_ok=True
        )
        
        return self._get_validation_summary()
    
    def _get_validation_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la validación."""
        if self.results is None:
            return {}
        
        summary = {
            'mAP50': float(self.results.box.map50),
            'mAP50-95': float(self.results.box.map),
            'precision': float(self.results.box.mp),
            'recall': float(self.results.box.mr),
        }
        
        # Métricas por clase
        if hasattr(self.results.box, 'ap_class_index'):
            class_names = self.results.names
            ap50_per_class = self.results.box.ap50
            
            summary['per_class'] = {}
            for idx, ap in zip(self.results.box.ap_class_index, ap50_per_class):
                class_name = class_names[idx]
                summary['per_class'][class_name] = float(ap)
        
        return summary
    
    def print_results(self) -> None:
        """Imprime resultados de validación de forma legible."""
        if self.results is None:
            print("No hay resultados de validación disponibles.")
            return
        
        summary = self._get_validation_summary()
        
        print("\n" + "="*60)
        print("📊 RESULTADOS DE VALIDACIÓN")
        print("="*60)
        print(f"\n🎯 Métricas Globales:")
        print(f"   mAP@0.5:      {summary['mAP50']:.4f}")
        print(f"   mAP@0.5:0.95: {summary['mAP50-95']:.4f}")
        print(f"   Precision:    {summary['precision']:.4f}")
        print(f"   Recall:       {summary['recall']:.4f}")
        
        if 'per_class' in summary:
            print(f"\n📋 mAP@0.5 por Clase:")
            for class_name, ap in sorted(summary['per_class'].items(), key=lambda x: x[1], reverse=True):
                print(f"   {class_name:25s}: {ap:.4f}")
        
        print("\n" + "="*60)
    
    def get_class_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Análisis detallado por clase.
        
        Returns:
            Diccionario con análisis por clase
        """
        if self.results is None:
            return {}
        
        analysis = {}
        class_names = self.results.names
        
        # Precision y Recall por clase
        if hasattr(self.results.box, 'p') and hasattr(self.results.box, 'r'):
            for idx in self.results.box.ap_class_index:
                class_name = class_names[idx]
                analysis[class_name] = {
                    'precision': float(self.results.box.p[idx]) if idx < len(self.results.box.p) else 0,
                    'recall': float(self.results.box.r[idx]) if idx < len(self.results.box.r) else 0,
                    'ap50': float(self.results.box.ap50[idx]) if idx < len(self.results.box.ap50) else 0,
                }
        
        return analysis


def validate_model(
    model_path: str,
    data_path: str,
    split: str = 'test',
    **kwargs
) -> Dict[str, Any]:
    """
    Función de conveniencia para validar un modelo.
    
    Args:
        model_path: Ruta al modelo
        data_path: Ruta al dataset
        split: Split a validar
        **kwargs: Argumentos adicionales para validate()
        
    Returns:
        Resultados de validación
    """
    validator = DetectionValidator(model_path)
    results = validator.validate(data_path, split=split, **kwargs)
    validator.print_results()
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validar modelo de detección')
    parser.add_argument('--model', type=str, required=True,
                        help='Ruta al modelo .pt')
    parser.add_argument('--data', type=str, required=True,
                        help='Ruta al dataset')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'valid', 'test'],
                        help='Split a validar')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Umbral de confianza')
    
    args = parser.parse_args()
    
    validate_model(
        model_path=args.model,
        data_path=args.data,
        split=args.split,
        conf_threshold=args.conf
    )
