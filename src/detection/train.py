"""
Script de entrenamiento para modelo de detección YOLOv11.

Uso:
    python -m src.detection.train --data_path /path/to/dataset --config configs/detection_config.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Importar ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics no está instalado. Ejecute: pip install ultralytics")
    sys.exit(1)


class DetectionTrainer:
    """
    Clase para entrenar modelos de detección YOLOv11.
    """
    
    def __init__(
        self,
        data_path: str,
        config_path: Optional[str] = None,
        model_name: str = "yolo11m.pt"
    ):
        """
        Inicializa el entrenador.
        
        Args:
            data_path: Ruta al directorio del dataset (debe contener data.yaml)
            config_path: Ruta al archivo de configuración
            model_name: Nombre del modelo YOLO a usar
        """
        self.data_path = Path(data_path)
        self.config = self._load_config(config_path)
        self.model_name = model_name
        self.model = None
        self.results = None
        
        # Validar dataset
        self._validate_dataset()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Carga la configuración desde archivo YAML."""
        default_config = {
            'training': {
                'epochs': 100,
                'batch_size': 16,
                'imgsz': 640,
                'patience': 15,
                'workers': 2,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.1,
            },
            'validation': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
            },
            'save': {
                'save_period': 10,
                'project': 'runs/detection',
                'name': 'titan_detection'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge configs
            for key in user_config:
                if key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(user_config[key])
                else:
                    default_config[key] = user_config[key]
                    
        return default_config
    
    def _validate_dataset(self) -> None:
        """Valida que el dataset tenga la estructura correcta."""
        data_yaml = self.data_path / "data.yaml"
        
        if not data_yaml.exists():
            raise FileNotFoundError(f"No se encontró data.yaml en {self.data_path}")
        
        # Verificar directorios
        for split in ['train', 'valid', 'test']:
            split_dir = self.data_path / split
            if split_dir.exists():
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                if images_dir.exists():
                    n_images = len(list(images_dir.glob('*')))
                    n_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
                    print(f"📁 {split}: {n_images} imágenes, {n_labels} labels")
                    
                    if n_images != n_labels:
                        print(f"   ⚠️  Advertencia: número de imágenes y labels no coincide")
            else:
                if split != 'test':
                    print(f"⚠️  Advertencia: directorio {split} no encontrado")
    
    def setup_model(self) -> None:
        """Configura el modelo YOLO."""
        print(f"\n🔧 Cargando modelo: {self.model_name}")
        self.model = YOLO(self.model_name)
        print(f"✅ Modelo cargado correctamente")
    
    def train(self, resume: bool = False) -> Dict[str, Any]:
        """
        Ejecuta el entrenamiento.
        
        Args:
            resume: Si True, reanuda desde el último checkpoint
            
        Returns:
            Resultados del entrenamiento
        """
        if self.model is None:
            self.setup_model()
        
        train_cfg = self.config['training']
        save_cfg = self.config['save']
        
        # Preparar argumentos de entrenamiento
        train_args = {
            'data': str(self.data_path / 'data.yaml'),
            'epochs': train_cfg['epochs'],
            'batch': train_cfg['batch_size'],
            'imgsz': train_cfg['imgsz'],
            'patience': train_cfg['patience'],
            'workers': train_cfg['workers'],
            'optimizer': train_cfg['optimizer'],
            'lr0': train_cfg['lr0'],
            'lrf': train_cfg['lrf'],
            'momentum': train_cfg['momentum'],
            'weight_decay': train_cfg['weight_decay'],
            'warmup_epochs': train_cfg['warmup_epochs'],
            'hsv_h': train_cfg['hsv_h'],
            'hsv_s': train_cfg['hsv_s'],
            'hsv_v': train_cfg['hsv_v'],
            'degrees': train_cfg['degrees'],
            'translate': train_cfg['translate'],
            'scale': train_cfg['scale'],
            'flipud': train_cfg['flipud'],
            'fliplr': train_cfg['fliplr'],
            'mosaic': train_cfg['mosaic'],
            'mixup': train_cfg['mixup'],
            'save_period': save_cfg['save_period'],
            'project': save_cfg['project'],
            'name': save_cfg['name'],
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
            'resume': resume
        }
        
        print("\n" + "="*60)
        print("🚀 INICIANDO ENTRENAMIENTO DE DETECCIÓN")
        print("="*60)
        print(f"📊 Dataset: {self.data_path}")
        print(f"🔢 Épocas: {train_cfg['epochs']}")
        print(f"📦 Batch size: {train_cfg['batch_size']}")
        print(f"📐 Tamaño imagen: {train_cfg['imgsz']}")
        print(f"💾 Guardando en: {save_cfg['project']}/{save_cfg['name']}")
        print("="*60 + "\n")
        
        # Entrenar
        self.results = self.model.train(**train_args)
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        
        return self._get_training_summary()
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Obtiene resumen del entrenamiento."""
        if self.results is None:
            return {}
        
        summary = {
            'model_path': str(self.results.save_dir / 'weights' / 'best.pt'),
            'metrics': {}
        }
        
        # Obtener métricas finales si están disponibles
        if hasattr(self.results, 'results_dict'):
            summary['metrics'] = self.results.results_dict
            
        return summary
    
    def export_model(self, format: str = 'onnx') -> str:
        """
        Exporta el modelo a otros formatos.
        
        Args:
            format: Formato de exportación ('onnx', 'tflite', 'coreml', etc.)
            
        Returns:
            Ruta al modelo exportado
        """
        if self.model is None:
            raise RuntimeError("No hay modelo entrenado para exportar")
        
        best_model_path = self.results.save_dir / 'weights' / 'best.pt'
        model = YOLO(best_model_path)
        
        print(f"\n📤 Exportando modelo a formato: {format}")
        export_path = model.export(format=format)
        print(f"✅ Modelo exportado: {export_path}")
        
        return str(export_path)


def main():
    """Función principal para ejecutar desde CLI."""
    parser = argparse.ArgumentParser(description='Entrenar modelo de detección YOLOv11')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Ruta al directorio del dataset')
    parser.add_argument('--config', type=str, default=None,
                        help='Ruta al archivo de configuración YAML')
    parser.add_argument('--model', type=str, default='yolo11m.pt',
                        help='Modelo YOLO a usar (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt)')
    parser.add_argument('--resume', action='store_true',
                        help='Reanudar entrenamiento desde último checkpoint')
    parser.add_argument('--export', type=str, default=None,
                        help='Formato de exportación después del entrenamiento (onnx, tflite)')
    
    args = parser.parse_args()
    
    # Crear trainer y entrenar
    trainer = DetectionTrainer(
        data_path=args.data_path,
        config_path=args.config,
        model_name=args.model
    )
    
    results = trainer.train(resume=args.resume)
    
    print("\n📊 Resumen del entrenamiento:")
    print(f"   Modelo guardado en: {results.get('model_path', 'N/A')}")
    
    if args.export:
        trainer.export_model(format=args.export)


if __name__ == "__main__":
    main()
