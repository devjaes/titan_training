"""
Script de entrenamiento para modelo de clasificación multi-label.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR

from .dataset import MultiLabelDataset, get_dataloaders, get_transforms
from .model import MultiLabelClassifier, create_model, get_loss_function

# Importar métricas
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import calculate_multilabel_metrics, calculate_class_weights


class ClassificationTrainer:
    """
    Clase para entrenar modelos de clasificación multi-label.
    """
    
    def __init__(
        self,
        data_path: str,
        config_path: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Inicializa el entrenador.
        
        Args:
            data_path: Ruta al directorio del dataset
            config_path: Ruta al archivo de configuración
            device: Dispositivo ('auto', 'cuda', 'cpu')
        """
        self.data_path = Path(data_path)
        self.config = self._load_config(config_path)
        
        # Configurar dispositivo
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Dispositivo: {self.device}")
        
        # Componentes (se inicializan después)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = None
        
        # Historial de entrenamiento
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1_macro': [],
            'val_f1_micro': [],
            'learning_rate': []
        }
        
        self.best_f1 = 0.0
        self.best_epoch = 0
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Carga configuración desde archivo YAML."""
        default_config = {
            'model': {
                'backbone': 'efficientnet_b2',
                'pretrained': True,
                'dropout': 0.3
            },
            'training': {
                'epochs': 80,
                'batch_size': 32,
                'imgsz': 224,
                'patience': 12,
                'workers': 2,
                'optimizer': 'AdamW',
                'learning_rate': 0.0003,
                'weight_decay': 0.01,
                'scheduler': 'cosine',
                'warmup_epochs': 5,
                'loss': 'BCE',
                'label_smoothing': 0.1,
                'pos_weight_auto': True
            },
            'save': {
                'save_period': 5,
                'project': 'runs/classification',
                'name': 'titan_classification'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            for key in user_config:
                if key in default_config and isinstance(default_config[key], dict):
                    default_config[key].update(user_config[key])
                else:
                    default_config[key] = user_config[key]
        
        return default_config
    
    def setup(self) -> None:
        """Configura todos los componentes para el entrenamiento."""
        train_cfg = self.config['training']
        model_cfg = self.config['model']
        
        print("\n📦 Configurando componentes...")
        
        # 1. Cargar datos
        print("   Cargando datasets...")
        self.train_loader, self.val_loader, self.test_loader, self.class_names = get_dataloaders(
            data_dir=str(self.data_path),
            batch_size=train_cfg['batch_size'],
            imgsz=train_cfg['imgsz'],
            num_workers=train_cfg['workers']
        )
        
        num_classes = len(self.class_names)
        
        # 2. Crear modelo
        print("   Creando modelo...")
        self.model = create_model(
            num_classes=num_classes,
            backbone=model_cfg['backbone'],
            pretrained=model_cfg['pretrained'],
            dropout=model_cfg['dropout'],
            device=str(self.device)
        )
        
        # 3. Configurar loss
        print("   Configurando función de pérdida...")
        pos_weight = None
        if train_cfg.get('pos_weight_auto', True):
            pos_weight = self.train_loader.dataset.get_class_weights()
            pos_weight = pos_weight.to(self.device)
            print(f"      Pesos calculados para {len(pos_weight)} clases")
        
        self.criterion = get_loss_function(
            loss_type=train_cfg['loss'],
            pos_weight=pos_weight
        )
        
        # 4. Configurar optimizador
        print("   Configurando optimizador...")
        if train_cfg['optimizer'] == 'AdamW':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=train_cfg['learning_rate'],
                weight_decay=train_cfg['weight_decay']
            )
        else:
            self.optimizer = SGD(
                self.model.parameters(),
                lr=train_cfg['learning_rate'],
                momentum=0.9,
                weight_decay=train_cfg['weight_decay']
            )
        
        # 5. Configurar scheduler
        print("   Configurando scheduler...")
        if train_cfg['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg['epochs'],
                eta_min=train_cfg['learning_rate'] * 0.01
            )
        elif train_cfg['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        
        print("✅ Configuración completada\n")
    
    def train_epoch(self) -> float:
        """
        Entrena una época.
        
        Returns:
            Loss promedio de la época
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return running_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """
        Ejecuta validación.
        
        Returns:
            Tupla (loss, métricas)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(self.val_loader, desc='Validation', leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Obtener probabilidades
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        # Concatenar predicciones
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Calcular métricas
        metrics = calculate_multilabel_metrics(
            all_labels, all_preds,
            threshold=0.5,
            class_names=self.class_names
        )
        
        avg_loss = running_loss / len(self.val_loader)
        
        return avg_loss, metrics
    
    def train(self, resume_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta el entrenamiento completo.
        
        Args:
            resume_path: Ruta a checkpoint para reanudar
            
        Returns:
            Resultados del entrenamiento
        """
        # Setup si no se ha hecho
        if self.model is None:
            self.setup()
        
        train_cfg = self.config['training']
        save_cfg = self.config['save']
        
        # Crear directorio de guardado
        save_dir = Path(save_cfg['project']) / save_cfg['name']
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Reanudar si se especifica
        start_epoch = 0
        if resume_path and Path(resume_path).exists():
            checkpoint = torch.load(resume_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.best_f1 = checkpoint.get('best_f1', 0)
            print(f"📂 Reanudando desde época {start_epoch}")
        
        print("\n" + "="*60)
        print("🚀 INICIANDO ENTRENAMIENTO DE CLASIFICACIÓN")
        print("="*60)
        print(f"📊 Dataset: {self.data_path}")
        print(f"🔢 Épocas: {train_cfg['epochs']}")
        print(f"📦 Batch size: {train_cfg['batch_size']}")
        print(f"📐 Tamaño imagen: {train_cfg['imgsz']}")
        print(f"🏗️  Backbone: {self.config['model']['backbone']}")
        print(f"💾 Guardando en: {save_dir}")
        print("="*60 + "\n")
        
        patience_counter = 0
        
        for epoch in range(start_epoch, train_cfg['epochs']):
            print(f"\n📅 Época {epoch + 1}/{train_cfg['epochs']}")
            print("-" * 40)
            
            # Entrenar
            train_loss = self.train_epoch()
            
            # Validar
            val_loss, metrics = self.validate()
            
            # Actualizar scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metrics['f1_macro'])
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(metrics['f1_macro'])
            self.history['val_f1_micro'].append(metrics['f1_micro'])
            self.history['learning_rate'].append(current_lr)
            
            # Imprimir métricas
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   F1 Macro:   {metrics['f1_macro']:.4f}")
            print(f"   F1 Micro:   {metrics['f1_micro']:.4f}")
            print(f"   Hamming:    {metrics['hamming_loss']:.4f}")
            print(f"   LR:         {current_lr:.6f}")
            
            # Guardar mejor modelo
            if metrics['f1_macro'] > self.best_f1:
                self.best_f1 = metrics['f1_macro']
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Guardar checkpoint
                self._save_checkpoint(
                    save_dir / 'best.pt',
                    epoch, metrics
                )
                print(f"   ✅ Nuevo mejor modelo guardado! (F1: {self.best_f1:.4f})")
            else:
                patience_counter += 1
            
            # Guardar checkpoint periódico
            if (epoch + 1) % save_cfg['save_period'] == 0:
                self._save_checkpoint(
                    save_dir / f'epoch_{epoch + 1}.pt',
                    epoch, metrics
                )
            
            # Early stopping
            if patience_counter >= train_cfg['patience']:
                print(f"\n⏹️  Early stopping en época {epoch + 1}")
                break
        
        # Guardar modelo final
        self._save_checkpoint(save_dir / 'last.pt', epoch, metrics)
        
        # Guardar historial
        self._save_history(save_dir / 'history.yaml')
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        print(f"   Mejor F1 Macro: {self.best_f1:.4f} (época {self.best_epoch})")
        print(f"   Modelo guardado en: {save_dir / 'best.pt'}")
        print("="*60)
        
        return {
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'model_path': str(save_dir / 'best.pt'),
            'history': self.history
        }
    
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """Guarda checkpoint del modelo."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'metrics': metrics,
            'class_names': self.class_names,
            'config': self.config
        }, path)
    
    def _save_history(self, path: Path) -> None:
        """Guarda historial de entrenamiento."""
        with open(path, 'w') as f:
            yaml.dump(self.history, f)
    
    def export_model(self, format: str = 'onnx') -> str:
        """
        Exporta el modelo a otros formatos.
        
        Args:
            format: 'onnx' o 'torchscript'
            
        Returns:
            Ruta al modelo exportado
        """
        save_dir = Path(self.config['save']['project']) / self.config['save']['name']
        best_path = save_dir / 'best.pt'
        
        if not best_path.exists():
            raise FileNotFoundError("No hay modelo entrenado para exportar")
        
        # Cargar mejor modelo
        checkpoint = torch.load(best_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Dummy input
        dummy_input = torch.randn(1, 3, self.config['training']['imgsz'], 
                                  self.config['training']['imgsz']).to(self.device)
        
        if format == 'onnx':
            export_path = save_dir / 'model.onnx'
            torch.onnx.export(
                self.model,
                dummy_input,
                str(export_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                opset_version=11
            )
        elif format == 'torchscript':
            export_path = save_dir / 'model.pt'
            traced = torch.jit.trace(self.model, dummy_input)
            traced.save(str(export_path))
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        print(f"✅ Modelo exportado: {export_path}")
        return str(export_path)


def main():
    """Función principal para ejecutar desde CLI."""
    parser = argparse.ArgumentParser(description='Entrenar modelo de clasificación multi-label')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Ruta al directorio del dataset')
    parser.add_argument('--config', type=str, default=None,
                        help='Ruta al archivo de configuración YAML')
    parser.add_argument('--resume', type=str, default=None,
                        help='Ruta a checkpoint para reanudar')
    parser.add_argument('--export', type=str, default=None,
                        help='Formato de exportación (onnx, torchscript)')
    
    args = parser.parse_args()
    
    trainer = ClassificationTrainer(
        data_path=args.data_path,
        config_path=args.config
    )
    
    results = trainer.train(resume_path=args.resume)
    
    if args.export:
        trainer.export_model(format=args.export)


if __name__ == "__main__":
    main()
