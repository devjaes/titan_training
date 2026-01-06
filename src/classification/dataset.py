"""
Dataset para clasificación multi-label de atributos de ciclistas.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MultiLabelDataset(Dataset):
    """
    Dataset para clasificación multi-label.
    Estructura esperada del dataset de Roboflow:
    
    dataset/
    ├── train/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── valid/
    ├── test/
    └── _classes.csv
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        label_file: str = '_classes.csv'
    ):
        """
        Inicializa el dataset.
        
        Args:
            data_dir: Directorio raíz del dataset
            split: Split a cargar ('train', 'valid', 'test')
            transform: Transformaciones a aplicar
            label_file: Nombre del archivo de labels
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Cargar labels
        self.labels_df = self._load_labels(label_file)
        
        # Obtener lista de imágenes y sus labels
        self.samples = self._get_samples()
        
        # Nombres de clases (excluyendo filename)
        self.class_names = [col for col in self.labels_df.columns if col != 'filename']
        self.num_classes = len(self.class_names)
        
        print(f"📁 Dataset {split}: {len(self.samples)} imágenes, {self.num_classes} clases")
    
    def _load_labels(self, label_file: str) -> pd.DataFrame:
        """Carga archivo de labels."""
        label_path = self.data_dir / label_file
        
        if not label_path.exists():
            # Buscar en subdirectorios
            for subdir in ['train', 'valid', 'test', '']:
                alt_path = self.data_dir / subdir / label_file
                if alt_path.exists():
                    label_path = alt_path
                    break
        
        if not label_path.exists():
            raise FileNotFoundError(f"No se encontró archivo de labels: {label_file}")
        
        df = pd.read_csv(label_path)
        return df
    
    def _get_samples(self) -> List[Tuple[Path, np.ndarray]]:
        """Obtiene lista de muestras (imagen, labels)."""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {split_dir}")
        
        # Extensiones de imagen soportadas
        extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for img_path in split_dir.iterdir():
            if img_path.suffix.lower() in extensions:
                # Buscar labels para esta imagen
                filename = img_path.name
                
                # Buscar en el DataFrame
                row = self.labels_df[self.labels_df['filename'] == filename]
                
                if len(row) == 0:
                    # Intentar con nombre sin extensión
                    base_name = img_path.stem
                    row = self.labels_df[self.labels_df['filename'].str.startswith(base_name)]
                
                if len(row) > 0:
                    # Obtener labels como array
                    label_cols = [col for col in self.labels_df.columns if col != 'filename']
                    labels = row[label_cols].values[0].astype(np.float32)
                    samples.append((img_path, labels))
        
        if len(samples) == 0:
            print(f"⚠️  Advertencia: No se encontraron muestras válidas en {split_dir}")
            print(f"   Archivos disponibles: {list(split_dir.iterdir())[:5]}...")
            print(f"   Labels disponibles: {self.labels_df['filename'].head().tolist()}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, labels = self.samples[idx]
        
        # Cargar imagen
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        # Convertir labels a tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calcula pesos para clases desbalanceadas.
        
        Returns:
            Tensor con pesos positivos para BCEWithLogitsLoss
        """
        all_labels = np.array([sample[1] for sample in self.samples])
        
        # Contar positivos por clase
        pos_counts = all_labels.sum(axis=0)
        neg_counts = len(all_labels) - pos_counts
        
        # pos_weight = neg / pos
        pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
        pos_weights = np.clip(pos_weights, 0.1, 10.0)
        
        return torch.tensor(pos_weights, dtype=torch.float32)
    
    def get_label_statistics(self) -> Dict[str, Dict]:
        """Obtiene estadísticas de las etiquetas."""
        all_labels = np.array([sample[1] for sample in self.samples])
        
        stats = {
            'total_samples': len(self.samples),
            'num_classes': self.num_classes,
            'avg_labels_per_sample': all_labels.sum(axis=1).mean(),
            'class_frequencies': {},
            'zero_count_classes': []
        }
        
        for i, class_name in enumerate(self.class_names):
            count = int(all_labels[:, i].sum())
            stats['class_frequencies'][class_name] = count
            if count == 0:
                stats['zero_count_classes'].append(class_name)
        
        return stats


def get_transforms(split: str, imgsz: int = 224) -> T.Compose:
    """
    Obtiene transformaciones según el split.
    
    Args:
        split: 'train', 'valid' o 'test'
        imgsz: Tamaño de imagen de salida
        
    Returns:
        Composición de transformaciones
    """
    if split == 'train':
        return T.Compose([
            T.Resize((imgsz + 32, imgsz + 32)),
            T.RandomCrop(imgsz),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((imgsz, imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    imgsz: int = 224,
    num_workers: int = 2,
    label_file: str = '_classes.csv'
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """
    Crea DataLoaders para entrenamiento, validación y test.
    
    Args:
        data_dir: Directorio del dataset
        batch_size: Tamaño de batch
        imgsz: Tamaño de imagen
        num_workers: Número de workers para carga de datos
        label_file: Archivo de labels
        
    Returns:
        Tupla (train_loader, val_loader, test_loader, class_names)
    """
    # Crear datasets
    train_dataset = MultiLabelDataset(
        data_dir, split='train',
        transform=get_transforms('train', imgsz),
        label_file=label_file
    )
    
    val_dataset = MultiLabelDataset(
        data_dir, split='valid',
        transform=get_transforms('valid', imgsz),
        label_file=label_file
    )
    
    # Test es opcional
    test_loader = None
    test_dir = Path(data_dir) / 'test'
    if test_dir.exists() and any(test_dir.iterdir()):
        test_dataset = MultiLabelDataset(
            data_dir, split='test',
            transform=get_transforms('test', imgsz),
            label_file=label_file
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=True
        )
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names
