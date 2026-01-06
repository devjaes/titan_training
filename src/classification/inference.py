"""
Script de inferencia para modelo de clasificación multi-label.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from .model import MultiLabelClassifier


class ClassificationInference:
    """
    Clase para realizar inferencia con modelos de clasificación multi-label.
    """
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        device: str = 'auto'
    ):
        """
        Inicializa el motor de inferencia.
        
        Args:
            model_path: Ruta al checkpoint del modelo (.pt)
            threshold: Umbral de clasificación
            device: Dispositivo
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        self.threshold = threshold
        
        # Configurar dispositivo
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Cargar modelo
        self._load_model()
        
        # Transformaciones
        self.transform = self._get_transform()
    
    def _load_model(self) -> None:
        """Carga el modelo desde checkpoint."""
        print(f"🔧 Cargando modelo: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extraer configuración
        self.class_names = checkpoint['class_names']
        config = checkpoint['config']
        
        # Crear modelo
        self.model = MultiLabelClassifier(
            num_classes=len(self.class_names),
            backbone=config['model']['backbone'],
            pretrained=False,
            dropout=config['model']['dropout']
        )
        
        # Cargar pesos
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Guardar tamaño de imagen
        self.imgsz = config['training']['imgsz']
        
        print(f"✅ Modelo cargado. Clases: {len(self.class_names)}")
    
    def _get_transform(self) -> T.Compose:
        """Obtiene transformaciones para inferencia."""
        return T.Compose([
            T.Resize((self.imgsz, self.imgsz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def predict_single(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Realiza predicción en una sola imagen.
        
        Args:
            image: Ruta a imagen, array numpy o PIL Image
            
        Returns:
            Diccionario con predicciones
        """
        # Cargar imagen
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')
        
        # Transformar
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predecir
        logits = self.model(img_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        return self._parse_predictions(probs)
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Realiza predicción en múltiples imágenes.
        
        Args:
            images: Lista de imágenes
            batch_size: Tamaño de batch
            
        Returns:
            Lista de diccionarios con predicciones
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Cargar y transformar
            tensors = []
            for img in batch_images:
                if isinstance(img, str):
                    pil_img = Image.open(img).convert('RGB')
                elif isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img).convert('RGB')
                else:
                    pil_img = img.convert('RGB')
                
                tensors.append(self.transform(pil_img))
            
            # Stack y predecir
            batch_tensor = torch.stack(tensors).to(self.device)
            logits = self.model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Parsear resultados
            for prob in probs:
                results.append(self._parse_predictions(prob))
        
        return results
    
    def _parse_predictions(self, probs: np.ndarray) -> Dict[str, Any]:
        """
        Parsea probabilidades a formato estructurado.
        
        Args:
            probs: Array de probabilidades
            
        Returns:
            Diccionario con predicciones
        """
        # Obtener predicciones por encima del umbral
        active_indices = np.where(probs >= self.threshold)[0]
        
        predictions = []
        for idx in active_indices:
            predictions.append({
                'class_name': self.class_names[idx],
                'probability': float(probs[idx])
            })
        
        # Ordenar por probabilidad
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Todas las probabilidades
        all_probs = {
            name: float(prob) 
            for name, prob in zip(self.class_names, probs)
        }
        
        return {
            'predictions': predictions,
            'num_predictions': len(predictions),
            'all_probabilities': all_probs
        }
    
    def get_labels(self, result: Dict[str, Any]) -> List[str]:
        """
        Obtiene lista de etiquetas predichas.
        
        Args:
            result: Resultado de predicción
            
        Returns:
            Lista de nombres de clases predichas
        """
        return [p['class_name'] for p in result['predictions']]
    
    def get_grouped_predictions(
        self,
        result: Dict[str, Any]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Agrupa predicciones por categoría.
        
        Args:
            result: Resultado de predicción
            
        Returns:
            Diccionario con predicciones agrupadas
        """
        groups = {
            'bicycle_colors': [],
            'helmet_colors': [],
            'clothes_colors': [],
            'bicycle_brands': [],
            'clothes_brands': [],
            'helmet_brands': [],
            'competitor_numbers': [],
            'other': []
        }
        
        for pred in result['predictions']:
            name = pred['class_name']
            prob = pred['probability']
            
            if name.startswith('bicycle.'):
                groups['bicycle_colors'].append((name, prob))
            elif name.startswith('helmet.') and not name.startswith('helmet_text'):
                groups['helmet_colors'].append((name, prob))
            elif name.startswith('cyclist_clothes.'):
                groups['clothes_colors'].append((name, prob))
            elif name.startswith('bicycle_text.'):
                groups['bicycle_brands'].append((name, prob))
            elif name.startswith('clothes_text.'):
                groups['clothes_brands'].append((name, prob))
            elif name.startswith('helmet_text.'):
                groups['helmet_brands'].append((name, prob))
            elif name.startswith('competidor_number.'):
                groups['competitor_numbers'].append((name, prob))
            else:
                groups['other'].append((name, prob))
        
        # Filtrar grupos vacíos
        return {k: v for k, v in groups.items() if v}
    
    def format_predictions_text(self, result: Dict[str, Any]) -> str:
        """
        Formatea predicciones como texto legible.
        
        Args:
            result: Resultado de predicción
            
        Returns:
            Texto formateado
        """
        grouped = self.get_grouped_predictions(result)
        
        lines = ["🏷️ Predicciones:"]
        
        group_names = {
            'bicycle_colors': '🚴 Colores de Bicicleta',
            'helmet_colors': '⛑️ Colores de Casco',
            'clothes_colors': '👕 Colores de Ropa',
            'bicycle_brands': '🏭 Marcas de Bicicleta',
            'clothes_brands': '👔 Marcas de Ropa',
            'helmet_brands': '🎿 Marcas de Casco',
            'competitor_numbers': '🔢 Números de Competidor'
        }
        
        for group_key, display_name in group_names.items():
            if group_key in grouped:
                lines.append(f"\n{display_name}:")
                for name, prob in grouped[group_key]:
                    clean_name = name.split('.')[-1]
                    lines.append(f"  - {clean_name}: {prob:.2%}")
        
        return '\n'.join(lines)


def run_inference(
    model_path: str,
    image_source: Union[str, List[str]],
    output_dir: str = 'classification_results',
    threshold: float = 0.5,
    save_json: bool = True
) -> List[Dict[str, Any]]:
    """
    Función de conveniencia para ejecutar inferencia.
    
    Args:
        model_path: Ruta al modelo
        image_source: Imagen, directorio o lista de imágenes
        output_dir: Directorio de salida
        threshold: Umbral de clasificación
        save_json: Guardar JSON con resultados
        
    Returns:
        Lista de resultados
    """
    import json
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Inicializar inferencia
    inference = ClassificationInference(model_path, threshold=threshold)
    
    # Obtener lista de imágenes
    if isinstance(image_source, str):
        source_path = Path(image_source)
        if source_path.is_dir():
            images = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
        else:
            images = [source_path]
    else:
        images = [Path(p) for p in image_source]
    
    print(f"\n🔍 Procesando {len(images)} imágenes...")
    
    # Ejecutar inferencia
    results = inference.predict_batch([str(img) for img in images])
    
    # Guardar resultados
    if save_json:
        export_data = []
        for path, result in zip(images, results):
            export_data.append({
                'image': str(path),
                'labels': inference.get_labels(result),
                'predictions': result['predictions']
            })
        
        json_path = output_path / 'classifications.json'
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Clasificaciones exportadas: {json_path}")
    
    # Mostrar resumen
    print("\n📊 Resumen:")
    for path, result in zip(images[:5], results[:5]):
        print(f"\n📸 {path.name}:")
        print(inference.format_predictions_text(result))
    
    if len(images) > 5:
        print(f"\n... y {len(images) - 5} imágenes más")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inferencia con modelo de clasificación')
    parser.add_argument('--model', type=str, required=True,
                        help='Ruta al modelo')
    parser.add_argument('--source', type=str, required=True,
                        help='Imagen o directorio de imágenes')
    parser.add_argument('--output', type=str, default='classification_results',
                        help='Directorio de salida')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Umbral de clasificación')
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        image_source=args.source,
        output_dir=args.output,
        threshold=args.threshold
    )
