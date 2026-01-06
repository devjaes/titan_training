"""
Script de inferencia para modelo de detección YOLOv11.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
import cv2
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics no está instalado. Ejecute: pip install ultralytics")


class DetectionInference:
    """
    Clase para realizar inferencia con modelos de detección YOLOv11.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        Inicializa el motor de inferencia.
        
        Args:
            model_path: Ruta al modelo entrenado (.pt, .onnx, .tflite)
            conf_threshold: Umbral de confianza
            iou_threshold: Umbral de IoU para NMS
            device: Dispositivo ('auto', 'cpu', 'cuda', '0', etc.)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        print(f"🔧 Cargando modelo: {model_path}")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        print(f"✅ Modelo cargado. Clases: {len(self.class_names)}")
    
    def predict_single(
        self,
        image: Union[str, np.ndarray, Image.Image],
        imgsz: int = 640,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Realiza predicción en una sola imagen.
        
        Args:
            image: Ruta a imagen, array numpy o PIL Image
            imgsz: Tamaño de imagen para inferencia
            verbose: Mostrar información detallada
            
        Returns:
            Diccionario con detecciones
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=verbose
        )
        
        return self._parse_results(results[0])
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        imgsz: int = 640,
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Realiza predicción en múltiples imágenes.
        
        Args:
            images: Lista de rutas o arrays
            imgsz: Tamaño de imagen
            batch_size: Tamaño de batch
            
        Returns:
            Lista de diccionarios con detecciones
        """
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            results = self.model.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=imgsz,
                device=self.device,
                verbose=False
            )
            
            for result in results:
                all_results.append(self._parse_results(result))
        
        return all_results
    
    def _parse_results(self, result) -> Dict[str, Any]:
        """
        Parsea resultados de YOLO a formato estructurado.
        
        Args:
            result: Resultado de YOLO para una imagen
            
        Returns:
            Diccionario con detecciones parseadas
        """
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Coordenadas
                xyxy = boxes.xyxy[i].cpu().numpy()
                xywh = boxes.xywh[i].cpu().numpy()
                
                # Clase y confianza
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                detection = {
                    'class_id': cls,
                    'class_name': self.class_names[cls],
                    'confidence': conf,
                    'bbox_xyxy': xyxy.tolist(),  # [x1, y1, x2, y2]
                    'bbox_xywh': xywh.tolist(),  # [cx, cy, w, h]
                }
                
                # Si hay segmentación (polígonos)
                if result.masks is not None and i < len(result.masks):
                    mask = result.masks[i]
                    if hasattr(mask, 'xy') and mask.xy is not None:
                        detection['polygon'] = mask.xy[0].tolist() if len(mask.xy) > 0 else []
                
                detections.append(detection)
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'image_shape': result.orig_shape,
        }
    
    def get_labels_for_image(self, result: Dict[str, Any]) -> List[str]:
        """
        Obtiene lista de etiquetas únicas detectadas.
        
        Args:
            result: Resultado de predicción
            
        Returns:
            Lista de nombres de clases detectadas
        """
        labels = set()
        for det in result['detections']:
            labels.add(det['class_name'])
        return sorted(list(labels))
    
    def draw_detections(
        self,
        image: Union[str, np.ndarray],
        result: Dict[str, Any],
        draw_labels: bool = True,
        draw_conf: bool = True,
        line_width: int = 2
    ) -> np.ndarray:
        """
        Dibuja detecciones en una imagen.
        
        Args:
            image: Imagen (ruta o array)
            result: Resultado de predicción
            draw_labels: Dibujar nombres de clase
            draw_conf: Dibujar confianza
            line_width: Grosor de línea
            
        Returns:
            Imagen con detecciones dibujadas
        """
        # Leer imagen si es ruta
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        # Colores por clase
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)
        
        for det in result['detections']:
            x1, y1, x2, y2 = map(int, det['bbox_xyxy'])
            cls_id = det['class_id']
            color = tuple(map(int, colors[cls_id]))
            
            # Dibujar rectángulo
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
            
            # Dibujar label
            if draw_labels:
                label = det['class_name']
                if draw_conf:
                    label += f" {det['confidence']:.2f}"
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def export_detections_to_json(
        self,
        results: List[Dict[str, Any]],
        image_paths: List[str],
        output_path: str
    ) -> None:
        """
        Exporta detecciones a archivo JSON.
        
        Args:
            results: Lista de resultados
            image_paths: Lista de rutas de imágenes
            output_path: Ruta de salida
        """
        import json
        
        export_data = []
        for path, result in zip(image_paths, results):
            export_data.append({
                'image': str(path),
                'detections': result['detections'],
                'labels': self.get_labels_for_image(result)
            })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Detecciones exportadas a: {output_path}")


def run_inference(
    model_path: str,
    image_source: Union[str, List[str]],
    output_dir: str = 'inference_results',
    save_images: bool = True,
    save_json: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Función de conveniencia para ejecutar inferencia.
    
    Args:
        model_path: Ruta al modelo
        image_source: Imagen, directorio o lista de imágenes
        output_dir: Directorio de salida
        save_images: Guardar imágenes con detecciones
        save_json: Guardar JSON con resultados
        **kwargs: Argumentos para DetectionInference
        
    Returns:
        Lista de resultados
    """
    import os
    from pathlib import Path
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Inicializar inferencia
    inference = DetectionInference(model_path, **kwargs)
    
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
    if save_images:
        for img_path, result in zip(images, results):
            annotated = inference.draw_detections(str(img_path), result)
            out_path = output_path / f"det_{img_path.name}"
            cv2.imwrite(str(out_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"✅ Imágenes guardadas en: {output_path}")
    
    if save_json:
        json_path = output_path / 'detections.json'
        inference.export_detections_to_json(results, [str(p) for p in images], str(json_path))
    
    # Resumen
    total_detections = sum(r['num_detections'] for r in results)
    print(f"\n📊 Resumen: {total_detections} detecciones en {len(images)} imágenes")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inferencia con modelo de detección')
    parser.add_argument('--model', type=str, required=True,
                        help='Ruta al modelo')
    parser.add_argument('--source', type=str, required=True,
                        help='Imagen o directorio de imágenes')
    parser.add_argument('--output', type=str, default='inference_results',
                        help='Directorio de salida')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Umbral de confianza')
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        image_source=args.source,
        output_dir=args.output,
        conf_threshold=args.conf
    )
