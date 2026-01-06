"""
Modelo de clasificación multi-label para atributos de ciclistas.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import torchvision.models as models


class MultiLabelClassifier(nn.Module):
    """
    Clasificador multi-label basado en backbones preentrenados.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'efficientnet_b2',
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Inicializa el clasificador.
        
        Args:
            num_classes: Número de clases (atributos)
            backbone: Arquitectura base
            pretrained: Usar pesos preentrenados
            dropout: Tasa de dropout
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Crear backbone
        self.backbone, num_features = self._create_backbone(backbone, pretrained)
        
        # Cabeza de clasificación
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        # Inicialización
        self._init_classifier()
    
    def _create_backbone(
        self,
        backbone: str,
        pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """
        Crea el backbone según la arquitectura especificada.
        
        Returns:
            Tupla (backbone_module, num_features)
        """
        weights = 'DEFAULT' if pretrained else None
        
        if backbone.startswith('efficientnet'):
            if backbone == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=weights)
                num_features = 1280
            elif backbone == 'efficientnet_b2':
                model = models.efficientnet_b2(weights=weights)
                num_features = 1408
            elif backbone == 'efficientnet_b4':
                model = models.efficientnet_b4(weights=weights)
                num_features = 1792
            else:
                model = models.efficientnet_b2(weights=weights)
                num_features = 1408
            
            # Quitar clasificador original
            model.classifier = nn.Identity()
            
        elif backbone.startswith('resnet'):
            if backbone == 'resnet18':
                model = models.resnet18(weights=weights)
                num_features = 512
            elif backbone == 'resnet34':
                model = models.resnet34(weights=weights)
                num_features = 512
            elif backbone == 'resnet50':
                model = models.resnet50(weights=weights)
                num_features = 2048
            else:
                model = models.resnet50(weights=weights)
                num_features = 2048
            
            # Quitar fc original
            model.fc = nn.Identity()
            
        elif backbone == 'mobilenet_v3':
            model = models.mobilenet_v3_large(weights=weights)
            num_features = 960
            model.classifier = nn.Identity()
            
        elif backbone == 'convnext_tiny':
            model = models.convnext_tiny(weights=weights)
            num_features = 768
            model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Backbone no soportado: {backbone}")
        
        return model, num_features
    
    def _init_classifier(self):
        """Inicializa pesos del clasificador."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada (B, C, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        features = self.backbone(x)
        
        # Asegurar que features sea 2D
        if features.dim() > 2:
            features = features.mean(dim=[2, 3])  # Global average pooling
        
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene probabilidades de predicción.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Probabilidades (0-1) para cada clase
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Obtiene predicciones binarias.
        
        Args:
            x: Tensor de entrada
            threshold: Umbral de clasificación
            
        Returns:
            Predicciones binarias (0 o 1)
        """
        proba = self.predict_proba(x)
        return (proba >= threshold).float()


def create_model(
    num_classes: int,
    backbone: str = 'efficientnet_b2',
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = 'cuda'
) -> MultiLabelClassifier:
    """
    Función de conveniencia para crear un modelo.
    
    Args:
        num_classes: Número de clases
        backbone: Arquitectura base
        pretrained: Usar pesos preentrenados
        dropout: Tasa de dropout
        device: Dispositivo
        
    Returns:
        Modelo inicializado
    """
    model = MultiLabelClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout
    )
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🔧 Modelo creado: {backbone}")
    print(f"   Total parámetros: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    print(f"   Clases: {num_classes}")
    print(f"   Dispositivo: {device}")
    
    return model


class FocalLoss(nn.Module):
    """
    Focal Loss para manejo de clases desbalanceadas.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = self.alpha * focal_weight * bce
        return loss.mean()


def get_loss_function(
    loss_type: str = 'BCE',
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Obtiene función de pérdida.
    
    Args:
        loss_type: 'BCE' o 'Focal'
        pos_weight: Pesos para clases positivas
        label_smoothing: Label smoothing
        
    Returns:
        Función de pérdida
    """
    if loss_type == 'BCE':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == 'Focal':
        return FocalLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Loss no soportado: {loss_type}")
