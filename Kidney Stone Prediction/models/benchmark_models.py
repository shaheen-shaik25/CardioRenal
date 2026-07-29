import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=2, weights=None):
    """ResNet50 Transfer Learning Model."""
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def get_mobilenetv2(num_classes=2, weights=None):
    """MobileNetV2 Lightweight Model."""
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

def get_vgg16(num_classes=2, weights=None):
    """VGG16 Architecture."""
    model = models.vgg16(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

def get_inceptionv3(num_classes=2, weights=None):
    """InceptionV3 Architecture."""
    model = models.inception_v3(weights=weights, aux_logits=True, init_weights=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if model.AuxLogits is not None:
        aux_in = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
    return model

def load_benchmark_model(model_name, num_classes=2, weight_path=None):
    """Utility loader for standard benchmark models."""
    name = model_name.lower()
    if name == 'xresnet50':
        from models.xresnet50 import XResNet50
        model = XResNet50(num_classes=num_classes)
    elif name == 'resnet50':
        model = get_resnet50(num_classes=num_classes)
    elif name == 'mobilenetv2':
        model = get_mobilenetv2(num_classes=num_classes)
    elif name == 'vgg16':
        model = get_vgg16(num_classes=num_classes)
    elif name == 'inceptionv3':
        model = get_inceptionv3(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if weight_path:
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        state_dict = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(state_dict)

    return model
