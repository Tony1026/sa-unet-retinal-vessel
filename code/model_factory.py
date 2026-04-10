from model_sa_unet import SAUNet
from model_sa_unetv2 import SAUNetV2


MODEL_REGISTRY = {
    'sa_unet': SAUNet,
    'sa_unetv2': SAUNetV2,
}


def create_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unsupported model: {model_name}')
    return MODEL_REGISTRY[model_name](**kwargs)
