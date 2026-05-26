from model_gflow_unet import (
    GFlowSAUNetV2,
    GFlowSAUNetV2Conditional,
    GFlowSAUNetV2ConditionalNoConservation,
    GFlowSAUNetV2ConditionalRandomFlow,
    GFlowSAUNetV2ConditionalRandomInit,
    GFlowSAUNetV2ConditionalUniformFlow,
    GFlowSAUNetV2NoConservation,
    GFlowSAUNetV2RandomFlow,
    GFlowSAUNetV2RandomInit,
    GFlowSAUNetV2UniformFlow,
    GFlowUNet,
    GFlowUNetDirectSink,
    GFlowUNetNoConservation,
    GFlowUNetNoConservationDirectSink,
    GFlowUNetRandomFlow,
    GFlowUNetRandomFlowDirectSink,
    GFlowUNetRandomInit,
    GFlowUNetRandomInitDirectSink,
    GFlowUNetUniformFlow,
    GFlowUNetUniformFlowDirectSink,
)
from model_multihead_unet import MultiHeadSAUNetV2, MultiHeadUNet
from model_sa_unet import SAUNet
from model_sa_unetv2 import SAUNetV2
from model_unet import UNet


MODEL_REGISTRY = {
    'gflow_unet': GFlowUNet,
    'gflow_unet_direct': GFlowUNetDirectSink,
    'gflow_unet_direct_no_cons': GFlowUNetNoConservationDirectSink,
    'gflow_unet_direct_random': GFlowUNetRandomFlowDirectSink,
    'gflow_unet_direct_randinit': GFlowUNetRandomInitDirectSink,
    'gflow_unet_direct_uniform': GFlowUNetUniformFlowDirectSink,
    'gflow_unet_no_cons': GFlowUNetNoConservation,
    'gflow_unet_random': GFlowUNetRandomFlow,
    'gflow_unet_randinit': GFlowUNetRandomInit,
    'gflow_unet_uniform': GFlowUNetUniformFlow,
    'gflow_sa_unetv2': GFlowSAUNetV2,
    'gflow_sa_unetv2_cond': GFlowSAUNetV2Conditional,
    'gflow_sa_unetv2_cond_no_cons': GFlowSAUNetV2ConditionalNoConservation,
    'gflow_sa_unetv2_cond_random': GFlowSAUNetV2ConditionalRandomFlow,
    'gflow_sa_unetv2_cond_randinit': GFlowSAUNetV2ConditionalRandomInit,
    'gflow_sa_unetv2_cond_uniform': GFlowSAUNetV2ConditionalUniformFlow,
    'gflow_sa_unetv2_no_cons': GFlowSAUNetV2NoConservation,
    'gflow_sa_unetv2_random': GFlowSAUNetV2RandomFlow,
    'gflow_sa_unetv2_randinit': GFlowSAUNetV2RandomInit,
    'gflow_sa_unetv2_uniform': GFlowSAUNetV2UniformFlow,
    'multihead_unet': MultiHeadUNet,
    'multihead_sa_unetv2': MultiHeadSAUNetV2,
    'unet': UNet,
    'sa_unet': SAUNet,
    'sa_unetv2': SAUNetV2,
}


def create_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f'Unsupported model: {model_name}')
    return MODEL_REGISTRY[model_name](**kwargs)


def available_models():
    return sorted(MODEL_REGISTRY)
