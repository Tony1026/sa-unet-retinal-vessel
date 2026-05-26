import importlib.util


for module in ['torch', 'albumentations', 'cv2', 'numpy']:
    print(f'{module}: {importlib.util.find_spec(module) is not None}')

try:
    import torch

    print(f'torch_version: {torch.__version__}')
    print(f'cuda_available: {torch.cuda.is_available()}')
    print(f'cuda_device_count: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        print(f'cuda_device_0: {torch.cuda.get_device_name(0)}')
except Exception as exc:
    print(f'torch_error: {type(exc).__name__}: {exc}')
