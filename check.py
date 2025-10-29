try:
    import torch
    print("PyTorch版本:", torch.__version__)
    print("CUDA可用:", torch.cuda.is_available())
except ImportError:
    print("PyTorch未安装")

try:
    import numpy
    print("NumPy版本:", numpy.__version__)
except ImportError:
    print("NumPy未安装")

try:
    import PIL
    print("Pillow版本:", PIL.__version__)
except ImportError:
    print("Pillow未安装")

try:
    import tqdm
    print("tqdm已安装")
except ImportError:
    print("tqdm未安装")

try:
    import requests
    print("requests已安装")
except ImportError:
    print("requests未安装")

try:
    import importlib
    # dynamic import so static analyzers won't flag unresolved imports
    realesrgan = importlib.import_module("realesrgan")
    print("realesrgan已安装")
except ImportError:
    realesrgan = None
    print("realesrgan未安装，请运行: pip install realesrgan")
