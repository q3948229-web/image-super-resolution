# 图像超分项目（Lanczos/ZSSR/Real-ESRGAN Anime）

该项目提供三条可选路径：

- 经典重采样：bicubic / lanczos（默认），可选锐化与对比度增强，轻量快速；
- ZSSR 思路：在每张输入图像上“自训练”的小模型，稳定不乱画，但较耗时；
- 二次元 Anime：集成 Real-ESRGAN anime_6B 模型，支持 2x/4x 放大与风格融合（style_sensitivity）。

你可以对一个目录下的所有图片进行批处理，并保持子目录结构。

---

## 1. 环境准备（Windows/PowerShell）

强烈建议使用虚拟环境隔离依赖：

```powershell
cd D:\CompusLearning\graph
# 创建并激活 .venv
python -m venv .\.venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1

# 升级基础工具
python -m pip install --upgrade pip setuptools wheel

# 安装常规依赖（已包含 realesrgan、opencv、requests、tqdm 等；固定 numpy<2 以兼容 torch）
python -m pip install -r requirements.txt

# 安装（可选）GPU 版 PyTorch/torchvision（建议从官方选择与你 CUDA 匹配的指令）
# 例如 CUDA 12.6：
# python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

提示：如果启用 GPU，请确保 `nvidia-smi` 可用，且 `torch.cuda.is_available()` 为 True。

PowerShell 若提示执行策略阻止激活脚本，可用上面命令在“当前会话”放宽策略；或手动在 Cmd 启动：`.\.venv\Scripts\activate.bat`。

---

## 2. 运行示例

激活虚拟环境后，即可运行项目。 .venv\Scripts\activate

基础用法（递归处理所有子目录图片，输出写入 `out/`，文件名追加 `_x{scale}` 后缀）：

```powershell
# 经典 Lanczos 放大 2x（已验证可运行）
python .\main.py -i .\input_images -o .\out -s 2 --method lanczos --recursive

# Bicubic 放大 1.5x + 锐化
python .\main.py -i .\input_images -o .\out -s 1.5 --method bicubic --sharpen --recursive

# ZSSR 本地训练（较慢），默认 200 epochs，可按需调整
python .\main.py -i .\input_images -o .\out -s 2 --method zssr --epochs 120 --batch-size 8 --recursive

# Real-ESRGAN Anime 4x（自动下载权重；若 GPU 可用会自动使用 CUDA）
# 注意：首次运行会尝试下载权重，若网络受限请参考下方“权重下载”手动放置
python .\main.py -i .\input_images -o .\out -s 4 --method anime --style_sensitivity 0.7 --recursive

# 指定设备（例如 GPU 0）
python .\main.py -i .\input_images -o .\out -s 4 --method anime --device cuda:0 --recursive
```

核心参数：

- `--method {bicubic,lanczos,zssr,anime}` 选择路径；
- `-s/--scale` 放大倍数（anime 支持 2 或 4 倍）；
- `--style_sensitivity` 在 anime 模式下控制与 bicubic 的融合强度，越大越锐利；
- `--sharpen ...` 与 `--contrast` 用于经典模式的感知增强；
- `--epochs`、`--batch-size` 为 ZSSR 训练超参数；
- `--device` 指定设备（如 `cuda:0` 或 `cpu`）。

---

## 3. Anime 权重下载

默认会尝试自动下载 `models/RealESRGAN_x4plus_anime_6B.pth`。若自动下载失败（公司网络或镜像限制），请手动下载并放到 `models/` 目录：

- 官方发布页（常见版本 v0.2.5/v0.3.0）
	- <https://github.com/xinntao/Real-ESRGAN/releases>
	- 资源名：`RealESRGAN_x4plus_anime_6B.pth`
- 若 GitHub 受限，可在 HuggingFace 等镜像搜索同名文件后下载

放置完成后再次运行 anime 模式即可。

---

## 4. 常见问题（FAQ）

1) ImportError/RuntimeError 与 numpy 相关（例如 “Numpy is not available” 或 “compiled using NumPy 1.x cannot run in 2.x”）

- 解决：将 numpy 固定为 `<2`（本项目已在 `requirements.txt` 固定）。若仍报错，执行：

```powershell
python -m pip install "numpy<2"
```

2) PowerShell 无法激活 `.venv`（执行策略限制）

- 解决：在当前会话放宽策略并激活：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
```

3) Real-ESRGAN 导入报错或版本不兼容（和 torchvision 的 API 变更有关）

- 本项目在导入前做了一个兼容性“垫片”，通常可缓解 `functional_tensor` 模块缺失的问题；
- 如果仍异常，建议按 PyTorch 官网安装与你 CUDA 匹配的 torch/torchvision 组合；
- 或暂时在 CPU 上运行以排查依赖（去掉 `--device cuda:0` 并确认能推理）。

---

## 5. 当前验证状态

- 已在 `.venv` 中完成依赖安装（固定 numpy<2），并成功运行 Lanczos 2x 示例；
- Anime 路径已接好并带自动/手动权重下载逻辑；若网络限制导致自动下载失败，请按上节手动放置权重后重试。

---

## 6. 参考/致谢

- Real-ESRGAN: <https://github.com/xinntao/Real-ESRGAN>
- ZSSR: <https://arxiv.org/abs/1711.06077>

---

## 7. 性能与加速（强烈建议）

- 自动分块：GPU 上如果未指定 `--tile`，程序会默认启用 `tile=512` 分块推理，显著降低显存占用并通常更快；也可手动指定：

```powershell
# 针对 8GB 显存可尝试 512；若仍慢或 OOM 可改为 256
python .\main.py -i .\input_images -o .\out -s 4 --method anime --tile 512 --recursive --device cuda:0
```

- 半精度：在 GPU 上会自动使用 FP16（half）推理，无需手动设置。

- 混合流程（更快）：先 Anime 2x，再用 Lanczos 2x 合成 4x，总体更快，质量接近全 4x Anime：

```powershell
# 第一步：Anime 2x（输出到 out2）
python .\main.py -i .\input_images -o .\out2 -s 2 --method anime --tile 512 --recursive --device cuda:0

# 第二步：Lanczos 2x（从 out2 到 out，得到 4x）
python .\main.py -i .\out2 -o .\out -s 2 --method lanczos --recursive
```

- 仅处理单文件：把目标图片放入一个临时目录（例如 `work/`），再把 `-i` 指向该目录。

- 预演模式：先用 `--dry-run` 看看会生成哪些文件和大小，不实际写盘：

```powershell
python .\main.py -i .\input_images -o .\out -s 4 --method anime --tile 512 --recursive --dry-run
```

- CPU 回退：用于排查或在无 GPU 的环境：

```powershell
python .\main.py -i .\input_images -o .\out -s 4 --method anime --device cpu --tile 128 --recursive
```

---

## 8. 参数详解（含默认值与建议）

- `-i, --input <dir>` 必填。输入目录（包含 png/jpg/jpeg）。如果只想处理单图，建议把图片放到一个临时目录并将该目录作为输入。

- `-o, --output <dir>` 必填。输出根目录。脚本会保持输入的相对目录结构，文件名会追加 `_x{scale}` 后缀。

- `-s, --scale <float>` 默认 `2.0`。放大倍数，支持小数（如 1.5）。注意：`--method anime` 通常仅支持 `2` 或 `4`，其它倍数会退回到 bicubic 以保证可用。

- `--recursive` 可选。递归处理子目录中的图片。

- `--method {bicubic,lanczos,zssr,anime}` 默认 `lanczos`。
  - `bicubic` 与 `lanczos`：经典重采样，速度快、占用低；可配合锐化与对比度增强。
  - `zssr`：每张图自训练的小模型，更稳定但耗时；适合强调“忠实不重绘”的场景。
  - `anime`：Real-ESRGAN 动漫模型，锐利清晰；建议 2x 或 4x，需模型权重。

- `--sharpen`（经典模式可用）启用 UnsharpMask 锐化，默认关闭。
  - `--sharpen-radius <float>` 默认 `2.0`
  - `--sharpen-percent <int>` 默认 `150`
  - `--sharpen-threshold <int>` 默认 `3`

- `--contrast <float>`（经典模式可用）默认 `1.0`。对比度增强系数，>1 增强对比度，<1 降低对比度。

- `--dry-run` 只打印将要处理的文件与目标路径，不实际写盘。用于预演批处理结果。

- `--epochs <int>`（仅 zssr）默认 `200`。增大可略提质量但更慢；快速试验可设为 `50~120`。

- `--batch-size <int>`（仅 zssr）默认 `8`。显存占用与速度的权衡；GPU 内存吃紧可调小。

- `--weights <path>`（仅 anime）指定模型权重路径。为空时程序会尝试自动下载到 `models/`。推荐：
  - 二次元：`models/RealESRGAN_x4plus_anime_6B.pth`（推荐）
  - 通用写实：`models/RealESRGAN_x4plus.pth`（可用）
  - 注意：`realesr-general-x4v3.pth` 为 ncnn 版本权重，与此 Python 推理链不兼容。

- `--style_sensitivity <0..1>`（仅 anime）默认 `0.7`。融合强度，越大越接近模型输出、越锐利；`0` 等于纯 bicubic，`1` 等于纯模型结果。推荐范围 `0.5~0.8`。

- `--device <str>` 默认自动检测。可显式指定 `cuda:0` 或 `cpu`。强烈建议在有 NVIDIA GPU 时使用 `cuda:0`。

- `--tile <int>` 默认 `0`（不分块）。在 GPU 上如果未指定，本项目已默认自动启用 `tile=512` 以提速并降低显存占用。
  - 更大的 tile 一般更快但更吃显存；8GB 显存建议 256~512；CPU 上建议 128。

- 其他说明：
 	- 输出命名：文件名会追加 `_x{scale}`，如果是小数会把 `.` 替换为 `p`（例如 `1.5` -> `_x1p5`）。
 - 成功计数：只有实际写出文件才计入 “完成：共处理 N 个图像”。若推理失败，会打印“未生成输出：…”提示。
	- 帮助：`python .\main.py --help` 查看所有参数。

