# 改进的图像超分脚本（支持 ZSSR 本地训练）

此项目新增了基于“ZSSR 思路”的可选深度学习模式（method=zssr）：

- ZSSR 本质是“在测试图像上自训练”——不依赖外部训练数据，网络在输入图像生成的降采样/重建对上进行训练，从而学习适合该图像的映射。
- 优点：对原图信息更忠实，不会随意生成不存在的细节（符合你对“稳定、不重绘”的要求）。
- 缺点：每张图片需要在本地进行短时训练，耗时和显存消耗依输入大小、训练轮次有关，但在 GPU（如 RTX4060）上可接受。

快速安装（PowerShell）：

```powershell
python -m pip install -r requirements.txt
# 注意：torch 的安装可能需要指定带 CUDA 的版本，例如：
# python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
```

虚拟环境 (建议)

强烈建议使用 Python 虚拟环境来隔离项目依赖。项目中已添加 `setup_env.ps1`，可以在 PowerShell 中运行以创建 `.venv` 并指导你激活与安装依赖。

在 PowerShell 中的典型流程（手动执行）如下：

```powershell
cd D:\CompusLearning\graph
# 创建虚拟环境
python -m venv .\.venv
# 激活（PowerShell）
.\\.venv\\Scripts\\Activate.ps1
# 升级 pip 并安装依赖
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

关于 PyTorch：为了利用你的 RTX4060 的 GPU 加速，你应该按 PyTorch 官方页面选择与你的 CUDA 版本匹配的安装命令（例如指定合适的 extra-index-url 或 wheel）。如果不知道 CUDA 版本，可以运行 `nvidia-smi` 在终端查看。若不确定，先安装 CPU 版本或在官方安装页选择推荐命令。

示例用法（PowerShell）：

```powershell
# 使用 ZSSR（在每张图像上本地训练，默认 200 epochs）
python main.py -i .\input_images -o .\out -s 2 --method zssr --recursive

# 使用快速 classical 方法（lanczos + 可选锐化）
python main.py -i .\input_images -o .\out -s 2 --method lanczos --sharpen
```

参数说明（与 README.md 重复，新增 zssr 说明）：

- `--method zssr`：启用基于图像内训练的深度学习方法（更稳定、忠实）。
- `--epochs`：zssr 模式下的训练轮数（已在 CLI 中暴露，默认 200）。
- `--batch-size`：zssr 模式下训练的批大小（已在 CLI 中暴露，默认 8）。

性能提示：RTX4060（8GB）适合运行此脚本；对于大分辨率图像，建议先使用 `--dry-run` 或在小分辨率上试验，以估算训练时间与显存需求。

***

如果你确认要继续，我可以：

- 把 `--epochs`, `--batch-size` 等 zssr 参数暴露为 CLI 选项；
- 添加一个示例小图并演示一次 end-to-end 运行（在本地终端执行）；
- 或者把 zssr 的训练日志/模型保存选项加入以便后续分析。
