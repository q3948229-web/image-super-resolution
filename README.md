# 简单图像超分脚本（基于 Pillow）

这是一个最初步的图像超分脚本示例，使用 Pillow 的双三次（BICUBIC）插值进行放大。适合作为快速测试或作为后续替换为深度学习模型的占位实现。

功能

- 支持 PNG / JPG / JPEG
- 可选择是否递归子目录
- 保持原始目录结构，将输出写入指定输出目录，文件名后添加 `_x{scale}` 后缀

依赖

- Python 3.8+
- Pillow

安装依赖（在 PowerShell 中）：

```powershell
python -m pip install -r requirements.txt
```

示例用法（PowerShell）：

```powershell
# 将 input_images 目录中的图片按 2 倍放大，输出到 out 目录
python main.py -i .\input_images -o .\out -s 2 --recursive
```

注意

- 该脚本仅为最初步实现，放大质量与基于深度学习的超分模型（例如 Real-ESRGAN、EDSR）不可同日而语。
- 对于 JPEG，脚本会将 alpha 通道展平到白色背景以避免保存错误。
