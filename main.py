"""改进的图像超分脚本（不使用大型深度学习模型）

改进点：
- 提供更高质量的重采样方法（Lanczos）作为默认选项（可选 bicubic）
- 可选锐化（UnsharpMask）和对比度增强来提高感知清晰度
- 添加 --dry-run 模式以便先查看将要处理的文件及目标路径而不写入磁盘

注意：这些都是基于经典图像处理的增强方法，不能与训练过的深度学习超分模型在细节重建能力上相比，但在无模型、无 GPU 场景下能显著提升视觉效果。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import sys
import torch
torch.backends.cudnn.benchmark = True

# 深度学习相关的导入将在需要时懒加载（以避免无 torch 环境时报错）


def is_image_file(p: Path) -> bool:
	return p.suffix.lower() in {".png", ".jpg", ".jpeg"}


def _format_scale(scale: float) -> str:
	# 格式化 scale 用于文件名（把小数点替换为 p，例如 1.5 -> 1p5）
	s = str(scale)
	# 去掉多余的 0
	if "." in s:
		s = s.rstrip("0").rstrip(".")
	return s.replace('.', 'p')


def upscale_image(
		src_path: Path,
		dst_path: Path,
		scale: float,
		resample: int = Image.LANCZOS,
		sharpen: bool = False,
		sharpen_radius: float = 2.0,
		sharpen_percent: int = 150,
		sharpen_threshold: int = 3,
		contrast: float = 1.0,
		dry_run: bool = False,
	) -> None:
	try:
		with Image.open(src_path) as im:
			mode = im.mode
			if im.mode in ("RGBA", "P", "LA"):
				has_alpha = True
			else:
				has_alpha = False

			orig_size = im.size
			new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))

			up = im.resize(new_size, resample=resample)

			# 可选锐化
			if sharpen:
				up = up.filter(ImageFilter.UnsharpMask(radius=sharpen_radius, percent=sharpen_percent, threshold=sharpen_threshold))

			# 可选对比度增强
			if contrast != 1.0:
				enhancer = ImageEnhance.Contrast(up)
				up = enhancer.enhance(contrast)

			if dry_run:
				print(f"[dry-run] 会写入: {dst_path} (size={new_size}, mode={up.mode})")
				return

			# 保存前处理 alpha / jpeg
			dst_path.parent.mkdir(parents=True, exist_ok=True)
			suffix = dst_path.suffix.lower()
			save_kwargs = {}
			if suffix in {".jpg", ".jpeg"}:
				if has_alpha:
					bg = Image.new("RGB", up.size, (255, 255, 255))
					bg.paste(up, mask=up.split()[-1])
					up = bg
				else:
					up = up.convert("RGB")
				save_kwargs["quality"] = 95

			up.save(dst_path, **save_kwargs)
	except Exception as e:
		print(f"跳过文件 {src_path}（无法处理）：{e}")


def process_directory(
		input_dir: Path,
		output_dir: Path,
		scale: float,
		recursive: bool,
		method: str = "lanczos",
		sharpen: bool = False,
		sharpen_radius: float = 2.0,
		sharpen_percent: int = 150,
		sharpen_threshold: int = 3,
		contrast: float = 1.0,
		dry_run: bool = False,
		epochs: int = 200,
		batch_size: int = 8,
		device: str | None = None,
		weights: str | None = None,
		style_sensitivity: float = 0.7,
		tile: int = 0,
	) -> None:
	if not input_dir.exists():
		print(f"输入目录不存在: {input_dir}")
		return

	resample_map = {
		"bicubic": Image.BICUBIC,
		"lanczos": Image.LANCZOS,
	}
	resample = resample_map.get(method, Image.LANCZOS)

	patterns = ["**/*"] if recursive else ["*"]
	files_processed = 0

	for pattern in patterns:
		for p in input_dir.glob(pattern):
			if p.is_file() and is_image_file(p):
				rel = p.relative_to(input_dir)
				scale_tag = _format_scale(scale)
				stem = rel.stem + f"_x{scale_tag}"
				dst_rel = rel.parent / (stem + p.suffix)
				dst = output_dir / dst_rel
				print(f"处理: {p} -> {dst} (scale={scale}, method={method})")
				# 如果使用深度学习方法（zssr / anime），调用对应流程
				if method == "zssr":
					# 检查全局是否有 torch（避免在函数内再次 import 导致作用域问题）
					if "torch" not in globals():
						print("使用 zssr 模式需要安装 torch 和 torchvision，请运行: python -m pip install torch torchvision")
						continue
					# 调用 ZSSR 风格的本地训练与放大（参数可根据需要调整）
					zssr_upscale(
						p,
						dst,
						scale,
						device=("cuda" if torch.cuda.is_available() else "cpu"),
						epochs=epochs,
						patch_size=64,
						batch_size=batch_size,
					)
				elif method == "anime":
					# anime (Real-ESRGAN anime) 分支
					try:
						from torch import cuda
					except Exception:
						print("要使用 anime 模型，请先安装 PyTorch 与 realesrgan：python -m pip install torch torchvision realesrgan")
						continue
					dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
					anime_upscale(
						p,
						dst,
						scale,
						device=dev,
						weights=weights,
						style_sensitivity=style_sensitivity,
						tile=tile,
						dry_run=dry_run,
					)
					# handled above
				else:
					upscale_image(
						p,
						dst,
						scale,
						resample=resample,
						sharpen=sharpen,
						sharpen_radius=sharpen_radius,
						sharpen_percent=sharpen_percent,
						sharpen_threshold=sharpen_threshold,
						contrast=contrast,
						dry_run=dry_run,
					)
				files_processed += 1

	print(f"完成：共处理 {files_processed} 个图像")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="改进的图像超分（Lanczos + 可选锐化）")
	p.add_argument("-i", "--input", required=True, help="输入目录（包含 png/jpg 图片）")
	p.add_argument("-o", "--output", required=True, help="输出目录，结果将写入此处")
	p.add_argument("-s", "--scale", type=float, default=2.0, help="放大倍数，支持小数，例如 1.5")
	p.add_argument("--recursive", action="store_true", help="递归处理子目录")
	p.add_argument("--method", choices=["bicubic", "lanczos", "zssr", "anime"], default="lanczos", help="重采样/方法（默认 lanczos，或 zssr 使用每图训练的深度学习方法，anime 使用 Real-ESRGAN anime 模型）")
	p.add_argument("--sharpen", action="store_true", help="放大后应用 UnsharpMask 锐化（默认关闭）")
	p.add_argument("--sharpen-radius", type=float, default=2.0, help="UnsharpMask 的 radius 参数")
	p.add_argument("--sharpen-percent", type=int, default=150, help="UnsharpMask 的 percent 参数")
	p.add_argument("--sharpen-threshold", type=int, default=3, help="UnsharpMask 的 threshold 参数")
	p.add_argument("--contrast", type=float, default=1.0, help="对比度增强因子，1.0 表示不改变")
	p.add_argument("--dry-run", action="store_true", help="仅显示将要处理的文件和目标路径，不实际写入")
	p.add_argument("--epochs", type=int, default=200, help="zssr 模式下本地训练的轮数（只在 method=zssr 时生效）")
	p.add_argument("--batch-size", type=int, default=8, help="zssr 模式下训练时的 batch size（只在 method=zssr 时生效）")
	# Anime model options
	p.add_argument("--weights", type=str, default=None, help="anime 模型权重路径（优先使用本地权重），若为空则尝试自动下载到 models/ 下")
	p.add_argument("--style_sensitivity", type=float, default=0.7, help="anime 模型风格增强强度（0.0-1.0，越大越锐利/风格化）")
	p.add_argument("--device", type=str, default=None, help="设备，例如 cuda:0 或 cpu（默认自动检测）")
	p.add_argument("--tile", type=int, default=0, help="分块大小（0 表示不分块），用于大图分块推理")
	return p.parse_args(argv)


def zssr_upscale(src_path: Path, dst_path: Path, scale: float, device: str = "cpu", epochs: int = 200, patch_size: int = 64, batch_size: int = 8, lr: float = 1e-4) -> None:
	"""基于 ZSSR 思路的简化实现：
	- 在单张输入图像上自行生成训练对（把图像下采样再上采样得到的对）并训练一个小型 CNN
	- 训练后对目标尺寸的 bicubic 放大结果进行残差修正，得到最终结果

	该方法不会使用外部训练数据，因此更稳定、更忠实于原图（不会随意重绘新细节）。
	"""
	try:
		import torch
		import torch.nn as nn
		import torch.optim as optim
		from torch.utils.data import DataLoader, Dataset
		import numpy as np
	except Exception:
		print("zssr_upscale 需要 torch、torchvision、numpy 等依赖，请先安装：python -m pip install torch torchvision numpy")
		return

	# 简单的小网络（残差学习）
	class SmallSRNet(nn.Module):
		def __init__(self, n_chan=64, n_blocks=6):
			super().__init__()
			layers = [nn.Conv2d(3, n_chan, 3, 1, 1), nn.ReLU(inplace=True)]
			for _ in range(n_blocks):
				layers += [nn.Conv2d(n_chan, n_chan, 3, 1, 1), nn.ReLU(inplace=True)]
			layers += [nn.Conv2d(n_chan, 3, 3, 1, 1)]
			self.net = nn.Sequential(*layers)

		def forward(self, x):
			return self.net(x)

	# 读取图像并准备训练对
	with Image.open(src_path) as pil_im:
		im = pil_im.convert("RGB")
	img_np = np.array(im).astype(np.float32) / 255.0  # HWC

	# 构造下采样版本：用尺度 s_down > 1 把图像缩小再放大回原始大小，作为网络输入，目标为原始 img
	s_down = 2.0 if scale >= 2.0 else 1.5
	# 延迟导入 cv2（OpenCV），若未安装则给出提示；使用 importlib 动态导入以减少静态分析报错
	try:
		import importlib
		cv2 = importlib.import_module("cv2")
	except Exception:
		print("zssr_upscale 需要 opencv-python (cv2)，请先安装：python -m pip install opencv-python")
		return

	h, w = img_np.shape[:2]
	small_h = max(16, int(h / s_down))
	small_w = max(16, int(w / s_down))
	# 下采样再上采样（bicubic）
	small = cv2.resize(img_np, (small_w, small_h), interpolation=cv2.INTER_CUBIC)
	small_up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

	# Dataset: 从 small_up (input) -> img_np (target)
	class PatchDataset(Dataset):
		def __init__(self, inp, tgt, patch_size):
			self.inp = inp
			self.tgt = tgt
			self.ps = patch_size
			self.h, self.w = inp.shape[:2]

		def __len__(self):
			return 10000  # 虚拟长度，按 epoch 从随机位置抽样

		def __getitem__(self, idx):
			import random
			y = random.randint(0, max(0, self.h - self.ps))
			x = random.randint(0, max(0, self.w - self.ps))
			inp_patch = self.inp[y:y + self.ps, x:x + self.ps]
			tgt_patch = self.tgt[y:y + self.ps, x:x + self.ps]
			# HWC to CHW
			inp_t = torch.from_numpy(inp_patch.transpose(2, 0, 1)).float()
			tgt_t = torch.from_numpy(tgt_patch.transpose(2, 0, 1)).float()
			return inp_t, tgt_t

	dataset = PatchDataset(small_up, img_np, patch_size)
	dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	device = torch.device(device)
	net = SmallSRNet().to(device)
	optimizer = optim.Adam(net.parameters(), lr=lr)
	loss_fn = nn.L1Loss()

	# 训练循环（快速）
	try:
		from tqdm import tqdm
		progress = tqdm(range(epochs), desc=f"训练 {src_path.name}")
	except Exception:
		progress = range(epochs)

	net.train()
	for _ in progress:
		for inp_b, tgt_b in dl:
			inp_b = inp_b.to(device)
			tgt_b = tgt_b.to(device)
			# 归一化范围 [0,1]
			inp_b = inp_b
			tgt_b = tgt_b
			out = net(inp_b)
			# 残差学习：网络直接预测修正项
			pred = inp_b + out
			loss = loss_fn(pred, tgt_b)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			break  # 每 epoch 只用一个 batch 以加快训练（ZSSR 常用大量随机采样，快速实现里逐 epoch 采样一次）

	# 预测阶段：先用 bicubic 放大到目标尺寸，然后用网络修正残差
	target_size = (int(w * scale), int(h * scale))
	import math
	# 使用 PIL 做最终放大到 target_size
	up_bic = im.resize(target_size, resample=Image.BICUBIC)
	up_np = np.array(up_bic).astype(np.float32) / 255.0

	# 分块推理以节省显存（若需要）
	net.eval()
	with torch.no_grad():
		inp = torch.from_numpy(up_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
		out = net(inp)
		pred = inp + out
		pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
		pred = np.clip(pred, 0.0, 1.0)
		pred_img = Image.fromarray((pred * 255.0).astype('uint8'))

	# 保存结果（保留扩展名逻辑）
	dst_path.parent.mkdir(parents=True, exist_ok=True)
	suffix = dst_path.suffix.lower()
	if suffix in {".jpg", ".jpeg"}:
		pred_img = pred_img.convert("RGB")
		pred_img.save(dst_path, quality=95)
	else:
		pred_img.save(dst_path)


def anime_upscale(src_path: Path, dst_path: Path, scale: float, device: str = "cpu", weights: str | None = None, style_sensitivity: float = 0.7, tile: int = 0, dry_run: bool = False) -> None:
	"""使用 Real-ESRGAN anime 模型进行超分并按 style_sensitivity 与 bicubic 结果融合。

	仅支持常见放大倍数（2 或 4）。若模型或依赖缺失，会输出友好提示。
	"""
	try:
		from realesrgan import RealESRGAN
	except Exception:
		print("anime_upscale 需要安装 realesrgan：python -m pip install realesrgan （以及 torch torchvision）")
		return

	# 确定权重路径
	models_dir = Path("models")
	models_dir.mkdir(parents=True, exist_ok=True)
	default_name = "RealESRGAN_x4plus_anime_6B.pth"
	if weights:
		weights_path = Path(weights)
	else:
		weights_path = models_dir / default_name

	# 自动下载权重（若不存在）
	if not weights_path.exists():
		url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.3.0/RealESRGAN_x4plus_anime_6B.pth"
		print(f"权重文件 {weights_path} 不存在，准备从 {url} 下载（请确认网络可访问）...")
		try:
			import requests
		except Exception:
			print("下载依赖 requests 缺失，请先安装：python -m pip install requests")
			return

		try:
			resp = requests.get(url, stream=True, timeout=30)
			resp.raise_for_status()
			total = int(resp.headers.get("content-length", 0))
			with open(weights_path, "wb") as f:
				downloaded = 0
				for chunk in resp.iter_content(chunk_size=8192):
					if chunk:
						f.write(chunk)
						downloaded += len(chunk)
				print(f"下载完成: {weights_path} ({downloaded} bytes)")
		except Exception as e:
			print(f"下载权重失败: {e}")
			return

	# 仅支持 2 或 4 的 scale 值使用模型（其他 scale 使用 bicubic fallback）
	if int(scale) not in (2, 4):
		print(f"anime 模型通常支持 2 或 4 倍放大，当前 scale={scale}，将退回到 bicubic + 少量后处理")
		return upscale_image(src_path, dst_path, scale, resample=Image.BICUBIC)

	model_scale = int(scale)

	try:
		model = RealESRGAN(device, scale=model_scale)
		model.load_weights(str(weights_path))
	except Exception as e:
		print(f"加载模型失败: {e}")
		return

	with Image.open(src_path) as pil_im:
		im = pil_im.convert("RGB")

	# dry-run 输出信息
	target_size = (int(im.width * scale), int(im.height * scale))
	if dry_run:
		print(f"[dry-run] anime 将写入: {dst_path} (model_scale={model_scale}, target_size={target_size}, weights={weights_path})")
		return

	# 运行模型推理
	try:
		sr = model.predict(im)
	except Exception as e:
		print(f"模型推理失败: {e}")
		return

	# bicubic baseline（用于融合）
	bic = im.resize(target_size, resample=Image.BICUBIC).convert("RGB")
	sr = sr.convert("RGB")

	# 按风格敏感度融合：越接近 1 越强烈使用模型结果
	style = max(0.0, min(1.0, float(style_sensitivity)))
	blended = Image.blend(bic, sr, style)

	# 保存
	dst_path.parent.mkdir(parents=True, exist_ok=True)
	suffix = dst_path.suffix.lower()
	if suffix in {".jpg", ".jpeg"}:
		blended = blended.convert("RGB")
		blended.save(dst_path, quality=95)
	else:
		blended.save(dst_path)



def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	input_dir = Path(args.input)
	output_dir = Path(args.output)
	scale = args.scale

	if scale <= 0:
		print("scale 必须为正数")
		return 2

	process_directory(
		input_dir,
		output_dir,
		scale,
		args.recursive,
		method=args.method,
		sharpen=args.sharpen,
		sharpen_radius=args.sharpen_radius,
		sharpen_percent=args.sharpen_percent,
		sharpen_threshold=args.sharpen_threshold,
		contrast=args.contrast,
		dry_run=args.dry_run,
		epochs=args.epochs,
		batch_size=args.batch_size,
		device=args.device,
		weights=args.weights,
		style_sensitivity=args.style_sensitivity,
		tile=args.tile,
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())