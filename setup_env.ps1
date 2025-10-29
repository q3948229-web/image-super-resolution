<#
PowerShell 脚本：在 Windows 下为本项目创建虚拟环境并安装依赖

用法（在项目根目录 d:\CompusLearning\graph 中运行）：
  .\setup_env.ps1

行为：
 - 在当前目录创建名为 .venv 的虚拟环境（如果已存在则不会覆盖）
 - 激活该虚拟环境（提供激活命令供你手动执行）
 - 升级 pip、setuptools、wheel
 - 尝试安装 requirements.txt 中列出的依赖

注意：
 - 对于 PyTorch（torch）推荐按照官方页面选择与你 CUDA 版本匹配的 wheel 安装命令。脚本会尝试通过 requirements.txt 安装，但若需要 GPU 加速的特定版本请按 README 中提示手动安装。
 - 在默认 PowerShell 中执行脚本前，可能需要调整执行策略（Set-ExecutionPolicy）；如果不希望改变策略，请在 PowerShell 中手动运行每条命令。
#>

Write-Host "== 创建虚拟环境 .venv（如果不存在） =="
if (-Not (Test-Path -Path .\.venv)) {
    python -m venv .\.venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "创建虚拟环境失败，请确认 python 可用并已添加到 PATH"
        exit 1
    }
    Write-Host ".venv 已创建"
} else {
    Write-Host ".venv 已存在，跳过创建"
}

Write-Host ""
Write-Host "要激活虚拟环境，请在 PowerShell 中运行："
Write-Host "  .\\.venv\\Scripts\\Activate.ps1"
Write-Host "或者（Cmd）: .\\.venv\\Scripts\\activate.bat"
Write-Host ""

Write-Host "将为你升级 pip、setuptools、wheel 并安装 requirements.txt 中的包（如果你已经激活虚拟环境，请在激活后运行以下命令）："
Write-Host "  python -m pip install --upgrade pip setuptools wheel"
Write-Host "  python -m pip install -r requirements.txt"

Write-Host "\n提示：如果你计划使用 GPU（例如 RTX4060），建议按照 PyTorch 官方安装向导（https://pytorch.org/get-started/locally/）选择合适的 CUDA wheel，以获得最佳性能。"

exit 0
