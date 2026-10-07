# OCRdoc — GPU Setup Script (Windows, CUDA 12.x)
# Chạy với quyền Administrator: .\scripts\install_gpu.ps1

Write-Host "=== OCRdoc GPU Setup ===" -ForegroundColor Cyan

# ── 1. Uninstall CPU builds ──────────────────────────────
Write-Host "`n[1/5] Gỡ cài bản CPU..." -ForegroundColor Yellow
pip uninstall torch torchvision torchaudio paddlepaddle -y

# ── 2. Cài PyTorch CUDA 12.4 ────────────────────────────
Write-Host "`n[2/5] Cài PyTorch CUDA 12.4..." -ForegroundColor Yellow
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 `
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. Cài PaddlePaddle GPU ─────────────────────────────
Write-Host "`n[3/5] Cài PaddlePaddle GPU (CUDA 12.0)..." -ForegroundColor Yellow
pip install paddlepaddle-gpu==2.6.2.post120 `
    -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# ── 4. Cài Poppler (cho pdf2image) ──────────────────────
Write-Host "`n[4/5] Cài Poppler cho pdf2image..." -ForegroundColor Yellow
$chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue
if ($chocoInstalled) {
    choco install poppler -y
} else {
    Write-Host "  Chocolatey chưa cài. Tải Poppler thủ công:" -ForegroundColor Red
    Write-Host "  https://github.com/oschwartz10612/poppler-windows/releases" -ForegroundColor Red
    Write-Host "  Giải nén vào C:\poppler và thêm C:\poppler\Library\bin vào PATH" -ForegroundColor Red
    Write-Host "  Hoặc set POPPLER_PATH=C:\poppler\Library\bin trong .env" -ForegroundColor Red
}

# ── 5. Kiểm tra ─────────────────────────────────────────
Write-Host "`n[5/5] Kiểm tra cài đặt..." -ForegroundColor Yellow

python -c "import torch; cuda_ok = torch.cuda.is_available(); print(f'PyTorch CUDA: {cuda_ok} | Device: {torch.cuda.get_device_name(0) if cuda_ok else \"N/A\"}')"
python -c "import paddle; paddle.utils.run_check()"

Write-Host "`n=== Xong! Chạy scripts\download_models.py để tải models ===" -ForegroundColor Green
