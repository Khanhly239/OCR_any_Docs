# setup_marker_venv.ps1
# Tạo venv riêng cho marker-pdf — không thể cài chung với transformers 5.x.
# Chạy 1 lần: scripts\setup_marker_venv.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir   = Join-Path $ScriptDir "marker_venv"

Write-Host "=== Marker Venv Setup ===" -ForegroundColor Cyan

if (Test-Path $VenvDir) {
    Write-Host "Venv da ton tai tai: $VenvDir" -ForegroundColor Yellow
    Write-Host "Xoa va tao lai? (y/N): " -NoNewline
    $ans = Read-Host
    if ($ans -ne "y" -and $ans -ne "Y") {
        Write-Host "Huy." -ForegroundColor Gray
        exit 0
    }
    Remove-Item -Recurse -Force $VenvDir
}

Write-Host "Tao venv tai $VenvDir ..." -ForegroundColor Cyan
python -m venv $VenvDir
if (-not $?) { Write-Host "Loi: khong tao duoc venv" -ForegroundColor Red; exit 1 }

Write-Host "Cai marker-pdf (co the mat 5-10 phut, download ~2GB models khi chay lan dau)..." -ForegroundColor Cyan
& "$VenvDir\Scripts\pip.exe" install marker-pdf
if (-not $?) { Write-Host "Loi: pip install that bai" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=== Hoan tat! ===" -ForegroundColor Green
Write-Host "Marker venv san sang tai: $VenvDir" -ForegroundColor Green
Write-Host "Khoi dong OCR Viewer va chon engine 'Marker' de test." -ForegroundColor Green
