from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Engine ────────────────────────────────────────────
    paddle_device: str = "gpu"
    vram_limit_mb: int = 7000

    # ── EasyOCR ───────────────────────────────────────────
    easyocr_model_dir: Path = Path("models/easyocr")

    # ── OpenOCR ───────────────────────────────────────────
    # mode: "mobile" (nhanh hơn) hoặc "server" (chính xác hơn)
    openocr_mode: str = "server"
    # Phiên bản Vietnamese weights: "v3" = vi_PP-OCRv3, "v4" = vi_PP-OCRv4
    # (PaddleOCR tự động dùng bản mới nhất có sẵn cho lang="vi")
    openocr_vi_weights: str = "v4"

    # ── Model paths ───────────────────────────────────────
    paddle_model_dir: Path = Path("models/paddle")
    # Vietnamese rec model (vi_PP-OCRv4 hoặc vi_PP-OCRv3, tải bằng download_models.py)
    # Để trống → tự detect theo openocr_vi_weights
    vi_rec_model_dir: str = ""

    # ── Preprocessing ─────────────────────────────────────
    default_dpi: int = 300
    max_pdf_pages: int = 100

    # ── API ───────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Poppler (Windows) ─────────────────────────────────
    poppler_path: str = ""

    def resolve_paddle_model_dir(self) -> str:
        return self.paddle_model_dir.as_posix()


# Singleton instance
settings = Settings()
