"""
Pre-download tất cả model weights trước lần chạy đầu.
Chạy: python scripts/download_models.py

Sau khi download xong, models được cache tại models/paddle/ và HuggingFace cache.
"""
import sys
import os
from pathlib import Path

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tarfile
import urllib.request

import numpy as np
from loguru import logger

PADDLE_MODEL_DIR = Path("models/paddle")

# vi_PP-OCRv3_rec: model Vietnamese gốc (bị loại bỏ ở PaddleOCR 2.9.x, thay bằng latin chung)
VI_REC_V3_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/vi_PP-OCRv3_rec_infer.tar"
VI_REC_V4_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/vi_PP-OCRv4_rec_infer.tar"

PADDLE_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def download_vi_rec_model(version: str = "v4") -> Path | None:
    """
    Tải vi_PP-OCRv3 hoặc vi_PP-OCRv4 rec model từ PaddleOCR multilingual hub.
    PaddleOCR 2.9.x không còn include vi model riêng — phải tải thủ công.
    Trả về path tới thư mục model, hoặc None nếu thất bại.
    """
    url = VI_REC_V4_URL if version == "v4" else VI_REC_V3_URL
    model_name = f"vi_PP-OCR{version}_rec_infer"
    out_dir = PADDLE_MODEL_DIR / model_name

    if out_dir.exists() and any(out_dir.iterdir()):
        logger.success(f"  {model_name} đã có tại {out_dir}")
        return out_dir

    tar_path = PADDLE_MODEL_DIR / f"{model_name}.tar"
    logger.info(f"  Tải {model_name} từ PaddleOCR hub...")

    def _show_progress(block, block_size, total):
        done = block * block_size
        pct = min(100, done * 100 // total) if total > 0 else 0
        if pct % 20 == 0:
            logger.debug(f"    {pct}% ({done // (1024*1024)}MB / {total // (1024*1024)}MB)")

    try:
        urllib.request.urlretrieve(url, tar_path, reporthook=_show_progress)
    except Exception as e:
        # v4 có thể chưa tồn tại → thử fallback v3
        if version == "v4":
            logger.warning(f"  vi_PP-OCRv4 không có ({e}), thử v3...")
            return download_vi_rec_model("v3")
        logger.error(f"  Tải {model_name} thất bại: {e}")
        return None

    try:
        with tarfile.open(tar_path) as tf:
            tf.extractall(PADDLE_MODEL_DIR)
        tar_path.unlink(missing_ok=True)
        logger.success(f"  {model_name} đã giải nén vào {out_dir}")
        return out_dir
    except Exception as e:
        logger.error(f"  Giải nén {model_name} thất bại: {e}")
        return None


def download_paddle_models() -> None:
    logger.info("Đang tải PaddleOCR models (PP-OCRv5 + PP-StructureV3)...")
    try:
        from paddleocr import PaddleOCR, PPStructure
        dummy = np.ones((64, 64, 3), dtype=np.uint8) * 255

        logger.info("  Tải detection + recognition model...")
        ocr = PaddleOCR(
            use_gpu=False,  # Download không cần GPU
            lang="vi",
            use_angle_cls=True,
            model_storage_directory=PADDLE_MODEL_DIR.as_posix(),
            show_log=False,
        )
        ocr.ocr(dummy, cls=True)
        logger.success("  PaddleOCR OK")

        logger.info("  Tải PP-StructureV3 models (table + layout)...")
        struct = PPStructure(
            table=True,
            layout=True,
            lang="en",  # layout model chỉ hỗ trợ 'en' và 'ch'
            use_gpu=False,
            model_storage_directory=PADDLE_MODEL_DIR.as_posix(),
            show_log=False,
        )
        struct(dummy)
        logger.success("  PP-StructureV3 OK")

    except Exception as e:
        logger.error(f"PaddleOCR download thất bại: {e}")
        raise


def download_chandra_models() -> None:
    """
    Warm-up Chandra OCR 2 model (VLM, ~7B params).
    Model tự download từ HuggingFace Hub (datalab-to/chandra-ocr-2) khi khởi tạo lần đầu.
    Cần ~8GB VRAM (CUDA) hoặc RAM (CPU).
    """
    logger.info("Đang khởi tạo Chandra OCR 2 model (VLM — có thể mất vài phút)...")
    try:
        from chandra.model import InferenceManager
        from chandra.model.schema import BatchInputItem
        from PIL import Image

        manager = InferenceManager(method="hf")
        # Warm-up với ảnh trắng nhỏ
        dummy_pil = Image.new("RGB", (64, 64), color=(255, 255, 255))
        batch = [BatchInputItem(image=dummy_pil, prompt_type="ocr_layout")]
        manager.generate(batch)
        logger.success("  Chandra OCR 2 model OK (cached tại HuggingFace cache)")
    except ImportError:
        logger.warning(
            "  chandra-ocr chưa cài — chạy: pip install \"chandra-ocr[hf]\""
        )
    except Exception as e:
        logger.error(f"  Chandra download thất bại: {e}")


def download_vintern_models() -> None:
    """
    Download Vintern-1B-v3.5 (5CD-AI/Vintern-1B-v3.5) về HuggingFace cache.
    ~2GB, chạy được trên RTX 3050 Laptop (~2GB VRAM).
    """
    logger.info("Đang tải Vintern-1B-v2 (5CD-AI/Vintern-1B-v2, ~2GB)...")
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download("5CD-AI/Vintern-1B-v2")
        logger.success(f"  Vintern-1B-v3.5 đã download: {local_dir}")
    except ImportError:
        logger.warning("  huggingface_hub chưa cài — chạy: pip install huggingface_hub")
    except Exception as e:
        logger.error(f"  Vintern download thất bại: {e}")


def download_openocr_models() -> None:
    """
    Warm-up OpenOCR detection model (SVTRv2/DBNet, PyTorch).
    OpenOCR tự động download weights về cache của nó khi khởi tạo lần đầu.
    PaddleOCR vi_PP-OCRv3 rec model dùng chung với models/paddle/.
    """
    logger.info("Đang khởi tạo OpenOCR detection model (SVTRv2)...")
    try:
        from openocr import OpenOCR
        import numpy as np
        import tempfile, os, cv2

        dummy = np.ones((64, 256, 3), dtype=np.uint8) * 255
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, dummy)

        try:
            det = OpenOCR(task="det", mode="server")
            det(image_path=tmp_path)
            logger.success("  OpenOCR detection model OK")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        logger.info("  PaddleOCR vi_PP-OCRv3 rec model (dùng chung models/paddle/)...")
        from paddleocr import PaddleOCR
        rec = PaddleOCR(
            use_gpu=False,
            lang="vi",
            use_angle_cls=True,
            det=False,
            model_storage_directory=PADDLE_MODEL_DIR.as_posix(),
            show_log=False,
        )
        rec.ocr(dummy, det=False, cls=False)
        logger.success("  vi_PP-OCRv3 recognition model OK")

    except ImportError:
        logger.warning(
            "openocr-python chưa được cài, bỏ qua. "
            "Cài bằng: pip install openocr-python==0.1.5"
        )
    except Exception as e:
        logger.error(f"OpenOCR download thất bại: {e}")
        raise


if __name__ == "__main__":
    logger.info("=== Tải tất cả OCR models ===")
    download_paddle_models()
    download_chandra_models()
    download_vintern_models()
    download_openocr_models()
    logger.info("Tải vi_PP-OCRv4 rec model (Vietnamese-specific weights)...")
    vi_path = download_vi_rec_model("v4")
    if vi_path:
        logger.success(f"  Vietnamese rec model: {vi_path}")
    else:
        logger.warning("  Không tải được vi model, sẽ dùng latin_PP-OCRv3 làm fallback")

    logger.info("Khởi tạo EasyOCR models (vi + en)...")
    try:
        import easyocr
        from pathlib import Path as _P
        model_dir = str(_P("models/easyocr"))
        _P(model_dir).mkdir(parents=True, exist_ok=True)
        reader = easyocr.Reader(
            lang_list=["vi", "en"],
            gpu=False,
            model_storage_directory=model_dir,
            download_enabled=True,
            verbose=False,
        )
        dummy = np.ones((32, 256, 3), dtype=np.uint8) * 255
        reader.readtext(dummy, detail=0)
        logger.success("  EasyOCR vi+en models OK")
    except ImportError:
        logger.warning("  easyocr chưa cài — chạy: pip install easyocr")
    except Exception as e:
        logger.error(f"  EasyOCR download thất bại: {e}")

    logger.success("=== Hoàn tất! Tất cả models đã sẵn sàng ===")
