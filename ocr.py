from paddleocr import PaddleOCR
import fitz
import io
from PIL import Image


def OCR(images:Image.Image) -> str:
    """
    Extract visible text from an image.
    Requires Tesseract OCR installed separately.
    """
    ocr = PaddleOCR(lang="en",enable_mkldnn=False)
    result = list(ocr.predict(images))
    full_text = "\n".join(result[0]['rec_text'])

