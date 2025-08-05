import logging
import string
import base64
from io import BytesIO
import requests

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from config import MODEL_NAME
from jinja2 import Environment, FileSystemLoader
import cv2
import re

MAX_NEW_TOKENS = 15000
ERROR_ANSWER = "Пожалуйста, пришлите изображение в лучшем качестве"
MODEL_TEMPERATURE = 0.2
MIN_WORDS = 5

logging.basicConfig(
    filename="app/ocr_service.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

ALLOWED_CHARS = set(
    string.ascii_letters +
    string.digits +
    string.punctuation +
    string.whitespace +
    'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
)

TEXT_CLEAN_RE = re.compile(r'[^a-zA-Zа-яА-Я ]')

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = api_key



class OCRService:
    def __init__(self, llm_client):
        self.api_key = ""
        self.llm = llm_client
        self.template_env = Environment(loader=FileSystemLoader("core/templates"))
        logger.info(MODEL_NAME)

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

        except Exception as e:
            raise

    def chat_completion(self, messages):
        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": "nanonets/Nanonets-OCR-s",
            "messages": messages,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": MODEL_TEMPERATURE,
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()

    def image_to_base64(image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def check_context(self, output_text):
        if not output_text:
            return False

        cleaned_text = TEXT_CLEAN_RE.sub('', output_text)
        words = cleaned_text.split()

        gibberish_chars = sum(1 for c in output_text if c not in ALLOWED_CHARS)
        gibberish_ratio = gibberish_chars / len(output_text)

        if gibberish_ratio > 0.6:
            return False

        return True

    def preprocess_image(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.fastNlMeansDenoising(binary, h=30)

        pil_image = Image.fromarray(denoised)
        return pil_image


    def predict(self, image: Image.Image) -> str:
        try:
            image = self.preprocess_image(image)
            img_b64 = self.image_to_base64(image)

            prompt_template = self.template_env.get_template("prompt.j2")
            prompt = prompt_template.render()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": img_b64},
                    {"type": "text", "text": prompt},
                ]},
            ]

            response = self.llm.chat_completion(messages)

            output_text = response["choices"][0]["message"]["content"]

            if not self.check_context(output_text):
                return ERROR_ANSWER

            return output_text

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

