import logging
import string
import base64
from io import BytesIO
import requests
from pydantic import BaseSettings

from config import MAX_NEW_TOKENS, ERROR_ANSWER, MODEL_TEMPERATURE, MIN_WORDS, TEXT_CLEAN_RE

import preprocessing

from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from config import MODEL_NAME
from jinja2 import Environment, FileSystemLoader
import re
import streamlit as st

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

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    api_url: str = "https://api.openai.com"
    openai_key: str

    class Config:
        env_file = ".env"




class OCRService:
    def __init__(self, llm_client):
        self.api_key = ""
        self.llm = llm_client
        self.template_env = Environment(loader=FileSystemLoader("core/templates"))
        logger.info(MODEL_NAME)

        self.settings = Settings()
        self.api_url = self.settings.api_url.rstrip("/") + "/v1/chat/completions"
        self.api_key = self.settings.openai_key

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
        try:
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        finally:
            buffered.close()

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

    def preprocess(img):
        a, col0, b = st.columns([1, 20, 1])
        col1, col2, col3 = st.columns([1, 1, 1])
        col4, col5, col6 = st.columns([1, 1, 1])

        img = preprocessing.convert_img(img)

        with col2.container(border=True):
            st.image(img, output_format="auto", caption="original image")

        img = preprocessing.normalize_img(img)
        with col4.container(border=True):
            st.image(img, output_format="auto", caption="normalized image")

        img = preprocessing.grayscale_img(img)
        with col5.container(border=True):
            st.image(img, output_format="auto", caption="grayscale image")

        img = preprocessing.denoise_img(img)

        img = preprocessing.deskew_img(img)
        with col6.container(border=True):
            st.image(img, output_format="auto", caption="deskew image")

        img = preprocessing.threshold_img(img, threshold_val=40)
        with col3.container(border=True):
            st.image(img, output_format="auto", caption="threshold image")

        img = Image.fromarray(img)
        return img


    def predict(self, image: Image.Image) -> str:
        try:
            image = self.preprocess(image)
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

