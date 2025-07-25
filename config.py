import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "nanonets/Nanonets-OCR-s")
MAX_NEW_TOKENS = 15000
ERROR_ANSWER = "Пожалуйста, пришлите изображение в лучшем качестве"
MODEL_TEMPERATURE = 0.2
MIN_WORDS = 5