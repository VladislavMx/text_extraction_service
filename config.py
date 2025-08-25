import os
import re
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "nanonets/Nanonets-OCR-s")
BILATERAL_FILTER_DIAMETER = 5
BILATERAL_FILTER_SIGMA_COLOR = 55
BILATERAL_FILTER_SIGMA_SPACE = 60

ADAPTIVE_THRESH_BLOCK_SIZE = 21
ADAPTIVE_THRESH_C = 4

MAX_NEW_TOKENS = 15000
ERROR_ANSWER = "Пожалуйста, пришлите изображение в лучшем качестве"
MODEL_TEMPERATURE = 0.2
MIN_WORDS = 5

TEXT_CLEAN_RE = re.compile(r'[^a-zA-Zа-яА-Я ]')