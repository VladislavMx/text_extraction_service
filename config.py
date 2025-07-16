import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "nanonets/Nanonets-OCR-s")
MAX_NEW_TOKENS = 15000