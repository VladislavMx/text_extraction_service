import logging
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from config import MODEL_NAME, MAX_NEW_TOKENS
from jinja2 import Environment, FileSystemLoader
import cv2


logging.basicConfig(
    filename="app/ocr_service.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self, model_path=MODEL_NAME):
        logger.info(model_path)

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.processor = AutoProcessor.from_pretrained(model_path)

        except Exception as e:
            raise

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

    def predict(self, image: Image.Image, MAX_NEW_TOKENS) -> str:
        try:
            image = self.preprocess_image(image)
            template_env = Environment(loader=FileSystemLoader("core/templates"))
            prompt_template = template_env.get_template("ocr_prompt.j2")
            prompt = prompt_template.render()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)


            output_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, repetition_penalty=1.2)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


            return output_text[0]

        except Exception as e:

            raise
