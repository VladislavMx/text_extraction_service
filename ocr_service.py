import logging
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from config import MODEL_NAME, MAX_NEW_TOKENS
from jinja2 import Environment, FileSystemLoader



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

    def predict(self, image: Image.Image, MAX_NEW_TOKENS) -> str:
        try:

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


            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


            return output_text[0]

        except Exception as e:

            raise
