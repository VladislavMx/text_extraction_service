import logging
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from io import BytesIO
import torch

"""
logging.basicConfig(
    filename="app/ocr_service.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)"""

class OCRService:
    def __init__(self, model_path="nanonets/Nanonets-OCR-s"):
        try:

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.processor = AutoProcessor.from_pretrained(model_path)


        except Exception as e:

            raise

    def predict(self, image: Image.Image, max_new_tokens=4096) -> str:
        try:


            prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""

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
