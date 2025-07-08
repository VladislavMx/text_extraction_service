from fastapi import FastAPI, File, UploadFile
from ocr_service import OCRService
from PIL import Image
from io import BytesIO

app = FastAPI()
ocr_service = OCRService()

@app.post("/ocr")

async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    result = ocr_service.predict(image, max_new_tokens=15000)
    return {"text": result}
