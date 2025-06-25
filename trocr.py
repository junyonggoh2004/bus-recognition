import os
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
from PIL import Image


class TrOCRInference:
    def __init__(self, model_path):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    def inference(self, image_path):
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        print(f"Output text: {generated_text}")
        return generated_text
