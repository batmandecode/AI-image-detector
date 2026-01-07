from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

def detect_ai_image(image_path):
    model_name = "umm-maybe/AI-image-detector"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    image = Image.open(https://static.getimg.ai/media/getimg_ai_img-IxDdULnjyCHUxuhFwoFQG.webp).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = outputs.logits.argmax(-1).item()
    label = model.config.id2label[prediction]

    return label

# -------- Main Program --------
image_path = "test.jpg"   # image file in same folder
result = detect_ai_image(image_path)

print("Image Type:", result)
