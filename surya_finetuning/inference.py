from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import torch

IMG = Image.open("scans/IMG_4931.jpg")
LANGS = ["cs"] # NOTE: IMPORTANT -- replace with your language
MODEL_PATH = "model.pt" # NOTE: place the model to this directory first
det_processor, det_model = load_det_processor(), load_det_model()
rec_processor = load_rec_processor()

with open(MODEL_PATH, "rb") as saved_rec:
    rec_model = torch.load(saved_rec, weights_only=False)

# NOTE: during actual inference, preprocessing is necessary. Check out `../line_dataset_generator/prepare_documents.py` for some prepocessing approaches.

predictions = run_ocr([IMG], [LANGS], det_model, det_processor, rec_model, rec_processor, recognition_batch_size=8)[0]
print("\n\n".join([line.text for line in predictions.text_lines if line.confidence > 0.7]))
