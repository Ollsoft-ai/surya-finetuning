from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import torch

IMG = Image.open("scans/IMG_4931.jpg")
LANGS = ["cs"] # NOTE: IMPORTANT -- replace with your language
MODEL_PATH = "model.pt" # NOTE: place the model to this directory first
recognition_predictor = RecognitionPredictor()
detection_predictor = DetectionPredictor()


with open(MODEL_PATH, "rb") as saved_rec:
    rec_model = torch.load(saved_rec, weights_only=False)
# recognition_predictor.model = rec_model
# NOTE: during actual inference, preprocessing is necessary. Check out `../line_dataset_generator/prepare_documents.py` for some prepocessing approaches.

predictions = recognition_predictor([IMG], [LANGS], detection_predictor, recognition_batch_size=8)[0]

print("\n\n".join([line.text for line in predictions.text_lines if line.confidence > 0.7]))
