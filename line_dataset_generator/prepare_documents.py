from PIL import Image, ImageFile
import pytesseract
from pillow_heif import register_heif_opener
from surya.detection import DetectionPredictor
from qreader import QReader
import numpy as np
import sys

import re
import os
import csv


register_heif_opener()

SCANS_FOLDER="scans"
JPG_FOLDER="scans_jpg"
if not os.path.isdir(JPG_FOLDER):
    os.mkdir(JPG_FOLDER)
PREPARED_FOLDER="scans_prepared"
if not os.path.isdir(PREPARED_FOLDER):
    os.mkdir(PREPARED_FOLDER)

qr = QReader()
detection_predictor = DetectionPredictor()

def fine_rotate_image(detection_result, image: ImageFile.ImageFile):
    """From Surya detection result, estimate the document rotation and fix it"""
    def get_angle_from_polygon(polygon: list[tuple[float, float]]) -> float:
        polygon = np.array(polygon)
        begin = (polygon[0] + polygon[3])/2
        end = (polygon[1] + polygon[2])/2
        vec = end - begin
        angle = np.arctan(vec[1]/vec[0])/np.pi*180
        return angle
    
    avg_angle = np.mean([get_angle_from_polygon(box.polygon) for box in detection_result.bboxes if box.confidence > 0.7])
    rotated = image.rotate(avg_angle, fillcolor=(255,255,255), expand=True)
    return rotated

def find_rotation_degree(img):
    # Use pytesseract to extract rotation information from the image
    text = pytesseract.image_to_osd(img)
    # Extract the rotation angle
    rotation = int(re.search(r'(?<=Rotate: )\d+', text).group(0))

    # Return the necessary rotation to correct the orientation
    if rotation == 90:
        return 270  # Rotate clockwise
    elif rotation == 270:
        return 90  # Rotate counterclockwise
    elif rotation == 180:
        return 180  # Upside-down
    else:
        return 0  # Proper orientation


with open("scans_target_mapping.csv", "w") as scans_targets:
    csv_w = csv.writer(scans_targets)
    for image_name in os.listdir(SCANS_FOLDER):
        print("Processing", image_name)
        raw_name, _ = os.path.splitext(image_name)
        img = Image.open(os.path.join(SCANS_FOLDER, image_name))
        img.save(os.path.join(JPG_FOLDER, raw_name + ".jpg"))

        img = Image.open(os.path.join(JPG_FOLDER, raw_name + ".jpg"))

        # Tesseract is used to fix wrong image orientation -- 90, 180, 270 degree rotation
        try:
            orientation = find_rotation_degree(img)
        except pytesseract.pytesseract.TesseractError:
            print("Skipped: Tesseract error")
            continue

        if orientation != 0:
            img = img.rotate(orientation, expand=True)
        
        # surya detection model is used to further improve the rotation such that the detected boxes are perfectly horizontal on average
        detected_polygons = detection_predictor([img])[0]
        img = fine_rotate_image(detected_polygons, img)

        img.save(os.path.join(PREPARED_FOLDER, raw_name + ".jpg"))

        qr_contents = qr.detect_and_decode(np.asarray(img))

        if len(qr_contents) != 1:
            print(f"Expected to find exactly  QR code, found", len(qr_contents), file=sys.stderr)
            continue

        qr_content = qr_contents[0]
        if qr_content:
            csv_w.writerow([raw_name + ".jpg", qr_content])