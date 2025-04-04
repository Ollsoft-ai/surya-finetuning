from PIL import Image, ImageOps
import numpy as np
import cv2
import os
import json
import csv

from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import surya.input.processing as surya_image_utils

from typing import List
import csv

# The following functions were copied and modified from the surya git
RECOGNITION_PAD_VALUE = 255
GOLD_DIR = "gold_text"
SCANS_DIR = "scans_prepared"
SCANS_TARGET_CSV = "scans_target_mapping.csv"
DATASET_DIR = "../dataset"
if not os.path.isdir(DATASET_DIR):
    os.mkdir(DATASET_DIR)
if not os.path.isdir(os.path.join(DATASET_DIR, "text")):
    os.mkdir(os.path.join(DATASET_DIR, "text"))
if not os.path.isdir(os.path.join(DATASET_DIR, "images")):
    os.mkdir(os.path.join(DATASET_DIR, "images"))


def levenshteinDistance(s1, s2):
    """
    Credit: https://stackoverflow.com/a/32558749
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def relative_distance(w1, w2):
    return levenshteinDistance(w1, w2) / (len(w1) + len(w2))

def align_all(length_tolerance = 0.2, word_similarity_tolerance = 0.3, sentence_similarity_tolerance = 0.08):
    """Generate a line dataset for all pictures in the `scans/` folder

    Args:
        length_tolerance (float, optional): The aligned line will have the length between predicted_length*(1 - `length_tolerance`) and predicted_length*(1 + `length_tolerance`). Defaults to 0.2.
        word_similarity_tolerance (float, optional): Words whose relative edit distance is smaller than this value will be considered the same. This is used when detecting a matching word in the next text box. Defaults to 0.3.
        sentence_similarity_tolerance (float, optional): Maximum allowed relative edit distance between the predicted and gold lines. If this can't be achieved, the line will be discarded. Defaults to 0.04.
    """

    with open(SCANS_TARGET_CSV) as list_csv:
        target_mapping = csv.reader(list_csv)
        with open(os.path.join(DATASET_DIR, "text", "labels.csv"), "w") as labels_file:
            label_csv = csv.writer(labels_file)
            langs = ["cs"] # Replace with your languages - optional but recommended
            det_processor, det_model = load_det_processor(), load_det_model()
            rec_model, rec_processor = load_rec_model(), load_rec_processor()
            for im_name, target_id in target_mapping:
                print("Processing", im_name)
                name_noext, _ = os.path.splitext(im_name)

                with open(os.path.join(GOLD_DIR, target_id + ".txt")) as gold:
                    gold_text = gold.read()

                gold_words = gold_text.split()

                image = Image.open(os.path.join(SCANS_DIR, im_name))
                image = ImageOps.exif_transpose(image)

                predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)

                img = surya_image_utils.convert_if_not_rgb([image])[0]
                img_array = np.array(img, dtype=np.uint8)

                gwords_slice_start = 0
                gwords_slice_end = 0
                for page_content in predictions:

                    text_lines = [tline for tline in page_content.text_lines if (tline.text and not tline.text.isspace() and tline.confidence > 0.7)]
                    for line_box_idx in range(len(text_lines)):
                        line_box = text_lines[line_box_idx]
                        line_text = line_box.text.strip()
                        line_len = len(line_text)

                        if line_box_idx == len(text_lines) - 1:
                            # add the remaining text to the last box
                            aligned_text = gold_words[gwords_slice_start:]
                        else:
                            # start building the line until minimum length is reaches
                            aligned_text = ""
                            while len(aligned_text) < (1 - length_tolerance)*line_len and gwords_slice_end <= len(gold_words):
                                gwords_slice_end += 1

                                aligned_text = " ".join(gold_words[gwords_slice_start: gwords_slice_end])

                                if gwords_slice_end == len(gold_words):
                                    current_edit_distance = relative_distance(line_text, aligned_text)

                                    if current_edit_distance <= sentence_similarity_tolerance:
                                        new_img_name = os.path.splitext(name_noext)[0] + "-" + str(line_box_idx) + ".png"
                                        label_csv.writerow((new_img_name, aligned_text))

                                        image_cropped = surya_image_utils.slice_and_pad_poly(img_array, np.array(line_box.polygon, dtype=int))
                                        image_cropped.save(os.path.join(DATASET_DIR, "images", new_img_name))

                                    break

                            next_line_first_word = text_lines[line_box_idx + 1].text.split()[0]

                            current_edit_distance = relative_distance(line_text, aligned_text)

                            while len(aligned_text) < (1 + length_tolerance)*line_len and gwords_slice_end < len(gold_words):
                                extended_line = " ".join(gold_words[gwords_slice_start: gwords_slice_end + 1])
                                next_dist = relative_distance(line_text, extended_line)
                                if next_dist < current_edit_distance:
                                    aligned_text = extended_line
                                    current_edit_distance = next_dist
                                    gwords_slice_end += 1
                                
                                elif relative_distance(next_line_first_word, gold_words[gwords_slice_end]) < word_similarity_tolerance:
                                    # found a match for the following word in the next line; aborting
                                    break
                                else:
                                    # TODO: maybe add another condition when unsure
                                    break
                            
                            gwords_slice_start = gwords_slice_end

                            if current_edit_distance > sentence_similarity_tolerance:
                                print("Unsure with the alignment (will be excluded from the dataset):")
                                print("\tPredicted line:", line_text)
                                print("\tAligned so far:", aligned_text)
                                if gwords_slice_end < len(gold_words):
                                    print("\tNext word:", gold_words[gwords_slice_end])
                                print()
                                continue

                            new_img_name = os.path.splitext(name_noext)[0] + "-" + str(line_box_idx) + ".png"
                            label_csv.writerow((new_img_name, aligned_text, line_text))

                            image_cropped = surya_image_utils.slice_and_pad_poly(img_array, np.array(line_box.polygon, dtype=int))
                            image_cropped.save(os.path.join(DATASET_DIR, "images", new_img_name))


if __name__ == "__main__":
    align_all()