# Generating a Line Dataset

## Dataset (!Read Before Printing!)

To pair the the photos with their labels, we use QR codes. Each printed document **must contain a QR code** that contains the name of the corresponding target text file, without the `.txt` extension. For example, if the correct (gold) transcript of a document is saved in `doc_gold1.txt`, then the printed/scanned document must contain a QR code containing the string `doc_gold1`.

We suggest using at least 20 documents.

## Objective

Given a scanned image and a target text file, the objective is to detect and cut out the individual lines and align the corresponding parts of the target text to them. This format is required for fine-tuning the Surya recognition module.

## How it Works

To avoid manual cutting out of the text lines and aligning them to the targets, we employ an automatic method. Even though Surya-ocr is not perfect without fine-tuning, we can use it to detect text boxes and their approximate text. Next, we align the target text to the predicted boxes based on the predicted text (at least a partial match is expected). We use heuristics such as text length, relative Levenshtein distance and next line checking.

## How to Run

Set up and activate the virtual environment as [described](../README.md#set-up-requirements).

### Running the Generator

First, create your printed file dataset according to [the instructions](#dataset-read-before-printing). Place scanned document images in the `scans` directory and their corresponding target text documents into the `gold_text` directory. Again, make sure the you that the [proper](#dataset-read-before-printing) QR codes are present and that the targets are saved in `.txt` format.

Note that an example picure-target pair is already in the directories.

Once the scans and targets are ready, run the following:

```bash
cd line_dataset_generator
python3 prepare_documents.py
python3 boxes-gold_align.py
```

If the alignment is not sufficient, you may need to adjust the parameters of the `align_all(...)` function in `boxes-gold_align.py`. The results of the alignment will be saved in the `../dataset` directory in the following format:

```
dataset
├── images
│   ├── doc55-0.png
│   ├── ...
│   └── doc55-N.png
└── text
    └── labels.csv
```

The `labels.csv` (a **valid** csv file) contains the gold labels for each image. The format is as follows:

```csv
img-0.png,"Lorem ipsum dolor sit amet, consectetur adipiscing elit."
img-1.png,Suspendisse in ipsum orci.
```