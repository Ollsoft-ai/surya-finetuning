# Surya Fine-Tuning

## Objective

The objective is fine-tune the Surya text recognition model to our specific needs.

## How to Run

### Dataset Format

Dataset is a folder of the following format:

```
dataset
├── images
│   ├── img0.png
│   ├── ...
│   └── imgN.png
└── text
    └── labels.csv
```

The images in the `images` folder are cropped single lines. The `labels.csv` (a **valid** csv file) contains the gold labels for each image. The images annotated in `labels.csv` must be a subset of the images ine the `images` folder. The format is as follows:

```csv
img0.png,"Lorem ipsum dolor sit amet, consectetur adipiscing elit."
img1.png,Suspendisse in ipsum orci.
```

### Dataset Generation

Datasets can be generated using the [`line_dataset_generator`](../line_dataset_generator/README.md)


### Running the Training Procedure

Set up and activate the virtual environment as [described](../README.md#set-up-requirements).

Change to this directory:

```bash
cd surya_finetuning/
```

The training procedure is provided in the file `surya_training.py`. To run the training procedure, first inspect the command-line arguments. The arguments can be specified either in the command or directly in the script by changing their default value. It is **important** that you change the `language` argument to your language code.

```
options:
  -h, --help            show this help message and exit
  --lang LANG           Language code of the documents
  --batch_size BATCH_SIZE
                        Batch size.
  --epochs EPOCHS       Number of epochs.
  --epochs_warmup EPOCHS_WARMUP
                        Number of epochs during which the LR is increased linearly from zero to the specified value.
  --label_smoothing LABEL_SMOOTHING
                        Label smoothing.
  --lr LR               Learning rate.
  --lr_decay LR_DECAY   Use cosine decay?
  --weight_decay WEIGHT_DECAY
                        Weight decay in the optimizer
  --seed SEED           Random seed.
  --threads THREADS     Maximum number of threads to use.
  --dataset_path DATASET_PATH
                        Path to the dataset folder.
```

Once you are familiar with the arguments, run the script as follows:

```
python3 surya_training.py [-h] [--lang LANG] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--epochs_warmup EPOCHS_WARMUP]
                         [--label_smoothing LABEL_SMOOTHING] [--lr LR] [--lr_decay LR_DECAY] [--weight_decay WEIGHT_DECAY] [--seed SEED]
                         [--threads THREADS] [--dataset_path DATASET_PATH]
```

Or simply (if you want to use the values specified in the script):

```bash
python3 surya_training.py
```

### Training Progress and Results

All outputs of the training are saved in the `logs` directory. Each run will create a new subdirectory, with the timestamp and arguments as its name. The directory contains:

1. `tensorboard` training logs, which can be visualized using

    ```bash
    tensorboard --logdir logs
    ```

    and following the instructions in the terminal,

1. saved `torch` recognition model in `finetuned_model/model.pt`.

## Inference

To test the fine-tuned model, place the `model.pt` to an accessible location. The script `inference.py` provides a simple example usage.

## How it Works

### Surya OCR Architecture

The Surya OCR model consists of two modules: *text detector* (for segmenting text into rectangle-like quadrilaterals containing a single line) and *text recognizer* (for recognizing a single line of text from a picture). In this project, we are only interested in fine-tuning the latter.

### Recognizer Architecture

The recognizer is an encoder-decoder vision/text Transformer.

#### Encoder

The encoder consists two parts `text_encoder` and `encoder` (for images). However, the model does not actually encode any text. Insted, a range of `0` to `127` is fed into it. The reason behid this is unclear. During both inference and training, the ranges and images are fed into the corresponding modules and the resulting hidden states are saved and used in the `decoder`.

#### Tokenization

Together with the hidden states from the encoder, the decoder works with tokens, which correspond to sub-word units (and special symbols). The output of the decoder are logits for each of the token from vocabulary. Surya OCR uses a modified version of the ByT5 tokenizer. On top of splitting the text into tokens, it adds two special tokens to the beginning (beginning of stream token and a language-specific marker) and one token to the end (EOS token).

#### Decoder

During inference, only the first special tokens are presented to the decoder. The decoder then makes predictions of the following token for each sample in the batch and caches the tokens seen so far. Next, the most likely token for each sample is selected from the prediction and and fed into the model auto-regressively. This is repeated until it generates the `EOS` token token is generated for each sample.

During training, a batch of target tokens is perpared and padded to the length of the longest sequence. An attention mask is also provided and its corresponding parts are fed into the decoder with the tokens. The padded token matrix is fed into the decoder and the resulting logits are sliced out and returned. Note that during training, the decoder is *not* auto-regressive.

The true tokens without the first special tokenes are used as target to compute the cross-entropy loss. The loss is then back-propagated using standard `torch` functions.


## Known Issues

### Learning rate problems

The learning rate seems to be secretly increased by a small number. Setting learning rate to 0 will still result in the model training. <s>This implicit learning rate is high enough to destroy the weights when the `AdamW` optimizer is used. For this reason, we use plain `SGD` instead.</s> AdamW is now functional, thanks to @Dagamies