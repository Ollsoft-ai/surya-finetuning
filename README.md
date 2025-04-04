# Surya-OCR Fine-Tuning Script

This project publishes a roadmap to fine-tuning the [Surya OCR](https://github.com/VikParuchuri/surya) model.

Please help us solve the [known issues](./surya_finetuning/README.md#known-issues).

## Set up Requirements

Create a Python virtual environment, activate it, and install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Roadmap to Fine-Tuning

1. Create a scan dataset -- see [instructions](./line_dataset_generator/README.md#dataset-read-before-printing).
1. [Creating a line dataset](./line_dataset_generator/README.md).
1. [Fine-tuning the Surya model](./surya_finetuning/README.md).