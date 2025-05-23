import torch
import torchmetrics
import numpy as np
import argparse
import os
import datetime
import re
from functools import partial

from surya.recognition import RecognitionPredictor
from surya.recognition.model.encoderdecoder import OCREncoderDecoderModel
from surya.recognition import tokenizer, SuryaProcessor

from surya.settings import settings
import torch.nn.functional as F
from PIL import Image

from trainable_module import TrainableModule
from image_text_dataset import ImageTextDataset

LEADING_META_TOKENS_COUNT = 2

parser = argparse.ArgumentParser()
# NOTE: IMPORTANT!!! change the language code below, see https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
parser.add_argument("--lang", default="cs", type=str, help="Language code of the documents")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=7, type=int, help="Number of epochs.")
parser.add_argument("--epochs_warmup", default=1, type=int, help="Number of epochs during which the LR is increased linearly from zero to the specified value.")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing.")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")
parser.add_argument("--lr_decay", default=True, type=bool, help="Use cosine decay?")
parser.add_argument("--weight_decay", default=5e-3, type=float, help="Weight decay in the optimizer")
parser.add_argument("--seed", default=43, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")

parser.add_argument("--dataset_path", default="../dataset", type=str, help="Path to the dataset folder.")


class OllsoftRecModel(TrainableModule):
    def __init__(self, model: OCREncoderDecoderModel):
        super().__init__()

        self._model = model

    def forward(self, images: torch.Tensor, padded_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._model.text_encoder.model._setup_cache(self._model.config, args.batch_size, self._model.device, self._model.dtype)

        encoder_hidden_states = self._model.encoder(pixel_values=images).last_hidden_state

        # I am not sure what is the purpose of the text_encoder. We only ever feed it ranges of numbers from 0 to 127
        text_encoder_input_ids = torch.arange(
            self._model.text_encoder.config.query_token_count,
            device=self._model.device,
            dtype=torch.long
        ).unsqueeze(0).expand(encoder_hidden_states.size(0), -1)
        # ???
        encoder_text_hidden_states = self._model.text_encoder(
            input_ids=text_encoder_input_ids,
            cache_position=None,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            use_cache=False
        ).hidden_states

        self._model.decoder.model._setup_cache(self._model.config, args.batch_size, self._model.device, self._model.dtype)
        preds = self._model.decoder(
                input_ids=padded_tokens,
                attention_mask=mask,
                encoder_hidden_states=encoder_text_hidden_states,
                use_cache=False,
                prefill=False
            )
        
        predictions = preds["logits"][:, LEADING_META_TOKENS_COUNT - 1: -1]      
        return torch.permute(predictions, (0, 2, 1))

    def save_model(self, folder_path: str):
        with open(os.path.join(folder_path, "model.pt"), "wb") as model_file:
            torch.save(self._model, model_file)
            
            
class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, args, train):
        self._warmup_steps = args.epochs_warmup * len(train)
        self._decay_steps = args.lr_decay and (args.epochs - args.epochs_warmup) * len(train)
        super().__init__(optimizer, self.current_learning_rate)

    def current_learning_rate(self, step):
        assert step >= 0 and (not self._decay_steps or step <= self._warmup_steps + self._decay_steps)
        return step / self._warmup_steps if step < self._warmup_steps else \
            0.5 * (1 + np.cos(np.pi * (step - self._warmup_steps) / self._decay_steps)) if self._decay_steps else 1

def prepare_batch(tok: tokenizer.Byt5LangTokenizer, processor: SuryaProcessor, data):
    images, labels = zip(*data)

    tokenized = tok(labels, [[args.lang]]*len(labels))["input_ids"]
    # append the eos token to each sample
    for tokenized_label in tokenized:
        tokenized_label.append(tok.eos_id)

    labels_tokenized = torch.nn.utils.rnn.pad_sequence([torch.tensor(t) for t in tokenized], batch_first=True, padding_value=tok.pad_id)
    
    target = labels_tokenized[:, LEADING_META_TOKENS_COUNT:]

    mask = labels_tokenized != tok.pad_id
    images_processed = processor(text=[""] * len(images), images=images, langs=[[args.lang]]*len(images))
    pixel_values = images_processed["pixel_values"]
    pixel_values = torch.tensor(np.stack(pixel_values, axis=0), dtype=torch.float16, device="cuda")


    return ((pixel_values, labels_tokenized, mask), target)
    
def main(args: argparse.Namespace) -> None:
    print(f"WARNING: the language is set to {args.lang}. Make sure this is your language.")
    # Set the random seed and the number of threads.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    dir_name = "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ).replace("/", r"_")
    args.logdir = os.path.join("logs", dir_name)

    predictor = RecognitionPredictor()
    rec_model = predictor.model
    rec_processor = predictor.processor

    # Load the data.
    dataset = ImageTextDataset(args.dataset_path)
    dataset_train, dataset_dev = torch.utils.data.random_split(dataset, [0.9, 0.1])
    print("Train dataset length:", len(dataset_train))

    tok = tokenizer.Byt5LangTokenizer()

    prepare_batch_partial = partial(prepare_batch, tok, rec_processor)

    train = torch.utils.data.DataLoader(dataset_train, args.batch_size, shuffle=True, collate_fn=prepare_batch_partial)
    dev = torch.utils.data.DataLoader(dataset_dev, args.batch_size, shuffle=True, collate_fn=prepare_batch_partial)

    custom_model = OllsoftRecModel(rec_model)
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-5)
    schedule = CosineWithWarmup(optimizer, args, train)
    custom_model.configure(
        optimizer=optimizer,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=tok.pad_id),
        schedule=schedule,
        metrics={"acc": torchmetrics.Accuracy(task="multiclass", num_classes=65792, ignore_index=tok.pad_id)},
        logdir=args.logdir,
    )
    try:
        custom_model.fit(train, epochs=args.epochs, dev=dev)
    except KeyboardInterrupt:
        pass

    os.mkdir(os.path.join(args.logdir, "finetuned_model"))
    custom_model.save_model(os.path.join(args.logdir, "finetuned_model"))
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
