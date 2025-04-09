import torch
import torchmetrics
import numpy as np
import argparse
import os
import datetime
import re
from functools import partial
from surya.model.recognition.model import load_model as load_rec_model, OCREncoderDecoderModel
from surya.model.recognition import tokenizer
from surya.model.recognition.processor import SuryaProcessor, load_processor as load_rec_processor
from surya.settings import settings
import torch.nn.functional as F
from PIL import Image

from trainable_module import TrainableModule
from image_text_dataset import ImageTextDataset

LEADING_META_TOKENS_COUNT = 2

parser = argparse.ArgumentParser()
# NOTE: IMPORTANT!!! change the language code below, see https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
parser.add_argument("--lang", default="cs", type=str, help="Language code of the documents")
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--epochs", default=7, type=int, help="Number of epochs.")
parser.add_argument("--epochs_warmup", default=1, type=int, help="Number of epochs during which the LR is increased linearly from zero to the specified value. Currently does nothing.")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing.")
parser.add_argument("--lr", default=5e-2, type=float, help="Learning rate.")
parser.add_argument("--lr_decay", default=False, type=bool, help="Use cosine decay? Currently does nothing.")
parser.add_argument("--weight_decay", default=5e-3, type=float, help="Weight decay in the optimizer. Currently does nothing")
parser.add_argument("--seed", default=43, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--early_stopping_patience", type=int, default=-1, help="Patience for early stopping (epochs without improvement). Default: -1 (disabled).")
parser.add_argument("--dataset_path", default="../dataset", type=str, help="Path to the dataset folder.")


class OllsoftRecModel(TrainableModule):
    def __init__(self, model: OCREncoderDecoderModel):
        super().__init__()

        self._model = model

    def forward(self, images: torch.Tensor, padded_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self._model.decoder.model._setup_cache(self._model.config, args.batch_size, self._model.device, self._model.dtype)
        self._model.text_encoder.model._setup_cache(self._model.config, args.batch_size, self._model.device, self._model.dtype)

        encoder_hidden_states = None
        batch_size = padded_tokens.shape[0]
        encoder_batch_size = batch_size // settings.RECOGNITION_ENCODER_BATCH_DIVISOR
        for z in range(0, images.shape[0], encoder_batch_size):
            encoder_pixel_values = images[z:min(z + encoder_batch_size, images.shape[0])]
            encoder_hidden_states_batch = self._model.encoder(pixel_values=encoder_pixel_values).last_hidden_state
            if encoder_hidden_states is None:
                encoder_hidden_states = encoder_hidden_states_batch
            else:
                encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_batch], dim=0)


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


        predictions = torch.zeros((padded_tokens.shape[0], padded_tokens.shape[1] - LEADING_META_TOKENS_COUNT, self._model.decoder.vocab_size), device=self._model.device)

        for slc in range(LEADING_META_TOKENS_COUNT, padded_tokens.shape[1]):
            if slc == LEADING_META_TOKENS_COUNT:
                data_slice = padded_tokens[:, :slc]
                mask_slice = mask[:, :slc]
                decoder_position_ids = torch.ones_like(data_slice[0, :], dtype=torch.int64, device=self._model.device).cumsum(0) - 1
                is_prefill = True
            else:
                data_slice = padded_tokens[:, slc - 1].unsqueeze(1)
                mask_slice = mask[:, slc - 1].unsqueeze(1)
                decoder_position_ids = decoder_position_ids[-1:] + 1
                max_position_id = torch.max(decoder_position_ids).item()
                decoder_position_ids = torch.ones_like(data_slice[0, :], dtype=torch.int64, device=self._model.device).cumsum(0) - 1 + max_position_id
                is_prefill = False

            preds = self._model.decoder(
                input_ids=data_slice,
                attention_mask=mask_slice,
                encoder_hidden_states=encoder_text_hidden_states,
                cache_position=decoder_position_ids,
                use_cache=True,
                prefill=is_prefill
            )

            predictions[:, slc - LEADING_META_TOKENS_COUNT, :] = preds["logits"][:, -1, :] # the next token predictions are saved in the last "column", therefore [:, -1, :]

        return torch.permute(predictions, (0, 2, 1))

    def save_model(self, folder_path: str):
        with open(os.path.join(folder_path, "model.pt"), "wb") as model_file:
            torch.save(self._model, model_file)



# Currently not used
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

# Added early stopping patience and save best model in training
class CheckpointCallback:
    def __init__(
        self,
        metric_name="dev_acc",
        mode="max",
        patience=-1,          # -1 = disabled, >0 = enabled
        min_delta=0.001,
        restore_best_weights=True
    ):
        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.best_checkpoint_dir = None
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, model, epoch, logs):
        if not hasattr(model, 'save_model'):
            return

        current_metric = logs.get(self.metric_name)
        if current_metric is None:
            print(f"Warning: Metric '{self.metric_name}' not found. Skipping checkpoint.")
            return

        # Check for improvement
        if self.mode == "max":
            is_better = (current_metric - self.best_metric) > self.min_delta
        else:
            is_better = (self.best_metric - current_metric) > self.min_delta

        if is_better:
            self.wait = 0
            self.best_metric = current_metric
            self.best_epoch = epoch + 1

            # Save best model
            if self.best_checkpoint_dir is not None:
                import shutil
                shutil.rmtree(self.best_checkpoint_dir, ignore_errors=True)

            self.best_checkpoint_dir = os.path.join(model.logdir, "best_model")
            os.makedirs(self.best_checkpoint_dir, exist_ok=True)
            model.save_model(self.best_checkpoint_dir)
            print(f"New best model (epoch {self.best_epoch}, {self.metric_name}={self.best_metric:.4f})")
        else:
            self.wait += 1
            if self.patience > 0:  # Only log if early stopping is enabled
                print(f"No improvement for {self.wait}/{self.patience} epochs.")

        # Early stopping check (skip if patience=-1)
        if self.patience > 0 and self.wait >= self.patience:
            self.stopped_epoch = epoch + 1
            print(f"\nEarly stopping at epoch {self.stopped_epoch}.")
            if self.restore_best_weights and self.best_checkpoint_dir:
                print("Restoring best model weights...")
                best_model_path = os.path.join(self.best_checkpoint_dir, "model.pt")
                model.load_state_dict(torch.load(best_model_path))
            raise KeyboardInterrupt

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

    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    # Freeze all model parameters (https://github.com/VikParuchuri/surya/issues/40#issuecomment-1962739520)
    for param in rec_model.parameters():
        param.requires_grad = False

    # Load the data.
    dataset = ImageTextDataset(args.dataset_path)
    dataset_train, dataset_dev = torch.utils.data.random_split(dataset, [0.9, 0.1])
    print("Train dataset length:", len(dataset_train))

    tok = tokenizer.Byt5LangTokenizer()

    prepare_batch_partial = partial(prepare_batch, tok, rec_processor)

    train = torch.utils.data.DataLoader(dataset_train, args.batch_size, shuffle=True, collate_fn=prepare_batch_partial)
    dev = torch.utils.data.DataLoader(dataset_dev, args.batch_size, shuffle=True, collate_fn=prepare_batch_partial)

    custom_model = OllsoftRecModel(rec_model)
    # TODO: fix the problem with implicit learning rate (see README.md)
    #       SGD is used necause it is less sensitive to learning rate
    #optimizer = torch.optim.AdamW(custom_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(custom_model.parameters(), lr=args.lr)
    #schedule = CosineWithWarmup(optimizer, args, train)
    custom_model.configure(
        optimizer=optimizer,
        loss=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=tok.pad_id),
        # schedule=schedule,
        metrics={"acc": torchmetrics.Accuracy(task="multiclass", num_classes=65792, ignore_index=tok.pad_id)},
        logdir=args.logdir,
    )
    callbacks = [
        CheckpointCallback(
            metric_name="dev_acc",
            mode="max",
            patience=args.early_stopping_patience,  # Set via command line
            min_delta=0.001,
            restore_best_weights=True
        )
    ]
    try:
        #custom_model.fit(train, epochs=args.epochs, dev=dev, callbacks=[CheckpointCallback()])
        custom_model.fit(
            train,
            epochs=args.epochs,
            dev=dev,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        pass

    os.mkdir(os.path.join(args.logdir, "finetuned_model"))
    custom_model.save_model(os.path.join(args.logdir, "finetuned_model"))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
