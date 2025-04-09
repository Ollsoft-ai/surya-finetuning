import os
import torch
from safetensors.torch import save_file
from surya.model.recognition.config import SuryaOCRConfig, SuryaOCRDecoderConfig, DonutSwinConfig
from surya.model.recognition.tokenizer import Byt5LangTokenizer
from surya.model.recognition.encoderdecoder import OCREncoderDecoderModel

def save_model_to_safetensors():
    # 1. Allow custom class for safe loading
    torch.serialization.add_safe_globals([OCREncoderDecoderModel])

    # 2. Load your trained model safely
    model = torch.load("model.pt", weights_only=False)  # Required for custom classes

    # 3. Create proper config structure
    encoder_config = DonutSwinConfig()
    decoder_config = SuryaOCRDecoderConfig()

    config = SuryaOCRConfig(
        encoder=encoder_config.to_dict(),
        decoder=decoder_config.to_dict()
    )

    # 4. Initialize tokenizer
    tokenizer = Byt5LangTokenizer()

    # 5. Save all required files
    output_dir = "surya_model"
    os.makedirs(output_dir, exist_ok=True)

    # Save weights as safetensors
    save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(config.to_json_string())

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Model successfully saved to {output_dir}")

if __name__ == "__main__":
    save_model_to_safetensors()
