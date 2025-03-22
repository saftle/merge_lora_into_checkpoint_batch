# Intro

Ever wanted to bulk merge the same LoRA into 1000s of checkpoints!?! Now you can! Why? Because!

This uses the amazing merging framework sd-mecha by ljleb. Be sure to show him your support and to join his discord community!: https://github.com/ljleb/sd-mecha

# Create and activate a venv (optional)

python3 -m venv venv

source venv/bin/activate

# How to Install

pip install -r requirements.txt

# Example Usage

python merge.py --checkpoint-folder "/path/to/checkpoint_folder" --output-folder "/path/to/output_folder" --lora-path "/path/to/LoRA.safetensors" --lora-alpha 1.0

# Command line arguments

--config (sd-mecha Model configuration sd1-ldm, sd1-supermerger_blocks, sd1-kohya)

--checkpoint-folder

--output-folder

--lora-path

--lora-alpha (Alpha for LoRA merging)

--debug (Enable detailed debugging)

--debug-keys (Print all keys in debugging)

--skip-cleanup (Skip temporary file cleanup)

# Current limitations

- Only works on SD 1.5 for now. Will add more support later.

- There seems to be a problem when using GPU/CUDA merging, so hardcoded it to CPU for now.

- If an extra unused set of ema keys are found, they will be automatically stripped. Looking into ways to potentially keep these and merge them as well, in order to keep models as-is even if they are fundamentally useless junk in a model file.
