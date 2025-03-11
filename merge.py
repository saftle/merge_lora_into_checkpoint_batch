import os
import glob
import pathlib
import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import sd_mecha
import json
import argparse


# Define a debug function to analyze model files
def print_model_stats(model_path: pathlib.Path, name="Model", print_keys=False, max_keys=10):
    """Print statistics about a model file including size and parameter count"""
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return {}

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    key_info = {}

    try:
        # For safetensors
        if model_path.suffix == '.safetensors':
            with safe_open(str(model_path), framework="pt") as f:
                keys = list(f.keys())

                # Get tensor shapes properly from safetensors
                tensor_info = {}
                for key in keys:
                    # Get the actual tensor and its shape
                    tensor = f.get_tensor(key)
                    tensor_info[key] = list(tensor.shape)

                key_info = {key: tensor_info[key] for key in keys if key in tensor_info}
                param_count = sum(np.prod(shape) for shape in tensor_info.values())
                key_count = len(keys)

                # Check for key groups
                key_groups = {}
                for key in keys:
                    prefix = key.split('.')[0] if '.' in key else 'other'
                    key_groups.setdefault(prefix, 0)
                    key_groups[prefix] += 1

        # For ckpt
        elif model_path.suffix == '.ckpt':
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            keys = list(state_dict.keys())
            key_info = {key: list(state_dict[key].shape) for key in keys}
            param_count = sum(torch.numel(tensor) for tensor in state_dict.values())
            key_count = len(keys)

            # Check for key groups
            key_groups = {}
            for key in keys:
                prefix = key.split('.')[0] if '.' in key else 'other'
                key_groups.setdefault(prefix, 0)
                key_groups[prefix] += 1
    except Exception as e:
        print(f"Error analyzing {model_path}: {str(e)}")
        return {}

    print(f"\n{name} Stats:")
    print(f"  - File size: {file_size_mb:.2f} MB")
    print(f"  - Number of keys: {key_count}")
    print(f"  - Number of parameters: {param_count:,}")
    print(f"  - Key groups: {json.dumps(key_groups, indent=2)}")

    if print_keys and keys:
        print(f"  - Sample keys ({min(max_keys, len(keys))} of {len(keys)}):")
        for key in sorted(keys)[:max_keys]:
            shape_str = str(key_info[key]).replace('(', '[').replace(')', ']')
            print(f"    * {key}: {shape_str}")

    return key_info


# Function to compare keys between models
def compare_models(model1_info, model2_info, name1="Model1", name2="Model2", max_missing=20):
    """Compare keys between two models and report differences"""
    if not model1_info or not model2_info:
        print("Cannot compare - one or both models couldn't be analyzed")
        return

    keys1 = set(model1_info.keys())
    keys2 = set(model2_info.keys())

    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1

    print(f"\nComparing {name1} vs {name2}:")
    print(f"  - Keys in {name1} but not in {name2}: {len(missing_in_2)}")
    if missing_in_2 and len(missing_in_2) <= max_missing:
        print("    Missing keys:")
        for key in sorted(missing_in_2):
            print(f"    * {key}: {model1_info[key]}")

    print(f"  - Keys in {name2} but not in {name1}: {len(missing_in_1)}")
    if missing_in_1 and len(missing_in_1) <= max_missing:
        print("    New keys:")
        for key in sorted(missing_in_1):
            print(f"    * {key}: {model2_info[key]}")


# Parse command line arguments
parser = argparse.ArgumentParser(description='Merge SD models with LoRA with debugging')
parser.add_argument('--config', type=str, default='sd1-ldm',
                    help='Model configuration (sd1-ldm, sd1-supermerger_blocks, sd1-kohya, sd1-ldm-complete)')
parser.add_argument('--checkpoint-folder', type=str, required=True, help='Enable detailed debugging')
parser.add_argument('--output-folder', type=str, required=True, help='Enable detailed debugging')
parser.add_argument('--lora-path', type=str, required=True, help='Enable detailed debugging')
parser.add_argument('--lora-alpha', type=float, default=1, required=True, help='Alpha for LoRA merging')
parser.add_argument('--debug', action='store_true', help='Enable detailed debugging')
parser.add_argument('--debug-keys', action='store_true', help='Print all keys in debugging')
parser.add_argument('--skip-cleanup', action='store_true', help='Skip temporary file cleanup')

args = parser.parse_args()

# Convert string paths to Path objects
from pathlib import Path
checkpoint_folder = Path(args.checkpoint_folder)
output_folder = Path(args.output_folder)
lora_path = Path(args.lora_path)

temp_folder = output_folder / "temp"
debug_folder = output_folder / "debug"

# Ensure output folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)
if args.debug:
    os.makedirs(debug_folder, exist_ok=True)

# STEP 1: Convert all .ckpt files to .safetensors format
ckpt_files = list(checkpoint_folder.glob("*.ckpt"))
safetensors_files = list(checkpoint_folder.glob("*.safetensors"))

print(f"Found {len(ckpt_files)} .ckpt files to convert")
print(f"Found {len(safetensors_files)} existing .safetensors files")
print(f"Using model config: {args.config}")

# Convert .ckpt files to .safetensors
for ckpt_file in ckpt_files:
    filename = ckpt_file.name
    name_without_ext = ckpt_file.stem
    safetensors_path = temp_folder / f"{name_without_ext}.safetensors"
    
    if os.path.exists(safetensors_path):
        print(f"Skipping {filename}, safetensors version already exists in temp folder")
        continue
        
    print(f"Converting {filename} to safetensors format...")
    # Debug: Analyze original CKPT file
    if args.debug:
        original_info = print_model_stats(ckpt_file, f"Original CKPT ({filename})", args.debug_keys)
    try:
        with torch.no_grad():
            map_location = torch.device('cpu')
            # Set weights_only=False explicitly to handle PyTorch Lightning checkpoints
            checkpoint = torch.load(ckpt_file, map_location=map_location, weights_only=False)
            weights = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            # Debug: Check what keys might be in EMA
            if args.debug and "state_dict" in checkpoint:
                ema_keys = [k for k in checkpoint.keys() if "ema" in k.lower()]
                if ema_keys:
                    print(f"  Found possible EMA keys in checkpoint: {ema_keys}")
                    # Try to include EMA weights in the safetensors file
                    for ema_key in ema_keys:
                        if isinstance(checkpoint[ema_key], dict):
                            for k, v in checkpoint[ema_key].items():
                                if isinstance(v, torch.Tensor):
                                    weights[f"ema.{k}"] = v
            print(f"Saving to {safetensors_path}...")
            save_file(weights, safetensors_path)
            print(f"Conversion complete: {safetensors_path}")
            # Debug: Analyze converted Safetensors file
            if args.debug:
                converted_info = print_model_stats(safetensors_path, f"Converted Safetensors ({filename})",
                                                args.debug_keys)
                compare_models(original_info, converted_info,
                            f"Original CKPT ({filename})",
                            f"Converted Safetensors ({filename})")
    except Exception as e:
        print(f"Error converting {filename}: {str(e)}")
        continue

# Copy existing safetensors files to temp folder if they don't already exist there
for st_file in safetensors_files:
    temp_path = temp_folder / st_file.name
    if not temp_path.exists():
        try:
            import shutil
            shutil.copy(st_file, temp_path)
            print(f"Copied {st_file.name} to temp folder")
        except Exception as e:
            print(f"Error copying {st_file.name} to temp folder: {str(e)}")
            continue

# Get all safetensors files from temp folder
all_safetensors_files = list(temp_folder.glob("*.safetensors"))
print(f"Total safetensors files to process: {len(all_safetensors_files)}")

# STEP 2: Process each safetensors file with the LoRA
# Load LoRA with explicit configuration
print(f"Loading LoRA from {lora_path}...")
lora_model = sd_mecha.model(lora_path)

# Debug: Analyze LoRA file
if args.debug:
    lora_info = print_model_stats(lora_path, f"LoRA Model", args.debug_keys)

# Process each checkpoint
for checkpoint_file in all_safetensors_files:
    try:
        filename = checkpoint_file.name
        final_output_path = output_folder / checkpoint_file.name

        # Skip if output already exists
        if os.path.exists(final_output_path):
            print(f"\nSkipping {filename} (output already exists)")
            continue

        print(f"\nProcessing {filename}...")

        # Debug: Analyze input checkpoint
        if args.debug:
            checkpoint_info = print_model_stats(checkpoint_file, f"Input Checkpoint ({filename})", args.debug_keys)

        # Load safetensors checkpoint
        checkpoint = sd_mecha.model(checkpoint_file)
        checkpoint_ema_only = sd_mecha.exchange_ema(sd_mecha.pick_component(checkpoint, "ema"))
        checkpoint_non_ema = sd_mecha.omit_component(checkpoint, "ema")

        # Convert LoRA to the same format as the checkpoint
        print(f"Converting LoRA to match checkpoint format...")
        diff = sd_mecha.convert(lora_model, checkpoint)

        # Create recipe for merging with LoRA delta
        print(f"Creating merge recipe...")
        checkpoint = sd_mecha.model(checkpoint_file)
        diff = sd_mecha.convert(lora_model, checkpoint)
        diff = diff | sd_mecha.exchange_ema(diff)
        recipe = sd_mecha.add_difference(checkpoint, diff, alpha=args.lora_alpha) | checkpoint

        # Always merge on CPU for stability
        print(f"Merging on CPU (safer but slower)...")

        # Try different merge parameters to preserve all tensors
        sd_mecha.merge(
            recipe=recipe,
            merge_device="cpu",
            merge_dtype=None,
            output_device=None,
            output_dtype=None,
            threads=os.cpu_count(),  # Use fewer CPU cores
            total_buffer_size=2 ** 24,  # Smaller buffer
            strict_weight_space=False,
            output=final_output_path,
        )

        print(f"✓ Successfully saved merged model to {final_output_path}")

        # Debug: Analyze output model and compare with input
        if args.debug:
            output_info = print_model_stats(final_output_path, f"Output Model ({filename})", args.debug_keys)
            compare_models(checkpoint_info, output_info,
                           f"Input Checkpoint ({filename})",
                           f"Output Model ({filename})")

    except Exception as e:
        print(f"× Error processing {os.path.basename(checkpoint_file)}: {str(e)}")
        print("  Skipping to next file...")
        continue

# Clean up temporary safetensors files
if not args.skip_cleanup:
    print("\nCleaning up temporary files...")
    temp_safetensors_files = glob.glob(os.path.join(temp_folder, "*.safetensors"))
    for temp_file in temp_safetensors_files:
        try:
            os.remove(temp_file)
            print(f"Removed temporary file: {os.path.basename(temp_file)}")
        except Exception as e:
            print(f"Could not remove {os.path.basename(temp_file)}: {str(e)}")

print("\nProcessing complete. Check output folder for merged models.")
