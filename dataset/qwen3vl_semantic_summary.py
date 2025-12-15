"""Qwen3-VL semantic summary generation for images.

This module provides functions to generate semantic summaries for images using Qwen3-VL model
from unsloth repository (unsloth/Qwen3-VL-2B-Instruct).
Semantic summaries are saved to CSV for later use.

Reference: https://huggingface.co/unsloth/Qwen3-VL-2B-Instruct
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from PIL import Image

try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    QWEN3VL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    print("Warning: transformers library not available. Qwen3-VL semantic summary generation will be disabled.")


BATCH_SIZE = 4  # Batch size for semantic summary generation (smaller due to VL model size)

# Semantic summary prompt template
SEMANTIC_SUMMARY_PROMPT = """Summarize the given image into a high-level semantic description.

Focus on the abstract attributes such as:
- object or product category
- functional purpose
- usage scenario
- user intent

Avoid describing low-level visual details.
Keep the summary concise."""

def _load_qwen3vl_model(device: torch.device):
    """Load Qwen3-VL model from unsloth repository using transformers.
    
    Uses unsloth/Qwen3-VL-2B-Instruct which includes unsloth chat template fixes.
    
    Args:
        device: Device to load model on
        
    Returns:
        Tuple of (model, processor)
    """
    if not QWEN3VL_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    print("Loading Qwen3-VL model from unsloth...")
    
    try:
        # Use unsloth's Qwen3-VL model (includes chat template fixes)
        model_name = "unsloth/Qwen3-VL-2B-Instruct"
        
        # Load processor and model
        # Note: Qwen3-VL requires latest transformers from source
        # pip install git+https://github.com/huggingface/transformers
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto" if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        
        if device.type == "cpu":
            model = model.to(device)
        
        model.eval()
        print(f"Qwen3-VL model loaded on {device}: {model_name}")
        return model, processor
        
    except Exception as e:
        print(f"Error loading unsloth/Qwen3-VL-2B-Instruct: {e}")
        # Try original Qwen repository as fallback
        try:
            model_name = "Qwen/Qwen3-VL-2B-Instruct"
            print(f"Trying original Qwen repository: {model_name}")
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype="auto" if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None,
                trust_remote_code=True,
            )
            if device.type == "cpu":
                model = model.to(device)
            model.eval()
            print(f"Qwen3-VL model loaded on {device}: {model_name}")
            return model, processor
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load Qwen3-VL model. Last error: {e2}\n"
                f"Note: Qwen3-VL requires latest transformers. Install with:\n"
                f"pip install git+https://github.com/huggingface/transformers"
            )


def generate_semantic_summaries(
    model,
    processor,
    device: torch.device,
    meta: Dict[int, Dict[str, Any]],
    num_items: int,
) -> Dict[int, str]:
    """Generate semantic summaries for all images in meta.
    
    Args:
        model: Qwen2-VL model
        processor: Qwen2-VL processor
        device: Device to run inference on
        meta: Dict {item_id: {image_path: str, ...}}
        num_items: Total number of items
        
    Returns:
        Dict {item_id: semantic_summary} - Semantic summaries for items with valid images
    """
    if not QWEN3VL_AVAILABLE:
        return {}
    
    # Collect items with valid images
    items_with_img = []
    for item_id in range(1, num_items + 1):
        info = meta.get(item_id, {})
        image_path = info.get("image_path") or info.get("image")
        if image_path and os.path.isfile(image_path):
            items_with_img.append((item_id, image_path))
    
    if not items_with_img:
        print("No images found for semantic summary generation")
        return {}
    
    print(f"Generating semantic summaries for {len(items_with_img)} images...")
    
    summaries = {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(items_with_img), BATCH_SIZE), desc="Qwen3 VL semantic summaries"):
            batch = items_with_img[i : i + BATCH_SIZE]
            batch_images = []
            batch_ids = []
            
            for item_id, path in batch:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    batch_ids.append(item_id)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            try:
                # Process each image individually (VL models typically process one at a time)
                for idx, (item_id, img) in enumerate(zip(batch_ids, batch_images)):
                    try:
                        # Build prompt with image
                        # Qwen3-VL uses specific format for image-text inputs
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": SEMANTIC_SUMMARY_PROMPT}
                                ]
                            }
                        ]
                        
                        # Prepare inputs using processor (Qwen3-VL API)
                        # apply_chat_template with tokenize=True returns dict with input_ids, etc.
                        inputs = processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        
                        # Generate summary
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                        )
                        
                        # Decode summary
                        # Trim input_ids from generated_ids
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                        ]
                        summary = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                        
                        summaries[item_id] = summary.strip()
                        
                    except Exception as e:
                        print(f"Error generating summary for item {item_id}: {e}")
                        continue
                
            except Exception as e:
                print(f"Error generating summaries for batch: {e}")
                continue
    
    print(f"Generated {len(summaries)} semantic summaries")
    return summaries


def maybe_generate_semantic_summaries(
    dataset,
    data: Dict[str, Any],
    args,
) -> Optional[Dict[int, str]]:
    """Generate Qwen3 VL semantic summaries if enabled and images are available.
    
    Args:
        dataset: Dataset instance
        data: Dataset data dict
        args: Arguments with use_image and generate_semantic_summary flags
        
    Returns:
        Dict {item_id: semantic_summary} or None if not generated
    """
    if not hasattr(args, 'generate_semantic_summary') or not args.generate_semantic_summary:
        return None
    
    if not args.use_image:
        print("Warning: generate_semantic_summary requires --use_image flag")
        return None
    
    if not QWEN3VL_AVAILABLE:
        print("Warning: Qwen3-VL not available. Skipping semantic summary generation.")
        print("Note: Qwen3-VL requires latest transformers. Install with:")
        print("pip install git+https://github.com/huggingface/transformers")
        return None
    
    meta = data.get("meta", {})
    if not meta:
        print("Warning: No metadata found. Skipping semantic summary generation.")
        return None
    
    num_items = max(meta.keys()) if meta else 0
    if num_items == 0:
        print("Warning: No items found. Skipping semantic summary generation.")
        return None
    
    # Check if summaries already exist
    preproc_folder = Path(dataset._get_preprocessed_folder_path())
    summaries_path = preproc_folder / "qwen3vl_semantic_summaries.pt"
    
    if summaries_path.exists():
        print(f"Loading existing semantic summaries from {summaries_path}")
        summaries = torch.load(summaries_path, map_location="cpu")
        return summaries
    
    # Generate summaries
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, processor = _load_qwen3vl_model(device)
    summaries = generate_semantic_summaries(model, processor, device, meta, num_items)
    
    # Save summaries to .pt file for caching (optional, CSV is the main storage)
    if summaries:
        summaries_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(summaries, summaries_path)
        print(f"Saved semantic summaries cache to {summaries_path}")
        print(f"Note: Semantic summaries will also be saved to CSV in dataset_single_export.csv")
    
    return summaries

