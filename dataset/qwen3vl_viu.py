import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from PIL import Image
from .reduce_viu import reduce_viu_for_reranker

try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from unsloth import FastVisionModel
    QWEN3VL_AVAILABLE = True
    FAST_VISION_MODEL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    FAST_VISION_MODEL_AVAILABLE = False
    # Try to import separately
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        QWEN3VL_AVAILABLE = True
    except ImportError:
        QWEN3VL_AVAILABLE = False
    try:
        from unsloth import FastVisionModel
        FAST_VISION_MODEL_AVAILABLE = True
    except ImportError:
        FAST_VISION_MODEL_AVAILABLE = False
    
    if not QWEN3VL_AVAILABLE or not FAST_VISION_MODEL_AVAILABLE:
        print("Warning: transformers or unsloth FastVisionModel not available. Qwen3-VL VIU generation will be disabled.")




# VIU prompt template
VIU_PROMPT = """You are a visual product attribute extractor.

Describe ONLY what is clearly visible on the product packaging in the image.
If any field is not readable or uncertain, write exactly: "Unknown".
Do NOT guess. Do NOT infer. Do NOT explain.

Return EXACTLY the following bullet list, in this exact order.
Do NOT add extra text, notes, explanations, or commentary.

- Product name:
- Category:
- Type/form:
- Brand:
- Packaging (primary color, secondary color, container type, closure type):
- Size/volume:
- On-pack claims:

Rules (must follow):
1) Product name:
   - Must be the specific name printed on the package.
   - If only a product line/series is visible, write that.
   - Do NOT use a generic type as a name (e.g., "Shampoo", "Conditioner").
   - If unclear, write "Unknown".

2) Category and Type/form:
   - ONLY if the exact words appear on the package (e.g., "shampoo", "conditioner", "cleanser").
   - If not explicitly written, write "Unknown".

3) Brand:
   - ONLY if clearly readable on the package.
   - Otherwise write "Unknown".

4) Packaging:
   - Container type must be one of:
     bottle, tube, jar, pump bottle, spray bottle, aerosol can, pouch
   - Closure type must be one of:
     flip-top cap, screw cap, pump, trigger spray, spray nozzle, aerosol
   - If any part is unclear, write "Unknown" for that part.

5) Size/volume:
   - ONLY if explicitly printed (e.g., "12 fl oz", "400 mL").
   - Otherwise write "Unknown".

6) On-pack claims:
   - Copy short phrases that appear on the package (verbatim or near-verbatim).
   - List each claim AT MOST ONCE.
   - Maximum 8 claims.
   - If none are readable, write exactly: "Unknown".

Strict output constraints:
- No repetition of words or phrases.
- No explanations, notes, or phrases like "inferred", "likely", "based on".
- No extra lines such as "The above fields…" or "Formatted exactly…".
- If output begins to repeat or loop, STOP and output "Unknown" for the affected field.
"""

def _load_qwen3vl_model(device: torch.device, use_quantization: bool = True):
    """Load Qwen3-VL model using Unsloth FastVisionModel for optimized inference.
    
    Uses unsloth FastVisionModel which provides:
    - Automatic 4-bit quantization
    - Optimized inference speed
    - Better memory efficiency
    - Chat template fixes
    
    Args:
        device: Device to load model on
        use_quantization: Whether to use 4-bit quantization (default: True)
        Note: 4-bit quantization is enabled by default for all LLM models to save memory
        
    Returns:
        Tuple of (model, processor)
    """
    if not QWEN3VL_AVAILABLE:
        raise ImportError("transformers library is required. Install with: pip install transformers")
    
    # Use unsloth's Qwen3-VL model (includes unsloth optimizations and chat template fixes)
    model_name = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
    
    # Load processor (still need from transformers)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Try to use FastVisionModel from Unsloth (preferred method)
    if FAST_VISION_MODEL_AVAILABLE:
        print("Loading Qwen3-VL model using Unsloth FastVisionModel with 4-bit quantization...")
        try:
            # Get max_seq_length from config if available
            try:
                from config import arg
                max_seq_length = getattr(arg, 'qwen_max_seq_length', 2048)
            except ImportError:
                max_seq_length = 2048  # Default fallback
            
            # Load model with FastVisionModel (automatically handles 4-bit quantization)
            model, _ = FastVisionModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
                load_in_4bit=use_quantization and device.type == "cuda",  # 4-bit quantization enabled by default
                use_gradient_checkpointing="unsloth",  # Memory efficient
            )
            
            model.eval()
            print(f"Qwen3-VL model loaded on {device} using Unsloth FastVisionModel: {model_name}")
            return model, processor
            
        except Exception as e:
            print(f"Warning: Failed to load with FastVisionModel: {e}")
            print("Falling back to standard transformers API...")
            # Fall through to transformers fallback
    
    # Fallback: Use standard transformers API (if FastVisionModel not available or failed)
    print("Loading Qwen3-VL model using standard transformers API...")
    try:
        # Setup 4-bit quantization manually if needed
        quantization_config = None
        if use_quantization and device.type == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("Using 4-bit quantization for Qwen3-VL model (via transformers)")
            except ImportError:
                print("Warning: bitsandbytes not available. Install with: pip install bitsandbytes")
                quantization_config = None
        
        # Load model with transformers API
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto" if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        
        if device.type == "cpu":
            model = model.to(device)
        
        model.eval()
        print(f"Qwen3-VL model loaded on {device} (from Unsloth repository, using transformers API): {model_name}")
        return model, processor
        
    except Exception as e:
        print(f"Error loading unsloth/Qwen3-VL-2B-Instruct: {e}")
        # Try original Qwen repository as fallback
        try:
            model_name = "Qwen/Qwen3-VL-2B-Instruct"
            print(f"Trying original Qwen repository: {model_name}")
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            # Try FastVisionModel first if available
            if FAST_VISION_MODEL_AVAILABLE:
                try:
                    try:
                        from config import arg
                        max_seq_length = getattr(arg, 'qwen_max_seq_length', 2048)
                    except ImportError:
                        max_seq_length = 2048
                    
                    model, _ = FastVisionModel.from_pretrained(
                        model_name=model_name,
                        max_seq_length=max_seq_length,
                        dtype=torch.float16 if device.type == "cuda" else torch.float32,
                        load_in_4bit=use_quantization and device.type == "cuda",
                        use_gradient_checkpointing="unsloth",
                    )
                    model.eval()
                    print(f"Qwen3-VL model loaded on {device} using FastVisionModel: {model_name}")
                    return model, processor
                except Exception:
                    pass  # Fall through to transformers API
            
            # Fallback to transformers API
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
                f"pip install git+https://github.com/huggingface/transformers\n"
                f"For Unsloth optimizations, install: pip install unsloth[colab-new]"
            )


def generate_viu(
    model,
    processor,
    device: torch.device,
    meta: Dict[int, Dict[str, Any]],
    num_items: int,
    batch_size: int = 4,
    use_torch_compile: bool = False,
    max_new_tokens: int = 64,
    preload_all_images: bool = False,
) -> Dict[int, str]:
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
        print("No images found for VIU generation")
        return {}
    
    print(f"Generating VIU for {len(items_with_img)} images...")
    print(f"Using batch size: {batch_size}, max_new_tokens: {max_new_tokens}")
    
    # Helper function to load and resize image (for parallel processing)
    def load_and_resize_image(item_id_path):
        """Load and resize image. Returns (item_id, img) or None if error."""
        item_id, path = item_id_path
        try:
            # If path is already a PIL Image (preloaded), return it
            if isinstance(path, Image.Image):
                return (item_id, path)
            
            img = Image.open(path).convert("RGB")
            # Resize image for Qwen3-VL (max 448px on longer side)
            width, height = img.size
            max_size = 448
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return (item_id, img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None
    
    # Pre-load all images if requested (faster but uses more RAM)
    if preload_all_images:
        print("Pre-loading all images into memory (this may use significant RAM)...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        preloaded_images = {}
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(load_and_resize_image, (item_id, path)): item_id 
                      for item_id, path in items_with_img}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-loading images"):
                result = future.result()
                if result is not None:
                    item_id, img = result
                    preloaded_images[item_id] = img
        print(f"Pre-loaded {len(preloaded_images)} images into memory")
        # Replace items_with_img with preloaded data (use Image objects as "paths")
        items_with_img = [(item_id, preloaded_images[item_id]) for item_id in preloaded_images.keys()]
    
    # Compile model if requested (PyTorch 2.0+)
    if use_torch_compile and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile() for faster inference...")
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}. Continuing without compilation.")
    
    summaries = {}
        # -----------------------------
    # Option A decoding config (strict / stable)
    # -----------------------------
    gen_cfg = dict(
        do_sample=False,
        num_beams=1,
        use_cache=True,
        repetition_penalty=1.10,
        no_repeat_ngram_size=4,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    # If caller passes a too-small max_new_tokens, Option A should override to a safer default
    if max_new_tokens is None or max_new_tokens < 120:
        max_new_tokens = 160

    
    # Pre-load all images in parallel (to reduce I/O bottleneck)
    print("Pre-loading images in parallel to reduce I/O bottleneck...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Pre-load next batch in background while processing current batch
    next_batch_future = None
    
    with torch.no_grad():
        # Process in batches with parallel image loading
        for i in tqdm(range(0, len(items_with_img), batch_size), desc="Qwen3 VL VIU"):
            batch = items_with_img[i : i + batch_size]
            
            # Wait for previous batch loading to complete (if exists)
            if next_batch_future is not None:
                batch_images_data = next_batch_future.result()
            else:
                # Load current batch images in parallel
                batch_images_data = []
                with ThreadPoolExecutor(max_workers=min(8, len(batch))) as executor:
                    futures = {executor.submit(load_and_resize_image, item): item[0] for item in batch}
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            batch_images_data.append(result)
            
            # Pre-load next batch in background (if not last batch)
            # This overlaps I/O with GPU computation to improve GPU utilization
            if i + batch_size < len(items_with_img):
                next_batch = items_with_img[i + batch_size : i + 2 * batch_size]
                next_executor = ThreadPoolExecutor(max_workers=min(8, len(next_batch)))
                next_batch_future = next_executor.submit(
                    lambda batch=next_batch: [
                        result for result in 
                        (load_and_resize_image(item) for item in batch)
                        if result is not None
                    ]
                )
            
            if not batch_images_data:
                continue
            
            batch_ids = [item_id for item_id, _ in batch_images_data]
            batch_images = [img for _, img in batch_images_data]
            
            try:
                # Try batch processing: process multiple images together if possible
                # Qwen3-VL may support batch processing with list of messages
                try:
                    # Attempt batch processing: create list of messages for all images
                    batch_messages = []
                    for img in batch_images:
                        batch_messages.append([
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": VIU_PROMPT}
                                ]
                            }
                        ])
                    
                    # Try to process batch (may not work if Qwen3-VL doesn't support it)
                    # If this fails, fall back to sequential processing
                    try:
                        # Process batch: apply_chat_template may support list of message lists
                        batch_inputs = processor.apply_chat_template(
                            batch_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        
                        # Process vision info if needed (Qwen3-VL may require this)
                        if hasattr(processor, 'process_vision_info') and 'vision_info' in batch_inputs:
                            vision_info = processor.process_vision_info(batch_inputs['vision_info'])
                            batch_inputs['vision_info'] = vision_info
                        
                        # Move to device - handle nested structures
                        def move_to_device(obj, dev):
                            """Recursively move tensors to device."""
                            if isinstance(obj, torch.Tensor):
                                return obj.to(dev)
                            elif isinstance(obj, dict):
                                return {k: move_to_device(v, dev) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return type(obj)(move_to_device(item, dev) for item in obj)
                            elif hasattr(obj, 'to'):  # Handle other objects with .to() method
                                try:
                                    return obj.to(dev)
                                except:
                                    return obj
                            else:
                                return obj
                        
                        batch_inputs = move_to_device(batch_inputs, device)
                        
                        # Ensure all tensor values are on device (double-check)
                        # Use model.device to ensure consistency
                        model_device = next(model.parameters()).device
                        for key, value in batch_inputs.items():
                            if isinstance(value, torch.Tensor) and value.device != model_device:
                                batch_inputs[key] = value.to(model_device)
                            elif isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, torch.Tensor) and sub_value.device != model_device:
                                        value[sub_key] = sub_value.to(model_device)
                            elif isinstance(value, (list, tuple)):
                                for idx, item in enumerate(value):
                                    if isinstance(item, torch.Tensor) and item.device != model_device:
                                        value[idx] = item.to(model_device)
                        
                        # Generate for batch
                        batch_generated_ids = model.generate(
                            **batch_inputs,
                            max_new_tokens=max_new_tokens,
                            **gen_cfg,
                        )

                        
                        # Decode batch
                        if isinstance(batch_inputs.get("input_ids"), torch.Tensor):
                            input_ids = batch_inputs["input_ids"]
                            if input_ids.dim() == 2:
                                # Batch processing worked!
                                batch_generated_ids_trimmed = [
                                    out_ids[len(in_ids):] 
                                    for in_ids, out_ids in zip(input_ids, batch_generated_ids)
                                ]
                                batch_summaries = processor.batch_decode(
                                    batch_generated_ids_trimmed, 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=False
                                )
                                
                                for idx, item_id in enumerate(batch_ids):
                                    if idx < len(batch_summaries):
                                        summaries[item_id] = batch_summaries[idx].strip()
                                continue  # Successfully processed batch, skip sequential processing
                    except Exception as batch_error:
                        # Batch processing failed, fall back to sequential
                        pass
                
                except Exception:
                    pass
                
                # Fallback: Process each image individually (if batch processing failed)
                # Optimized sequential processing with reduced overhead
                for idx, (item_id, img) in enumerate(zip(batch_ids, batch_images)):
                    try:
                        # Build prompt with image
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img},
                                    {"type": "text", "text": VIU_PROMPT}
                                ]
                            }
                        ]
                        
                        # Prepare inputs
                        inputs = processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        
                        # Process vision info if needed (Qwen3-VL may require this)
                        if hasattr(processor, 'process_vision_info') and 'vision_info' in inputs:
                            vision_info = processor.process_vision_info(inputs['vision_info'])
                            inputs['vision_info'] = vision_info
                        
                        # Move to device - handle nested structures (Qwen3-VL may have complex input format)
                        def move_to_device(obj, dev):
                            """Recursively move tensors to device."""
                            if isinstance(obj, torch.Tensor):
                                return obj.to(dev)
                            elif isinstance(obj, dict):
                                return {k: move_to_device(v, dev) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return type(obj)(move_to_device(item, dev) for item in obj)
                            elif hasattr(obj, 'to'):  # Handle other objects with .to() method
                                try:
                                    return obj.to(dev)
                                except:
                                    return obj
                            else:
                                return obj
                        
                        inputs = move_to_device(inputs, device)
                        
                        # Ensure all tensor values are on device (double-check)
                        # Use model.device to ensure consistency
                        model_device = next(model.parameters()).device
                        for key, value in inputs.items():
                            if isinstance(value, torch.Tensor) and value.device != model_device:
                                inputs[key] = value.to(model_device)
                            elif isinstance(value, dict):
                                for sub_key, sub_value in value.items():
                                    if isinstance(sub_value, torch.Tensor) and sub_value.device != model_device:
                                        value[sub_key] = sub_value.to(model_device)
                            elif isinstance(value, (list, tuple)):
                                for idx, item in enumerate(value):
                                    if isinstance(item, torch.Tensor) and item.device != model_device:
                                        value[idx] = item.to(model_device)
                        
                        # Generate summary with optimized settings
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            **gen_cfg,
                        )

                        
                        # Decode summary
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                        ]
                        summary = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0].strip()

                        sanitized, dbg = sanitize_viu(summary, max_claims=8, strict=True, keep_unknown=True)

                        if sanitized is None:
                            # Nếu fail: bạn có thể regenerate 1 lần (tùy bạn), hoặc lưu Unknown toàn bộ để tránh rác
                            # Ở đây chọn phương án an toàn: lưu skeleton Unknown
                            fallback = "\n".join([
                                "- Product name: Unknown",
                                "- Category: Unknown",
                                "- Type/form: Unknown",
                                "- Brand: Unknown",
                                "- Packaging (primary color, secondary color, container type, closure type): Unknown",
                                "- Size/volume: Unknown",
                                "- On-pack claims: Unknown",
                            ])
                            summaries[item_id] = fallback
                        else:
                            summaries[item_id] = sanitized

                    
                    except Exception as e:
                        print(f"Error generating summary for item {item_id}: {e}")
                        continue
                
            except Exception as e:
                print(f"Error generating summaries for batch: {e}")
                continue
    
    print(f"Generated {len(summaries)} VIU")
    return summaries


def maybe_generate_viu(
    dataset,
    data: Dict[str, Any],
    args,
) -> Optional[Dict[int, str]]:
    if not hasattr(args, 'generate_viu') or not args.generate_viu:
        return None
    
    if not args.use_image:
        print("Warning: generate_viu requires --use_image flag")
        return None

    if not QWEN3VL_AVAILABLE:
        print("Warning: Qwen3-VL not available. Skipping VIU generation.")
        print("Note: Qwen3-VL requires latest transformers. Install with:")
        print("pip install git+https://github.com/huggingface/transformers")
        return None
    
    meta = data.get("meta", {})
    if not meta:
        print("Warning: No metadata found. Skipping VIU generation.")
        return None
    
    num_items = max(meta.keys()) if meta else 0
    if num_items == 0:
        print("Warning: No items found. Skipping VIU generation.")
        return None
    
    # Check if summaries already exist
    preproc_folder = Path(dataset._get_preprocessed_folder_path())
    summaries_path = preproc_folder / "qwen3vl_viu.pt"
    
    if summaries_path.exists():
        print(f"Loading existing VIU from {summaries_path}")
        summaries = torch.load(summaries_path, map_location="cpu")
        return summaries
    
    # Generate VIU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get optimization settings from args
    batch_size = getattr(args, 'viu_batch_size', 4)
    max_new_tokens = getattr(args, 'viu_max_tokens', 160)
    # Priority: Use 4-bit quantization by default for all LLM models (Unsloth best practice)
    use_quantization = getattr(args, 'use_quantization', True)  # Default: True (4-bit enabled)
    use_torch_compile = getattr(args, 'use_torch_compile', False)
    preload_all_images = getattr(args, 'preload_all_images', False)
    
    # Load model with Unsloth and 4-bit quantization (default)
    model, processor = _load_qwen3vl_model(device, use_quantization=use_quantization)
    safe_batch_size = 1
    summaries = generate_viu(
        model, processor, device, meta, num_items,
        batch_size=safe_batch_size,
        use_torch_compile=use_torch_compile,
        max_new_tokens=max_new_tokens,
        preload_all_images=preload_all_images
    )
    
    # Save summaries to .pt file for caching (optional, CSV is the main storage)
    if summaries:
        summaries_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(summaries, summaries_path)
        print(f"Saved VIU cache to {summaries_path}")
        print(f"Note: VIU will also be saved to CSV in dataset_single_export.csv")
    
    return summaries



import re
from typing import Dict, List, Optional, Tuple

# ---- Canonical keys (must match your prompt order) ----
CANON_KEYS = [
    "Product name",
    "Category",
    "Type/form",
    "Brand",
    "Packaging (primary color, secondary color, container type, closure type)",
    "Size/volume",
    "On-pack claims",
]

# Accept some common variants seen in your logs
KEY_ALIASES = {
    "product name": "Product name",
    "name": "Product name",
    "category": "Category",
    "type": "Type/form",
    "type/form": "Type/form",
    "type form": "Type/form",
    "brand": "Brand",
    "packaging": "Packaging (primary color, secondary color, container type, closure type)",
    "packaging (primary color, secondary color, container type, closure type)": "Packaging (primary color, secondary color, container type, closure type)",
    "packaging ( primary color, secondary colour, container type , closure type)": "Packaging (primary color, secondary color, container type, closure type)",
    "packaging ( primary color, secondary colour, container type , closure type )": "Packaging (primary color, secondary color, container type, closure type)",
    "packaging ( primary color, secondary colour, container type, closure type)": "Packaging (primary color, secondary color, container type, closure type)",
    "size/volume": "Size/volume",
    "size/volumes": "Size/volume",
    "size/volumne": "Size/volume",
    "size/volume:": "Size/volume",
    "on-pack claims": "On-pack claims",
    "on pack claims": "On-pack claims",
    "on-pack claims (verbatim or near-verbatim text)": "On-pack claims",
    "on-pack claims:": "On-pack claims",
}

# Some phrases that indicate "explanations / inference" you want to forbid
BANNED_META_PATTERNS = [
    r"\binferred\b",
    r"\bnote:\b",
    r"\bthe above fields\b",
    r"\bformatted exactly\b",
    r"\baccording to\b",
    r"\bprovided instructions\b",
    r"\bthere are no repeated\b",
]

def _normalize_key(raw_key: str) -> Optional[str]:
    k = raw_key.strip().strip("-•*").strip().rstrip(":").strip()
    k_l = re.sub(r"\s+", " ", k).lower()
    # Direct alias
    if k_l in KEY_ALIASES:
        return KEY_ALIASES[k_l]
    # Heuristic: match prefixes
    if k_l.startswith("product name"):
        return "Product name"
    if k_l.startswith("category"):
        return "Category"
    if k_l.startswith("type"):
        return "Type/form"
    if k_l.startswith("brand"):
        return "Brand"
    if k_l.startswith("packaging"):
        return "Packaging (primary color, secondary color, container type, closure type)"
    if k_l.startswith("size"):
        return "Size/volume"
    if "claims" in k_l:
        return "On-pack claims"
    return None

def _dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        key = it.strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it.strip())
    return out

def _clean_value(v: str) -> str:
    v = v.strip()
    # remove surrounding quotes if present
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        v = v[1:-1].strip()
    # collapse whitespace
    v = re.sub(r"\s+", " ", v).strip()
    # normalize common "Unknown:" patterns
    if re.fullmatch(r"unknown[:\s]*", v, flags=re.I):
        return "Unknown"
    if v == "":
        return "Unknown"
    return v

def _contains_banned_meta(text: str) -> bool:
    t = text.lower()
    return any(re.search(pat, t) for pat in BANNED_META_PATTERNS)

def _too_repetitive(text: str, max_repeat: int = 2) -> bool:
    """
    Simple repetition check: if any non-trivial token (len>=3) repeats > max_repeat.
    """
    tokens = re.findall(r"[A-Za-z0-9\+\-]{3,}", text.lower())
    if not tokens:
        return False
    freq: Dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
        if freq[tok] > max_repeat:
            return True
    return False

def sanitize_viu(
    raw: str,
    *,
    max_claims: int = 8,
    strict: bool = True,
    keep_unknown: bool = True,
) -> Tuple[Optional[str], Dict[str, str]]:
    """
    Sanitize a VIU text into the canonical 7-field bullet format.

    Returns:
      (sanitized_text_or_None, debug_info)

    strict=True will return None if severe issues are found (meta/explanations, extreme repetition).
    If strict=False, it will attempt best-effort cleanup.

    keep_unknown=True keeps "Unknown" markers; set False only for display/UI.
    """
    debug = {"status": "ok", "reason": ""}

    if raw is None:
        debug["status"] = "fail"
        debug["reason"] = "raw_is_none"
        return None, debug

    text = raw.strip()

    # Quick block: if the model adds meta/explanations, treat as fail in strict mode.
    if _contains_banned_meta(text):
        if strict:
            debug["status"] = "fail"
            debug["reason"] = "banned_meta_detected"
            return None, debug

    # If overall output is extremely repetitive, fail in strict mode.
    if _too_repetitive(text, max_repeat=6):  # higher threshold for whole text
        if strict:
            debug["status"] = "fail"
            debug["reason"] = "global_repetition_detected"
            return None, debug

    # Parse lines
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() != ""]
    fields: Dict[str, str] = {k: "Unknown" for k in CANON_KEYS}
    claims: List[str] = []

    current_key: Optional[str] = None

    # Patterns for bullet field line like "- Key: value" or "Key: value"
    field_line_re = re.compile(r"^\s*[-•*]?\s*([^:]{2,80})\s*:\s*(.*)$")

    # Claim bullet like "  - CLAIM" or "- CLAIM" under claims section
    claim_re = re.compile(r"^\s*[-•*]\s+(.*)$")

    for ln in lines:
        # If we hit a new field line
        m = field_line_re.match(ln)
        if m:
            raw_key = m.group(1)
            val = m.group(2)
            norm_key = _normalize_key(raw_key)

            if norm_key is not None and norm_key in fields:
                current_key = norm_key
                # Special handling for On-pack claims: may have empty value then bullets follow
                if norm_key == "On-pack claims":
                    # if val contains claims inline (CSV/quoted), parse as claims
                    inline = val.strip()
                    if inline and inline.lower() != "unknown":
                        # split on commas but keep phrases
                        parts = [p.strip() for p in re.split(r"\s*,\s*", inline) if p.strip()]
                        claims.extend(parts)
                    continue
                else:
                    cleaned = _clean_value(val)
                    fields[norm_key] = cleaned
                    continue

        # If not a field line, maybe claims bullets under On-pack claims
        if current_key == "On-pack claims":
            m2 = claim_re.match(ln)
            if m2:
                c = _clean_value(m2.group(1))
                # avoid nested "Unknown:" noise
                if c.lower().startswith("unknown"):
                    continue
                # strip trailing punctuation
                c = c.strip().strip(",;")
                if c:
                    claims.append(c)
                continue

        # Otherwise ignore stray lines (like repeated "Unknown" lines)
        # (Optionally you could collect them for debugging)

    # Post-process fields (fix obviously bad generic product names)
    # If product name equals a generic type, set Unknown
    generic_names = {"shampoo", "conditioner", "leave in conditioner", "leave-in conditioner",
                     "cleanser", "wipes", "gel", "cream", "lotion", "soap", "body wash"}
    pn = fields["Product name"].strip().lower()
    if pn in generic_names:
        fields["Product name"] = "Unknown"

    # Dedup & cap claims
    cleaned_claims = []
    for c in claims:
        cc = _clean_value(c)
        if cc.lower() == "unknown":
            continue
        # Prevent super-long marketing chains
        if len(cc) > 80:
            cc = cc[:80].rstrip()
        cleaned_claims.append(cc)

    cleaned_claims = _dedup_preserve_order(cleaned_claims)

    # If claims themselves are repetitive, cut and/or mark Unknown
    # (this catches "SALON PROFESSIONAL" repeated)
    if cleaned_claims:
        joined = " ".join(cleaned_claims)
        if _too_repetitive(joined, max_repeat=2):
            # too repetitive claims -> keep unique only (already), then cap hard
            pass

    if len(cleaned_claims) > max_claims:
        cleaned_claims = cleaned_claims[:max_claims]

    # If still empty -> Unknown
    if not cleaned_claims:
        fields["On-pack claims"] = "Unknown"
    else:
        # We'll render as multiline bullets
        fields["On-pack claims"] = "\n" + "\n".join([f"  - {c}" for c in cleaned_claims])

    # Final strict checks: no huge loops / no extra fields
    final_text = []
    for k in CANON_KEYS:
        v = fields[k]
        if not keep_unknown and v.strip() == "Unknown":
            v = ""
        if k == "On-pack claims" and v != "Unknown" and v.startswith("\n"):
            final_text.append(f"- {k}:{v}")
        else:
            final_text.append(f"- {k}: {v}")

    out = "\n".join(final_text).strip()

    # If the output still contains obvious meta/explanation or pathological repetition, fail strict
    if strict:
        if _contains_banned_meta(out):
            debug["status"] = "fail"
            debug["reason"] = "banned_meta_after_sanitize"
            return None, debug
        # Detect the classic loop like "cap/pump: cap/pump"
        if re.search(r"(cap/pump:)\s*(\1\s*){2,}", raw, flags=re.I):
            debug["status"] = "fail"
            debug["reason"] = "loop_detected_cap_pump"
            return None, debug

    return out, debug
