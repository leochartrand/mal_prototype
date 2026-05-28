"""
Build complete memory-mapped dataset from pkl files.

Steps:
  1. Plan       — count samples, split train/test, create shuffle mapping
  2. Write      — stream images/labels to mmap at shuffled positions
  3. Labels     — create text model hidden states + attention masks
  4. Images     — create Theia image embeddings
  5. Decoder    — update metadata for decoder (reuses existing mmap files)
  6. Validate   — check consistency and print summary

Produces:
  mmap_data/
    - metadata.json, source_files.npy, file_names.pkl
    - train_indices.npy, test_indices.npy
    - initial_224.npy, target_224.npy (float32, N×3×224×224, [0,1])
    - labels.pkl
    - labels_hidden_{text_model}.npy, labels_attn_mask_{text_model}.npy
    - labels_pooled_{text_model}.npy  (float32, N×pooled_dim, for AdaLN)
    - initial_embed_{theia_model}.npy, target_embed_{theia_model}.npy

  Decoder training reuses the same mmap_data/ files directly.
  DecoderMemoryMappedDataset reads initial_* and target_* arrays,
  treating each trajectory as 2 samples (initial + target frame).
"""

from __future__ import annotations

import numpy as np
import pickle
import json
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    CLIPTextModel,
    AutoModel, AutoTokenizer,
)

# ============================================================================
# Configuration
# ============================================================================

# Project root: scripts/ lives one level below
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"

# PKL_DIRS and MMAP_DIR are set from CLI args (see main())
# PKL_DIRS: list of (path, strategy) where strategy is one of:
#   "all_test"    — all samples go to test (e.g. val_pkl)
#   "bridge"      — TEST_PREFIXES held out entirely, rest train
#   "random"      — random TEST_SPLIT_RATIO per sample (e.g. droid_pkl)
PKL_DIRS: list = None  # type: ignore
MMAP_DIR: Path = None  # type: ignore

THEIA_MODELS = {
    # "theia_tiny_cdiv": _MODELS_DIR / "theia_tiny_cdiv",
    # "theia_tiny_cddsv": _MODELS_DIR / "theia_tiny_cddsv",
    "theia_small_cdiv": _MODELS_DIR / "theia_small_cdiv",
    # "theia_small_cddsv": _MODELS_DIR / "theia_small_cddsv",
    # "theia_base_cdiv": _MODELS_DIR / "theia_base_cdiv",
    # "theia_base_cddsv": _MODELS_DIR / "theia_base_cddsv",
}

TEXT_MODELS = {
    "all-MiniLM-L6-v2": Path("/home/chal2525/mal_prototype/models/all-MiniLM-L6-v2"),
    # "siglip2-base-patch16-224": Path("/home/chal2525/mal_prototype/models/siglip2-base-patch16-224"),
    # "bge-small-en-v1.5": Path("/home/chal2525/mal_prototype/models/bge-small-en-v1.5"),
}

# Test split:
#   - Bridge: entire files matching TEST_PREFIXES
#   - DROID:  random 15% of samples per file (to maintain ~85/15 overall)
TEST_SPLIT_RATIO = 0.15

# Test split prefixes (Bridge environments held out entirely)
TEST_PREFIXES = [
    "berkeley_toysink",
    "datacol2_toysink",
    "upenn_toysink",
    "datacol2_folding_table",
    "deepthought_folding_table",
    "minsky_folding_table",
]

# Prefixes whose samples are randomly split (not held out by file)
RANDOM_SPLIT_PREFIXES = [
    "droid_",
]

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Step 1: Plan — count, split, shuffle, save mapping
# ============================================================================

def step1_plan():
    """
    Count samples per pkl file, determine train/test split, create shuffle
    mapping, and save the plan to disk.  No images are kept in RAM.
    
    Outputs (in MMAP_DIR):
      - file_names.pkl        list of pkl file names
      - shuffle_plan.pkl      dict with loc_to_out_pos mapping + counts
      - metadata.json         n_samples, n_train, n_test, etc.
      - train_indices.npy     [0, n_train)
      - test_indices.npy      [n_train, total)
    """
    print("\n" + "="*60)
    print("STEP 1: Plan (count, split, shuffle)")
    print("="*60)
    
    MMAP_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all pkl files across all source dirs with their strategy
    # file_records: list of (full_path, strategy)
    file_records = []
    for pkl_dir, strategy in PKL_DIRS:
        files = sorted(Path(pkl_dir).glob("*.pkl"))
        print(f"  {pkl_dir} [{strategy}]: {len(files)} pkl files")
        for f in files:
            file_records.append((f, strategy))
    print(f"Found {len(file_records)} pkl files total")

    # Store full paths so step2 can locate them without knowing PKL_DIRS
    file_paths = [str(r[0]) for r in file_records]

    def _split_strategy(file_path: Path, strategy: str) -> str:
        """Return 'test', 'train', or 'random' for this file."""
        if strategy == "all_test":
            return "test"
        elif strategy == "bridge":
            fname = file_path.name
            if any(fname.startswith(p) for p in TEST_PREFIXES):
                return "test"
            return "train"
        elif strategy == "random":
            return "random"
        raise ValueError(f"Unknown strategy: {strategy}")

    # Count samples per file
    print("\nCounting samples per file...")
    file_counts = []  # (file_idx, n_samples, split)
    for file_idx, (fpath, strategy) in enumerate(tqdm(file_records, desc="Counting")):
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        file_counts.append((file_idx, len(data), _split_strategy(fpath, strategy)))
        del data

    # Build flat index
    train_indices_list = []
    test_indices_list = []
    rng_split = np.random.default_rng(SEED)

    for file_idx, count, split in file_counts:
        entries = [(file_idx, j) for j in range(count)]
        if split == "test":
            test_indices_list.extend(entries)
        elif split == "train":
            train_indices_list.extend(entries)
        else:  # random
            mask = rng_split.random(count) < TEST_SPLIT_RATIO
            for j, is_test in enumerate(mask):
                if is_test:
                    test_indices_list.append((file_idx, j))
                else:
                    train_indices_list.append((file_idx, j))
    
    n_train = len(train_indices_list)
    n_test = len(test_indices_list)
    total = n_train + n_test
    
    print(f"\nTotal samples: {total:,}")
    print(f"Train: {n_train:,} ({100*n_train/total:.1f}%)")
    print(f"Test:  {n_test:,} ({100*n_test/total:.1f}%)")
    
    # Shuffle each split (separate rng from the split rng)
    rng_shuf = np.random.default_rng(SEED + 100)
    rng_shuf.shuffle(train_indices_list)
    rng_shuf.shuffle(test_indices_list)
    
    # Combined order: train first, then test
    ordered_indices = train_indices_list + test_indices_list
    
    # Build reverse lookup: (file_idx, sample_idx) → output position
    loc_to_out_pos = {}
    for out_pos, (file_idx, sample_idx) in enumerate(ordered_indices):
        loc_to_out_pos[(file_idx, sample_idx)] = out_pos
    
    # Save plan
    plan = {
        "loc_to_out_pos": loc_to_out_pos,
        "n_train": n_train,
        "n_test": n_test,
        "total": total,
    }
    with open(MMAP_DIR / "shuffle_plan.pkl", 'wb') as f:
        pickle.dump(plan, f)

    # Save full file paths (step2 uses these directly)
    with open(MMAP_DIR / "file_names.pkl", 'wb') as f:
        pickle.dump(file_paths, f)
    
    # Train indices are simply [0, n_train), test are [n_train, total)
    np.save(MMAP_DIR / "train_indices.npy", np.arange(n_train, dtype=np.int32))
    np.save(MMAP_DIR / "test_indices.npy", np.arange(n_train, total, dtype=np.int32))
    
    # Save metadata
    metadata = {
        "n_samples": total,
        "n_train": n_train,
        "n_test": n_test,
        "img_224_shape": [3, 224, 224],
        "seed": SEED,
        "order": "train first (shuffled), then test (shuffled)",
    }
    with open(MMAP_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPlan saved. Ready for step 2 (write).")


# ============================================================================
# Step 2: Write — stream images/labels to mmap using saved plan
# ============================================================================

def step2_write():
    """
    Stream pkl files and write resized images + labels to mmap at the
    shuffled positions determined by step 1.  Only one pkl in RAM at a time.
    
    Outputs (in MMAP_DIR):
      - initial_224.npy   (N, 3, 224, 224) float32 [0,1]
      - target_224.npy    (N, 3, 224, 224) float32 [0,1]
      - labels.pkl        list[str]
      - source_files.npy  int32
    """
    print("\n" + "="*60)
    print("STEP 2: Write images & labels to mmap")
    print("="*60)
    
    # Load plan from step 1
    with open(MMAP_DIR / "shuffle_plan.pkl", 'rb') as f:
        plan = pickle.load(f)
    loc_to_out_pos = plan["loc_to_out_pos"]
    total = plan["total"]
    
    with open(MMAP_DIR / "metadata.json") as f:
        metadata = json.load(f)
    
    with open(MMAP_DIR / "file_names.pkl", 'rb') as f:
        pkl_files = [Path(p) for p in pickle.load(f)]
    print(f"Found {len(pkl_files)} pkl files, {total:,} total samples")
    print(f"Image shape: (3, 224, 224) float32 [0,1]")
    
    # Create memory-mapped arrays
    initial_224 = np.lib.format.open_memmap(
        MMAP_DIR / "initial_224.npy", mode='w+',
        dtype=np.float32, shape=(total, 3, 224, 224)
    )
    target_224 = np.lib.format.open_memmap(
        MMAP_DIR / "target_224.npy", mode='w+',
        dtype=np.float32, shape=(total, 3, 224, 224)
    )
    
    labels = [None] * total
    source_files = np.zeros(total, dtype=np.int32)
    
    def extract_pair(sample):
        """Return (initial_hwc_uint8, target_hwc_uint8, label_str) from any sample format."""
        if isinstance(sample, dict):
            # droid / bridge / bridge_msr: images list of HWC uint8, first→last
            imgs = sample['images']
            initial = np.array(imgs[0], dtype=np.uint8)
            target  = np.array(imgs[-1], dtype=np.uint8)
            label   = sample['language']
        elif isinstance(sample, tuple):
            # val: (CHW float32, CHW float32, ?, label_str)
            def chw_float_to_hwc_uint8(arr):
                a = np.array(arr)
                if isinstance(arr, torch.Tensor):
                    a = arr.cpu().numpy()
                return (a.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            initial = chw_float_to_hwc_uint8(sample[0])
            target  = chw_float_to_hwc_uint8(sample[1])
            label   = str(sample[3])
        else:
            raise ValueError(f"Unknown sample type: {type(sample)}")
        return initial, target, label

    for file_idx, pkl_file in enumerate(tqdm(pkl_files, desc="Writing")):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        for sample_idx, sample in enumerate(data):
            out_pos = loc_to_out_pos[(file_idx, sample_idx)]
            initial, target, label = extract_pair(sample)

            # Resize to 224×224, HWC→CHW, normalize to [0,1] float32
            initial_224[out_pos] = cv2.resize(initial, (224, 224), interpolation=cv2.INTER_LANCZOS4).transpose(2, 0, 1).astype(np.float32) / 255.0
            target_224[out_pos]  = cv2.resize(target,  (224, 224), interpolation=cv2.INTER_LANCZOS4).transpose(2, 0, 1).astype(np.float32) / 255.0
            labels[out_pos] = label
            source_files[out_pos] = file_idx

        del data
    
    del initial_224, target_224
    
    # Save labels (in shuffled order)
    with open(MMAP_DIR / "labels.pkl", 'wb') as f:
        pickle.dump(labels, f)
    
    # Save source file mapping (in shuffled order)
    np.save(MMAP_DIR / "source_files.npy", source_files)
    
    print(f"\nWrite complete: {total:,} samples to mmap")


# ============================================================================
# Step 3: Create text label embeddings
# ============================================================================

def step3_embed_labels():
    """Create text model hidden states and attention masks for labels."""
    print("\n" + "="*60)
    print("STEP 3: Embed labels (hidden states + attention masks)")
    print("="*60)
    
    with open(MMAP_DIR / "labels.pkl", 'rb') as f:
        labels = pickle.load(f)
    n_samples = len(labels)
    
    for model_name, model_path in TEXT_MODELS.items():
        print(f"\n--- {model_name} ---")
        
        # ── CLIP / TinyCLIP ──
        if "clip" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            text_model = CLIPTextModel.from_pretrained(model_path).to(DEVICE).eval()
            max_len = 25        # dataset max is 24 tokens
            hidden_dim = text_model.config.hidden_size  # 768 (CLIP) or 512 (TinyCLIP-29M)
            
            def encode(batch_labels):
                inputs = tokenizer(
                    batch_labels, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True
                )
                attn = inputs["attention_mask"].numpy()
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    out = text_model(**inputs).last_hidden_state.cpu().numpy()
                return out, attn
        
        # ── SigLIP ──
        elif "siglip" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            full_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(DEVICE).eval()
            text_encoder = full_model.text_model
            max_len = 25        # SigLIP position embeddings support 64
            hidden_dim = full_model.config.text_config.hidden_size  # 768
            
            def encode(batch_labels):
                inputs = tokenizer(
                    batch_labels, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True
                )
                # SigLIP tokenizer doesn't return attention_mask; build from pad_token_id
                input_ids = inputs["input_ids"]
                attn = (input_ids != tokenizer.pad_token_id).int().numpy()
                inputs["attention_mask"] = torch.from_numpy(attn)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    out = text_encoder(**inputs).last_hidden_state.cpu().numpy()
                return out, attn
        
        # ── MiniLM / sentence-transformers ──
        elif "minilm" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            text_model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()
            max_len = 25        # default; MiniLM supports up to 256/512
            hidden_dim = text_model.config.hidden_size  # 384
            
            def encode(batch_labels):
                inputs = tokenizer(
                    batch_labels, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True
                )
                attn = inputs["attention_mask"].numpy()
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    out = text_model(**inputs).last_hidden_state.cpu().numpy()
                return out, attn
        
        # ── BGE (BAAI/bge-small-en-v1.5) ──
        elif "bge" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            text_model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()
            max_len = 25
            hidden_dim = text_model.config.hidden_size  # 384
            
            def encode(batch_labels):
                inputs = tokenizer(
                    batch_labels, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True
                )
                attn = inputs["attention_mask"].numpy()
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    out = text_model(**inputs).last_hidden_state.cpu().numpy()
                return out, attn
        
        else:
            raise ValueError(f"Unknown text model type: {model_name}")
        
        print(f"Max token length: {max_len}, hidden_dim: {hidden_dim}")
        
        labels_hidden = np.lib.format.open_memmap(
            MMAP_DIR / f"labels_hidden_{model_name}.npy", mode='w+',
            dtype=np.float32, shape=(n_samples, max_len, hidden_dim)
        )
        labels_attn_mask = np.lib.format.open_memmap(
            MMAP_DIR / f"labels_attn_mask_{model_name}.npy", mode='w+',
            dtype=np.int32, shape=(n_samples, max_len)
        )
        
        batch_size = 64
        for start in tqdm(range(0, n_samples, batch_size), desc=f"Embedding labels ({model_name})"):
            end = min(start + batch_size, n_samples)
            hidden, attn = encode(labels[start:end])
            labels_hidden[start:end] = hidden
            labels_attn_mask[start:end] = attn
        
        del labels_hidden, labels_attn_mask
        # Clean up all model references
        for v in list(locals().values()):
            if isinstance(v, torch.nn.Module):
                del v
        torch.cuda.empty_cache()
        
        # Update metadata
        with open(MMAP_DIR / "metadata.json") as f:
            metadata = json.load(f)
        metadata[f"label_hidden_shape_{model_name}"] = [max_len, hidden_dim]
        metadata[f"label_attn_mask_shape_{model_name}"] = [max_len]
        metadata[f"label_embed_model_{model_name}"] = str(model_path)
        with open(MMAP_DIR / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"{model_name} label embeddings complete")


# ============================================================================
# Step 3b: Create pooled text embeddings (for AdaLN conditioning)
# ============================================================================

def step3b_pooled_labels():
    """Create pooled (single-vector) text embeddings for AdaLN conditioning.
    
    For CLIP/TinyCLIP: uses pooler_output (EOS token hidden state after LN).
    For SigLIP/MiniLM: mean-pooling over last_hidden_state with attention mask.
    For BGE: CLS token pooling.
    
    Outputs (in MMAP_DIR):
      - labels_pooled_{text_model}.npy  (N, pooled_dim) float32
    """
    print("\n" + "="*60)
    print("STEP 3b: Pooled label embeddings (for AdaLN)")
    print("="*60)
    
    with open(MMAP_DIR / "labels.pkl", 'rb') as f:
        labels = pickle.load(f)
    n_samples = len(labels)
    
    for model_name, model_path in TEXT_MODELS.items():
        print(f"\n--- {model_name} (pooled) ---")
        
        # ── CLIP / TinyCLIP ──
        if "clip" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            text_model = CLIPTextModel.from_pretrained(model_path).to(DEVICE).eval()
            max_len = 25
            pooled_dim = text_model.config.hidden_size
            
            def encode_pooled(batch_labels):
                inputs = tokenizer(
                    batch_labels, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    out = text_model(**inputs)
                return out.pooler_output.cpu().numpy()  # [B, pooled_dim]
        
        # ── SigLIP (mean pooling) ──
        elif "siglip" in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            full_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(DEVICE).eval()
            text_encoder = full_model.text_model
            max_len = 25
            pooled_dim = full_model.config.text_config.hidden_size
            
            def encode_pooled(batch_labels):
                inputs = tokenizer(
                    batch_labels, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True
                )
                input_ids = inputs["input_ids"]
                attn = (input_ids != tokenizer.pad_token_id).int()
                inputs["attention_mask"] = attn
                inputs_dev = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    hidden = text_encoder(**inputs_dev).last_hidden_state  # [B, seq, D]
                attn_expanded = attn.unsqueeze(-1).float().to(DEVICE)  # [B, seq, 1]
                pooled = (hidden * attn_expanded).sum(dim=1) / attn_expanded.sum(dim=1).clamp(min=1e-9)
                return pooled.cpu().numpy()  # [B, pooled_dim]
        
        # ── MiniLM / sentence-transformers (mean pooling) ──
        elif "minilm" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            text_model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()
            max_len = 25
            pooled_dim = text_model.config.hidden_size
            
            def encode_pooled(batch_labels):
                inputs = tokenizer(
                    batch_labels, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True
                )
                attn_mask = inputs["attention_mask"]  # [B, seq_len]
                inputs_dev = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    hidden = text_model(**inputs_dev).last_hidden_state  # [B, seq, D]
                attn_expanded = attn_mask.unsqueeze(-1).float().to(DEVICE)  # [B, seq, 1]
                pooled = (hidden * attn_expanded).sum(dim=1) / attn_expanded.sum(dim=1).clamp(min=1e-9)
                return pooled.cpu().numpy()  # [B, pooled_dim]
        
        # ── BGE (CLS token pooling) ──
        elif "bge" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            text_model = AutoModel.from_pretrained(model_path).to(DEVICE).eval()
            max_len = 25
            pooled_dim = text_model.config.hidden_size  # 384
            
            def encode_pooled(batch_labels):
                inputs = tokenizer(
                    batch_labels, return_tensors="pt",
                    padding="max_length", max_length=max_len, truncation=True
                )
                inputs_dev = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    hidden = text_model(**inputs_dev).last_hidden_state  # [B, seq, D]
                pooled = hidden[:, 0]  # CLS token
                return pooled.cpu().numpy()  # [B, pooled_dim]
        
        else:
            raise ValueError(f"Unknown text model type: {model_name}")
        
        print(f"Pooled dim: {pooled_dim}")
        
        labels_pooled = np.lib.format.open_memmap(
            MMAP_DIR / f"labels_pooled_{model_name}.npy", mode='w+',
            dtype=np.float32, shape=(n_samples, pooled_dim)
        )
        
        batch_size = 64
        for start in tqdm(range(0, n_samples, batch_size), desc=f"Pooled ({model_name})"):
            end = min(start + batch_size, n_samples)
            labels_pooled[start:end] = encode_pooled(labels[start:end])
        
        del labels_pooled
        # Clean up all model references
        for v in list(locals().values()):
            if isinstance(v, torch.nn.Module):
                del v
        torch.cuda.empty_cache()
        
        # Update metadata
        with open(MMAP_DIR / "metadata.json") as f:
            metadata = json.load(f)
        metadata[f"label_pooled_dim_{model_name}"] = pooled_dim
        metadata[f"label_pooled_model_{model_name}"] = str(model_path)
        with open(MMAP_DIR / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"{model_name} pooled embeddings complete")


# ============================================================================
# Step 4: Create Theia image embeddings
# ============================================================================

def step4_embed_images(model_name: str):
    """Create Theia embeddings for images."""
    print("\n" + "="*60)
    print(f"STEP 4: Embed images with {model_name}")
    print("="*60)
    
    model_path = THEIA_MODELS[model_name]
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    
    with open(MMAP_DIR / "metadata.json") as f:
        metadata = json.load(f)
    n_samples = metadata["n_samples"]
    
    initial_224 = np.load(MMAP_DIR / "initial_224.npy", mmap_mode='r')
    target_224 = np.load(MMAP_DIR / "target_224.npy", mmap_mode='r')
    
    def chw_float_to_hwc_uint8(arr):
        """Convert (3, 224, 224) float32 [0,1] → (224, 224, 3) uint8."""
        return (arr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    
    # Detect embedding shape
    test_img = torch.from_numpy(chw_float_to_hwc_uint8(initial_224[0])).to(DEVICE)
    with torch.no_grad():
        test_out = model.forward_feature(test_img, do_resize=True, do_rescale=True, do_normalize=True)
    embed_shape = tuple(test_out.squeeze(0).shape)
    print(f"Embedding shape: {embed_shape}")
    
    initial_embed = np.lib.format.open_memmap(
        MMAP_DIR / f"initial_embed_{model_name}.npy", mode='w+',
        dtype=np.float32, shape=(n_samples, *embed_shape)
    )
    target_embed = np.lib.format.open_memmap(
        MMAP_DIR / f"target_embed_{model_name}.npy", mode='w+',
        dtype=np.float32, shape=(n_samples, *embed_shape)
    )
    
    for i in tqdm(range(n_samples), desc=f"Embedding ({model_name})"):
        init_img = torch.from_numpy(chw_float_to_hwc_uint8(initial_224[i])).to(DEVICE)
        targ_img = torch.from_numpy(chw_float_to_hwc_uint8(target_224[i])).to(DEVICE)
        
        with torch.no_grad():
            init_emb = model.forward_feature(init_img, do_resize=True, do_rescale=True, do_normalize=True)
            targ_emb = model.forward_feature(targ_img, do_resize=True, do_rescale=True, do_normalize=True)
        
        initial_embed[i] = init_emb.squeeze(0).cpu().numpy()
        target_embed[i] = targ_emb.squeeze(0).cpu().numpy()
    
    del initial_embed, target_embed
    
    # Update metadata
    with open(MMAP_DIR / "metadata.json") as f:
        metadata = json.load(f)
    metadata[f"embed_shape_{model_name}"] = list(embed_shape)
    metadata[f"embed_model_{model_name}"] = str(model_path)
    with open(MMAP_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"{model_name} embeddings complete")


# ============================================================================
# Step 5: Update metadata for decoder (no data copying)
# ============================================================================

def step5_create_decoder():
    """
    Update metadata with decoder info.  No data is copied — the decoder
    dataset class reads directly from the existing initial_*/target_* mmap
    files, treating each trajectory as 2 samples (initial + target frame).
    """
    print("\n" + "="*60)
    print("STEP 5: Update decoder metadata (no data copy)")
    print("="*60)
    
    with open(MMAP_DIR / "metadata.json") as f:
        metadata = json.load(f)
    n_train = metadata["n_train"]
    n_test = metadata["n_test"]
    n_samples = metadata["n_samples"]
    
    n_dec_train = n_train * 2
    n_dec_test = n_test * 2
    n_dec_total = n_dec_train + n_dec_test
    
    print(f"Trajectories: {n_samples:,} (train: {n_train:,}, test: {n_test:,})")
    print(f"Decoder samples: {n_dec_total:,} (train: {n_dec_train:,}, test: {n_dec_test:,})")
    print(f"No data copied — decoder reads from existing initial_*/target_* files")
    
    # Verify required files exist
    for model_name in THEIA_MODELS.keys():
        init_path = MMAP_DIR / f"initial_embed_{model_name}.npy"
        targ_path = MMAP_DIR / f"target_embed_{model_name}.npy"
        if init_path.exists() and targ_path.exists():
            print(f"  {model_name}: OK")
        else:
            print(f"  {model_name}: MISSING (run step 4 first)")
    
    # Update metadata with decoder info
    metadata["decoder_n_samples"] = n_dec_total
    metadata["decoder_n_train"] = n_dec_train
    metadata["decoder_n_test"] = n_dec_test
    metadata["decoder_note"] = "Decoder reads initial_*/target_* directly; each trajectory = 2 samples"
    with open(MMAP_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Decoder metadata updated (no extra disk usage)")


# ============================================================================
# Step 6: Validate and print summary
# ============================================================================

def step6_validate():
    """Validate data consistency and print summary."""
    print("\n" + "="*60)
    print("STEP 6: Validation & Summary")
    print("="*60)
    
    # Check files exist and print sizes
    print("\nmmap_data/:")
    total_size = 0
    for f in sorted(MMAP_DIR.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            print(f"  {f.name}: {size / 1e9:.2f} GB")
    print(f"  Total: {total_size / 1e9:.2f} GB")
    
    # Load metadata
    with open(MMAP_DIR / "metadata.json") as f:
        metadata = json.load(f)
    
    print("\n--- Structural Validation ---")
    
    # Check main data indices
    train_indices = np.load(MMAP_DIR / "train_indices.npy")
    test_indices = np.load(MMAP_DIR / "test_indices.npy")
    
    assert len(train_indices) == metadata["n_train"], "Train count mismatch"
    assert len(test_indices) == metadata["n_test"], "Test count mismatch"
    assert train_indices[0] == 0, "Train should start at 0"
    assert test_indices[0] == metadata["n_train"], "Test should start after train"
    print("  Main indices: OK")
    
    # Check decoder info in metadata
    n_train = metadata["n_train"]
    n_test = metadata["n_test"]
    n_samples = metadata["n_samples"]
    
    if "decoder_n_samples" in metadata:
        assert metadata["decoder_n_samples"] == 2 * n_samples, "Decoder total mismatch"
        assert metadata["decoder_n_train"] == 2 * n_train, "Decoder train mismatch"
        assert metadata["decoder_n_test"] == 2 * n_test, "Decoder test mismatch"
        print("  Decoder metadata: OK")
    else:
        print("  Decoder metadata: not found (run step 5)")
    
    print("\n--- Data Consistency Validation ---")
    
    initial_224 = np.load(MMAP_DIR / "initial_224.npy", mmap_mode='r')
    target_224 = np.load(MMAP_DIR / "target_224.npy", mmap_mode='r')
    
    assert initial_224.shape[0] == n_samples, f"initial_224 has {initial_224.shape[0]} samples, expected {n_samples}"
    assert target_224.shape[0] == n_samples, f"target_224 has {target_224.shape[0]} samples, expected {n_samples}"
    print(f"  initial_224: {initial_224.shape} OK")
    print(f"  target_224:  {target_224.shape} OK")
    
    # Check pooled label embeddings
    for model_name in TEXT_MODELS.keys():
        pooled_path = MMAP_DIR / f"labels_pooled_{model_name}.npy"
        if pooled_path.exists():
            lp = np.load(pooled_path, mmap_mode='r')
            assert lp.shape[0] == n_samples, f"{model_name} pooled count mismatch"
            print(f"  labels_pooled_{model_name}: {lp.shape} OK")
        else:
            print(f"  labels_pooled_{model_name}: MISSING")
    
    # Check embeddings
    for model_name in THEIA_MODELS.keys():
        init_path = MMAP_DIR / f"initial_embed_{model_name}.npy"
        targ_path = MMAP_DIR / f"target_embed_{model_name}.npy"
        if init_path.exists() and targ_path.exists():
            ie = np.load(init_path, mmap_mode='r')
            te = np.load(targ_path, mmap_mode='r')
            assert ie.shape[0] == n_samples, f"{model_name} initial embed count mismatch"
            assert te.shape[0] == n_samples, f"{model_name} target embed count mismatch"
            print(f"  {model_name}: initial {ie.shape}, target {te.shape} OK")
        else:
            print(f"  {model_name}: MISSING")
    
    # Spot-check: initial and target should differ
    sample_indices = [0, 100, 1000, n_samples // 2, n_samples - 1]
    print(f"\n  Spot-checking {len(sample_indices)} indices...")
    for idx in sample_indices:
        diff = np.abs(initial_224[idx].astype(np.float64) - target_224[idx].astype(np.float64)).mean()
        print(f"    idx {idx}: mean |init-target| = {diff:.4f} {'(identical!)' if diff < 1e-6 else ''}")
    
    print(f"\n--- Summary ---")
    print(f"  Trajectories: {n_samples:,}")
    print(f"  Train: {n_train:,} ({100*n_train/n_samples:.1f}%)")
    print(f"  Test: {n_test:,} ({100*n_test/n_samples:.1f}%)")
    print(f"  Decoder samples: {2*n_samples:,} (2x trajectories, no extra files)")
    print(f"  Data layout: pre-shuffled, train first, then test")
    print(f"  Sequential access: just iterate 0..N-1")
    
    print("\n" + "="*60)
    print("BUILD COMPLETE")
    print("="*60)


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0, help="Run specific step (0=all)")
    parser.add_argument("--model", type=str, default=None, help="Specific model for step 4")
    args = parser.parse_args()

    global PKL_DIRS, MMAP_DIR
    MMAP_DIR = Path("/mnt/sda1/Datasets/chal2525/mmap_data")
    PKL_DIRS = [
        (Path("/mnt/sda1/Datasets/chal2525/val_pkl"),        "all_test"),
        (Path("/mnt/sda1/Datasets/chal2525/bridge_pkl"),     "bridge"),
        (Path("/mnt/sda1/Datasets/chal2525/bridge_msr_pkl"), "bridge"),
        (Path("/mnt/sda1/Datasets/chal2525/droid_pkl"),      "random"),
    ]
    
    if args.step == 0 or args.step == 1:
        step1_plan()
    
    if args.step == 0 or args.step == 2:
        step2_write()
    
    if args.step == 0 or args.step == 3:
        step3_embed_labels()
        step3b_pooled_labels()
    
    if args.step == 0 or args.step == 4:
        models = [args.model] if args.model else list(THEIA_MODELS.keys())
        for model_name in models:
            step4_embed_images(model_name)
    
    if args.step == 0 or args.step == 5:
        step5_create_decoder()
    
    if args.step == 0 or args.step == 6:
        step6_validate()


if __name__ == "__main__":
    main()
