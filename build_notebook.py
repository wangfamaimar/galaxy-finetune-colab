"""Generate the Colab fine-tuning notebook."""
from __future__ import annotations

import nbformat as nbf


_counter = iter(range(10_000))


def _next_id(prefix: str) -> str:
    return f"{prefix}-{next(_counter):03d}"


def md(text: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(text)
    cell["id"] = _next_id("md")
    return cell


def code(src: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(src)
    cell["id"] = _next_id("code")
    return cell


nb = nbf.v4.new_notebook()

nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}

cells: list[nbf.NotebookNode] = []

cells.append(
    md(
        """# Galaxy Image Classifier — Fine-Tune in Colab

Fine-tune [`dima806/galaxy_type_image_detection`](https://huggingface.co/dima806/galaxy_type_image_detection) on **your own galaxy images**.

The base model is a ViT (`google/vit-base-patch16-224-in21k` backbone) pre-trained by `dima806` to classify three galaxy morphologies: **E** (elliptical), **S** (spiral), **SB** (barred spiral).

This notebook lets you:

1. Mount your Google Drive.
2. Point at a folder of labelled galaxy images (one sub-folder per class).
3. Fine-tune the classifier — either keeping the original E/S/SB heads or defining your own classes.
4. Save the fine-tuned model back to Drive and run inference on new images.

> **Runtime tip:** `Runtime → Change runtime type → GPU` (T4 is fine). Training on CPU works but is ~20× slower.
"""
    )
)

cells.append(md("## 1. Install dependencies"))
cells.append(
    code(
        """# Colab already has torch, torchvision, PIL. We add the HuggingFace stack and a few helpers.
!pip -q install --upgrade \\
    "transformers>=4.44" \\
    "datasets>=2.20" \\
    "accelerate>=0.33" \\
    "evaluate>=0.4" \\
    scikit-learn"""
    )
)

cells.append(md("## 2. Mount Google Drive"))
cells.append(
    code(
        """from google.colab import drive
drive.mount('/content/drive')"""
    )
)

cells.append(
    md(
        """## 3. Configure paths

Organise your training images on Drive like this:

```
My Drive/
└── galaxy_finetune/
    ├── data/
    │   ├── E/
    │   │   ├── img001.jpg
    │   │   └── ...
    │   ├── S/
    │   └── SB/
    └── output/        # created automatically — fine-tuned model goes here
```

The sub-folder names become the class labels. You can use any class names you want — they don't have to match the original E/S/SB. If they differ, a **fresh classification head** is trained on top of the pre-trained ViT backbone (which still gives a big head start over training from scratch).

Edit the paths below to match your Drive layout.
"""
    )
)

cells.append(
    code(
        """import os

# --- EDIT THESE ---
DATA_DIR   = '/content/drive/MyDrive/galaxy_finetune/data'    # folder with one sub-folder per class
OUTPUT_DIR = '/content/drive/MyDrive/galaxy_finetune/output'  # where the fine-tuned model is saved
# ------------------

BASE_MODEL = 'dima806/galaxy_type_image_detection'

# Training hyper-parameters (tweak freely)
NUM_EPOCHS        = 5
BATCH_SIZE        = 16
LEARNING_RATE     = 5e-5
VAL_FRACTION      = 0.15     # fraction of images held out for validation
SEED              = 42
IMAGE_SIZE        = 224      # the ViT expects 224x224

os.makedirs(OUTPUT_DIR, exist_ok=True)
assert os.path.isdir(DATA_DIR), f"DATA_DIR does not exist: {DATA_DIR}"
print('Data directory:', DATA_DIR)
print('Output directory:', OUTPUT_DIR)"""
    )
)

cells.append(md("## 4. Inspect the dataset"))
cells.append(
    code(
        """from collections import Counter
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

def scan_imagefolder(root: str):
    root = Path(root)
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.')])
    samples = []
    for cls_dir in class_dirs:
        for p in cls_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), cls_dir.name))
    return samples, [d.name for d in class_dirs]

samples, classes = scan_imagefolder(DATA_DIR)
counts = Counter(lbl for _, lbl in samples)
print(f"Found {len(samples)} images across {len(classes)} classes:")
for c in classes:
    print(f"  {c:20s} {counts[c]}")
assert len(samples) > 0, "No images found. Check DATA_DIR and sub-folder layout."
assert len(classes) >= 2, "Need at least two class sub-folders to fine-tune."

label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in label2id.items()}
NUM_CLASSES = len(classes)"""
    )
)

cells.append(
    md(
        "## 5. Load the pre-trained model + preprocessor\n\n"
        "If your classes match the original E/S/SB, we reuse the existing classification head. "
        "Otherwise, a new head sized for your classes is initialised on top of the pre-trained backbone."
    )
)

cells.append(
    code(
        """import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

image_processor = AutoImageProcessor.from_pretrained(BASE_MODEL)

ORIGINAL_LABELS = ['E', 'S', 'SB']
reuse_head = sorted(classes) == sorted(ORIGINAL_LABELS)

if reuse_head:
    print('Classes match the original model — reusing its classification head.')
    model = AutoModelForImageClassification.from_pretrained(BASE_MODEL)
    # Remap in case the user's sub-folder order differs from the original id2label.
    model.config.label2id = label2id
    model.config.id2label = id2label
else:
    print(f'Training a fresh classification head for {NUM_CLASSES} classes: {classes}')
    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
model = model.to(device)"""
    )
)

cells.append(md("## 6. Build train / validation splits"))
cells.append(
    code(
        """import random
from collections import defaultdict

random.seed(SEED)

by_class: dict[str, list[str]] = defaultdict(list)
for path, lbl in samples:
    by_class[lbl].append(path)

train_items, val_items = [], []
for cls, paths in by_class.items():
    paths = sorted(paths)
    random.shuffle(paths)
    n_val = max(1, int(round(len(paths) * VAL_FRACTION))) if len(paths) > 1 else 0
    val_items.extend((p, cls) for p in paths[:n_val])
    train_items.extend((p, cls) for p in paths[n_val:])

print(f'Train: {len(train_items)}  Val: {len(val_items)}')"""
    )
)

cells.append(md("## 7. Torch Dataset with augmentation"))
cells.append(
    code(
        """from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

mean = image_processor.image_mean
std  = image_processor.image_std
size = image_processor.size.get('shortest_edge', IMAGE_SIZE) if isinstance(image_processor.size, dict) else IMAGE_SIZE
if isinstance(image_processor.size, dict) and 'height' in image_processor.size:
    size = image_processor.size['height']

train_tfm = T.Compose([
    T.Resize((size, size)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

eval_tfm = T.Compose([
    T.Resize((size, size)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

class GalaxyDataset(Dataset):
    def __init__(self, items, tfm):
        self.items = items
        self.tfm = tfm
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, cls = self.items[idx]
        img = Image.open(path).convert('RGB')
        return {'pixel_values': self.tfm(img), 'labels': label2id[cls]}

train_ds = GalaxyDataset(train_items, train_tfm)
val_ds   = GalaxyDataset(val_items,   eval_tfm)"""
    )
)

cells.append(md("## 8. Trainer configuration"))
cells.append(
    code(
        """import numpy as np
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

def collate(batch):
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'labels': torch.tensor([b['labels'] for b in batch], dtype=torch.long),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
    }

has_val = len(val_ds) > 0

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=20,
    eval_strategy='epoch' if has_val else 'no',
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=has_val,
    metric_for_best_model='f1_weighted' if has_val else None,
    greater_is_better=True,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
    report_to='none',
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds if has_val else None,
    data_collator=collate,
    compute_metrics=compute_metrics if has_val else None,
)"""
    )
)

cells.append(md("## 9. Fine-tune"))
cells.append(
    code(
        """train_result = trainer.train()
print(train_result.metrics)"""
    )
)

cells.append(md("## 10. Evaluate"))
cells.append(
    code(
        """if has_val:
    metrics = trainer.evaluate()
    print(metrics)
else:
    print('No validation split (VAL_FRACTION=0 or only 1 image per class) — skipping evaluation.')"""
    )
)

cells.append(
    md(
        "## 11. Save the fine-tuned model to Drive\n\n"
        "This writes model weights + preprocessor + label mapping to `OUTPUT_DIR` on your Drive. "
        "You can re-load it later with `AutoModelForImageClassification.from_pretrained(OUTPUT_DIR)`."
    )
)
cells.append(
    code(
        """final_dir = os.path.join(OUTPUT_DIR, 'final')
os.makedirs(final_dir, exist_ok=True)
trainer.save_model(final_dir)
image_processor.save_pretrained(final_dir)
print('Saved fine-tuned model to:', final_dir)"""
    )
)

cells.append(md("## 12. Inference on a single image"))
cells.append(
    code(
        """from PIL import Image
import torch

# Point this at any image you want to classify.
TEST_IMAGE = ''  # e.g. '/content/drive/MyDrive/galaxy_finetune/data/E/img001.jpg'

if TEST_IMAGE:
    model.eval()
    img = Image.open(TEST_IMAGE).convert('RGB')
    inputs = image_processor(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = logits.softmax(-1).cpu().tolist()
    ranked = sorted(zip(probs, [id2label[i] for i in range(NUM_CLASSES)]), reverse=True)
    print(f'Prediction for {TEST_IMAGE}:')
    for p, lbl in ranked:
        print(f'  {lbl:20s} {p:.3f}')
else:
    print('Set TEST_IMAGE to a path on your Drive to run inference.')"""
    )
)

cells.append(
    md(
        """## 13. (Optional) Push the fine-tuned model to the HuggingFace Hub

```python
from huggingface_hub import notebook_login
notebook_login()  # paste a write-access token

trainer.push_to_hub(
    repo_id='your-username/my-galaxy-classifier',
    commit_message='Fine-tuned from dima806/galaxy_type_image_detection',
)
```

---
That's it. Re-run section 3 onwards any time you add new images to your Drive folder.
"""
    )
)

nb["cells"] = cells

with open('notebooks/Galaxy_Finetune.ipynb', 'w') as f:
    nbf.write(nb, f)

print('wrote notebooks/Galaxy_Finetune.ipynb with', len(cells), 'cells')
