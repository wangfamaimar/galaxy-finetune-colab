# Galaxy Finetune (Colab)

Fine-tune [`dima806/galaxy_type_image_detection`](https://huggingface.co/dima806/galaxy_type_image_detection) — a ViT classifier for galaxy morphology (E / S / SB) — on **your own galaxy images**, straight from Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wangfamaimar/galaxy-finetune-colab/blob/main/notebooks/Galaxy_Finetune.ipynb)

## What you get

- A single Colab notebook that walks end-to-end through:
  1. Mounting your Google Drive.
  2. Loading your labelled galaxy images (one sub-folder per class).
  3. Fine-tuning the pre-trained ViT (either reusing the original E/S/SB head, or training a fresh head for custom classes).
  4. Saving the fine-tuned model back to Drive.
  5. Running inference on new images.
  6. (Optional) Pushing to the HuggingFace Hub.
- No CLI, no Python environment to manage — everything runs in Colab.

## How to use

1. **Upload your images to Google Drive.** Organise them like this (sub-folder names become the class labels — they can be anything):

   ```
   My Drive/
   └── galaxy_finetune/
       ├── data/
       │   ├── E/
       │   │   ├── img001.jpg
       │   │   └── ...
       │   ├── S/
       │   └── SB/
       └── output/        # auto-created — fine-tuned model is saved here
   ```

2. **Click the "Open in Colab" badge above** (or open [`notebooks/Galaxy_Finetune.ipynb`](notebooks/Galaxy_Finetune.ipynb) directly).

3. In Colab, switch to a GPU runtime: `Runtime → Change runtime type → T4 GPU`.

4. Run the cells in order. When prompted, authorize the Drive mount. Edit the `DATA_DIR` / `OUTPUT_DIR` paths in section 3 to match your Drive layout.

5. Training ~5 epochs on a few hundred images typically takes a couple of minutes on a T4.

## Notes

- If your class names are `E`, `S`, `SB` (matching the original model), the notebook **reuses the existing classification head**, so you're literally just continuing to fine-tune the published model on your data.
- If you use different class names, a **fresh classification head** is initialised on top of the pre-trained ViT backbone, so you still benefit from the pre-trained features.
- The notebook does light image augmentation (horizontal flip, rotation ≤ 15°, mild colour jitter). Tweak in section 7 if your data isn't rotation-invariant.
- The base model was trained at 224×224 on the Galaxy DECaLS dataset; `dima806` reports ~78% weighted F1 on the three-class problem.

## Bundled sample dataset

For a zero-setup smoke test, the repo includes ~15 public-domain NASA / ESA / Hubble / ESO galaxy thumbnails under `sample_data/{E,S,SB}/` (5 per class, ~5 MB total). See [`sample_data/SOURCES.md`](sample_data/SOURCES.md) for the list and attributions.

In the notebook, set `USE_SAMPLE_DATA = True` in **section 3b** to clone the repo into Colab and fine-tune on this bundled set — no Drive mount required. The resulting model will only be a toy (few images, few epochs), but it proves the whole pipeline runs end-to-end in ~1 minute on a T4. Use your own Drive dataset for anything serious.

## Files

- `notebooks/Galaxy_Finetune.ipynb` — the Colab notebook (this is the deliverable).
- `build_notebook.py` — script that regenerates the notebook. Edit and re-run if you want to change default hyper-parameters.
- `sample_data/` — bundled public-domain galaxy thumbnails for quick smoke-testing.
- `scripts/download_samples.py` — re-downloads / refreshes the sample set from Wikipedia's `pageimages` API.

## License

MIT.
