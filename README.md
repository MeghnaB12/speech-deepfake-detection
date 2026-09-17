# IndicTTS Deepfake Detection Challenge

This project contains the complete code for a deep learning model designed to detect AI-generated (Text-to-Speech) audio across 16 Indian languages.

----
## 🚀 Key Result

The final model achieved a **0.99998 ROC-AUC on the challenge test set**. This is a dataset-specific result and should not be interpreted as evidence of near-perfect performance on unseen domains, codecs, speakers, languages, or future TTS systems.

## 📈 Methodology

The solution treats audio classification as an image-classification problem. Audio is converted into Mel spectrograms and classified with a pre-trained Vision Transformer (ViT).

1. **Audio Preprocessing (Feature Extraction):**
   * Each `.wav` file is loaded using `librosa`.
   * It is converted into a **Mel Spectrogram**.
   * The spectrogram is converted to the decibel scale (`librosa.power_to_db`).

2. **Image-like Normalization:**
   * The decibel-scaled spectrogram is normalized to a [0, 1] range.
   * The single-channel spectrogram is stacked into 3 channels (`np.stack([mel_spec, mel_spec, mel_spec], axis=0)`) for the image backbone.

3. **The Model (Vision Transformer):**
   * Uses `vit_base_patch16_224`, pre-trained on ImageNet and loaded via `timm`.
   * Spectrogram images are resized to 224x224.
   * The classification head is replaced with a 2-class output: Real vs Fake/TTS.

4. **Training & Inference:**
   * The model is fine-tuned with PyTorch.
   * The notebook includes inference over the challenge test split and submission generation.

## 🛠️ Tech Stack

* **Core:** Python
* **Deep Learning:** PyTorch
* **Model Architecture:** `timm` Vision Transformer
* **Audio Processing:** `librosa`
* **Data Handling:** `pandas`, `NumPy`
* **Metrics:** `scikit-learn` ROC-AUC
* **Utilities:** `tqdm`, Jupyter

## 🏃 Running the Project

### 1. Dependencies

```bash
pip install torch torchvision timm librosa pandas numpy scikit-learn tqdm jupyter
```

### 2. Dataset

This model was trained on the Multilingual Indian Speech Data dataset used for the university challenge. Due to access restrictions, the dataset is not included in this repository.

As a result, the notebook cannot reproduce training or challenge-set inference without access to the original data.

### 3. Notebook Review

`indic_deepfake.ipynb` contains the end-to-end methodology, including:

* data preprocessing
* `FakeVoiceDataset`
* Vision Transformer model definition
* training and validation loop
* challenge inference pipeline

The reported ROC-AUC reflects the original challenge evaluation only; broader robustness would require additional external datasets and distribution-shift testing.
