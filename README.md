# IndicTTS Deepfake Detection Challenge
This project contains the complete code for a deep learning model designed to detect AI-generated (Text-to-Speech) audio. The model was developed for a challenge involving 16 different Indian languages and achieved near-perfect classification.

🚀 Key Result
The final model achieved a 0.99998 ROC-AUC on the challenge's test set, demonstrating an extremely high-performing and robust ability to distinguish between real human speech and fake, AI-generated speech.

📈 Methodology
The core of this solution is to treat audio classification as an image classification problem. Instead of feeding raw audio to a model, we convert audio files into their visual representations (spectrograms) and use a powerful, pre-trained Vision Transformer (ViT) to classify them.

Audio Preprocessing (Feature Extraction):

Each .wav file is loaded using librosa.

It is then converted into a Mel Spectrogram, which is a visual representation of the spectrum of frequencies in the audio.

The spectrogram is converted to the decibel scale (librosa.power_to_db).

Image-like Normalization:

The decibel-scaled spectrogram is normalized to a [0, 1] range, similar to pixels in an image.

Because the pre-trained model expects a 3-channel (RGB) image, the single-channel spectrogram is stacked 3 times (np.stack([mel_spec, mel_spec, mel_spec], axis=0)).

The Model (Vision Transformer):

The model used is a vit_base_patch16_224 (Vision Transformer) pre-trained on the ImageNet dataset, loaded via the timm library.

The spectrogram "images" are resized to 224x224 to match the ViT's expected input.

The model's final classification head is replaced with one that outputs 2 classes: (0: Real, 1: Fake/TTS).

Training & Inference:

The model is fine-tuned on the training set using PyTorch.

The notebook includes a full inference pipeline that processes the test.csv, generates predictions, and saves the final submission.csv file.
