# Multi-Model Stress Detection

A multimodal deep learning system designed to detect and classify human stress levels from video files. The system processes facial expressions (images) and vocal characteristics (audio), applies attention-based deep learning models to predict emotions, and fuses the predictions using a confidence-weighted decision fusion algorithm to output a final stress score (LOW, MEDIUM, or HIGH).

---

## 🌟 Key Features

- **Audio Emotion Detection**: Extracts and analyzes audio features using a deep learning architecture integrated with custom Attention Layers.
- **Visual Emotion Detection**: Processes facial frames from video using a convolutional neural network reinforced with a **CBAM (Convolutional Block Attention Module)** layer.
- **Multimodal Decision Fusion**:
  - **Certainty Calculation**: Evaluates model certainty using Shannon entropy.
  - **Confidence-Weighted Fusion**: Weighs the predictions of each model dynamically based on their individual certainty levels.
  - **Agreement Boosting**: Boosts the confidence score if both models agree on the detected emotion.
  - **Max Fusion**: Evaluates maximum likelihood across modalities to ensure robust detection under high disagreement.
- **Stress Score Mapping**: Translates fused emotion probabilities into quantitative stress metrics categorized into `LOW`, `MEDIUM`, or `HIGH` stress levels.

---

## 📁 Directory Structure

```text
Multi-Model-Stress-Detection/
│
├── backend/
│   ├── models/
│   │   ├── train/
│   │   │   ├── audio_model/           # Audio model structure and attention layers
│   │   │   └── image_model/           # Image model structure with CBAM attention
│   │   │
│   │   ├── preprocess/
│   │   │   ├── train/                 # Training data preprocessing and augmentation
│   │   │   └── inference/             # Inference frame preprocessing
│   │   │
│   │   ├── audio/                     # Audio prediction models & utilities
│   │   ├── image/                     # Image prediction models & utilities
│   │   └── evaluate_model.py          # Model evaluation helper
│   │
│   ├── service/
│   │   ├── audio_service.py           # Extracts audio components from videos
│   │   ├── image_service.py           # Extracts and crops face frames from videos
│   │   ├── fusion_service.py          # Confidence-based decision fusion logic
│   │   ├── stress_service.py          # Maps emotion classes to stress scores
│   │   └── emotion_pipeline.py        # Core processing pipeline from video to stress level
│   │
│   ├── utils/                         # Global helper scripts (audio, image, video utilities)
│   ├── config.py                      # Local configuration (Git ignored)
│   └── example_config.py              # Configuration template for path references
│
├── audio_store/                       # (Temp) Extracted audio files
├── image_store/                       # (Temp) Extracted facial frames
├── saved_model_1/                     # Trained models and encoder files
└── .gitignore                         # Git exclusion rules
```

---

## ⚙️ Configuration Setup

Since `backend/config.py` contains absolute paths specific to your local machine, it is ignored by Git. To set it up:

1. Copy `backend/example_config.py` to create `backend/config.py`:
   ```powershell
   cp backend/example_config.py backend/config.py
   ```
2. Open `backend/config.py` and configure the following variables:
   - **`image_train_dir` / `image_test_dir`**: Paths to your facial emotion training and testing datasets.
   - **`audio_dir`**: Path to your audio dataset.
   - **`video_store_path`**: Path to the target MP4 video you want to analyze for stress.
   - **`saved_model_1/`**: Directory where your `.keras` models and `.pkl` encoders will be stored/loaded.

---

## 🚀 Running the System

### 1. Training the Models
Before running inference, you need to train both the audio and image classifiers:
- Run the training script for the image model located under `backend/models/train/image_model/image_model.py`.
- Run the training script for the audio model located under `backend/models/train/audio_model/audio_model.py`.
- Both training scripts will output `.keras` files and `.pkl` encoders/classes into your `saved_model_1` directory.

### 2. Running the Stress Detection Pipeline
Once the models are trained and saved:
1. Ensure the `video_store_path` in `backend/config.py` points to the video you want to analyze.
2. Run the main pipeline script:
   ```powershell
   python -m backend.service.emotion_pipeline
   ```
3. The script will:
   - Extract audio and visual frames from the video.
   - Run predictions using the image and audio models.
   - Align the predictions over time windows.
   - Output the **final fused stress score** and a **Stress Level (`LOW`, `MEDIUM`, or `HIGH`)** to the console.
