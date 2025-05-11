# Multimodal-Behavioral-and-Stress-Evaluation
A multimodal approach for evaluating candidate interview performance using machine learning to analyze verbal and non-verbal cues.

## Project Overview

This research explores an objective and data-driven approach to interview evaluation by analyzing multimodal signals to assess both behavioral traits and stress indicators during interview scenarios. The project leverages both the MIT Interview Dataset and the MAS (Multimodal Affective Signals) Dataset to extract, analyze, and model key features across audio, video, and lexical modalities.

## Research Goals

- Provide a more objective assessment of interview performance compared to traditional subjective methods
- Quantify engagement, confidence, and stress levels during interviews using multimodal signals
- Develop predictive models for behavioral traits and stress indicators
- Compare the effectiveness of different machine learning approaches for these tasks

## Datasets

### MIT Interview Dataset
- Contains 138 recorded interview videos (10.5 hours total) of 69 internship-seeking students
- Includes Amazon Mechanical Turk ratings and ground truth scores
- Used for predicting behavioral analytics scores

### Multi-Affect-Stress (MAS) Dataset
- Collection of 353 video clips from YouTube interview channels
- Annotated with stress indicators (sentiment stress, vocal stress, fidgeting, facial stress)
- Used for predicting overall stress scores

## Methodology

### Feature Extraction
- **Audio Features**: Pitch statistics, energy measurements, speaking rate, pauses, jitter, shimmer
- **Video Features**: Head pose (pitch, yaw, roll), facial landmarks, eye/lip measurements
- **Lexical Features**: Word counts, speech rate, filler word analysis, emotional sentiment

### Early Fusion
- Combined features from all modalities into a unified representation
- Applied preprocessing techniques:
  - Z-score normalization
  - Feature selection using F-regression
  - Removal of highly correlated features

### Model Training and Evaluation
- **MIT Dataset Prediction Tasks**:
  - Eye Contact, Speaking Rate, Engagement, Pauses
  - Calmness, Not Stressed, Focused, Authentic, Not Awkward
  
- **MAS Dataset Prediction Tasks**:
  - Overall Stress Score

## Models and Results

### MIT Dataset Performance (MSE)
- **Random Forest**: 0.2925 avg MSE
- **Gradient Boosting**: 0.2549 avg MSE (best performer)
- **Ridge**: 0.5139 avg MSE

### MAS Dataset Performance (MSE)
- **LightGBM**: 0.4681 MSE
- **Neural Network**: 0.6698 MSE
- **Gradient Boosting (3-fold)**: 0.0161 MSE (best performer)

## Files and Directory Structure

- **Experiments/**: Model training and evaluation scripts
  - `gbr_three_fold_mas.py`: Gradient Boosting Regressor for MAS dataset
  - `gradient_mit.py`: Gradient Boosting for MIT dataset
  - `light_gbm_mas.py`: LightGBM model for MAS dataset
  - `models_training_MAS.ipynb`: Complete training notebook for MAS
  - `neural_network_mas.py`: Neural Network implementation for MAS
  - `randomforest.py`: Random Forest implementation
  - `ridge_mit.py`: Ridge Regression for MIT dataset

- **Feature Extraction/**: Scripts for extracting features from different modalities
  - `Audio_features`: Audio feature extraction from audio files
  - `audio-MIT.py`: Audio feature extraction for MIT dataset
  - `lexical_features.py`: Text-based feature extraction
  - `video_features.py`: Video and facial feature extraction

- **PreProcessing/**: Data preparation and feature engineering
  - `facial_aggregated.py`: Aggregation of facial features
  - `fuse_features.py`: Feature fusion across modalities
  - `normalization.py`: Feature normalization procedures
  - `prosodic_aggregated.py`: Aggregation of prosodic features
  - `stress_score.py`: Stress score computation for MAS dataset
  - `turker_labels.py`: Processing Amazon MTurk labels

## Technologies Used

- **Python**: Primary programming language
- **Machine Learning Libraries**: scikit-learn, XGBoost, LightGBM
- **Deep Learning**: PyTorch, Transformers
- **Audio Processing**: Librosa, Parselmouth
- **Video Processing**: OpenCV, Mediapipe
- **Text Processing**: NLTK, Google Cloud Speech API
- **Data Analysis**: Pandas, NumPy

## Requirements

- Python 3.7+
- Libraries listed in requirements.txt (to be installed via pip)
- Access to Google Cloud Speech API (for text extraction)
- GPU recommended for faster video processing

## Usage

1. Feature extraction:
   ```
   python Feature\ Extraction/video_features.py
   python Feature\ Extraction/audio-MIT.py
   python Feature\ Extraction/lexical_features.py
   ```

2. Feature preprocessing:
   ```
   python PreProcessing/facial_aggregated.py
   python PreProcessing/prosodic_aggregated.py
   python PreProcessing/fuse_features.py
   ```

3. Model training:
   ```
   python Experiments/gradient_mit.py
   python Experiments/gbr_three_fold_mas.py
   ```

## Future Work

- Integrate additional modalities (e.g., physiological signals)
- Explore deep learning architectures for end-to-end feature learning
- Develop a real-time analysis system for live interview feedback
- Investigate cross-cultural differences in interview behavior interpretation
