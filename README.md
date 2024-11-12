# Multimodal Sentiment Analysis for Human-Computer Interaction

### Overview
This project focuses on building a multimodal sentiment analysis model that leverages **audio, visual, and textual data** to improve emotion detection accuracy in **Human-Computer Interaction (HCI)**. By using the **CMU MOSEI dataset** and implementing various data fusion techniques, this model captures complex emotional cues, aiming to advance the development of emotionally intelligent systems for applications such as virtual meetings, online learning, and customer service.

---

## Table of Contents
- [Project Features](#project-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)

---

### Project Features
- **Multimodal Data Processing**: Supports text, audio, and video data from the CMU MOSEI dataset.
- **Fusion Techniques**: Implements early, late, and hybrid fusion techniques for combining multimodal data.
- **Performance Evaluation**: Evaluates model performance using accuracy, precision, recall, and F1 score.
- **Comparative Analysis**: Compares the results of multimodal models against unimodal baselines.

---

### Dataset
The model is trained and evaluated using the **[CMU Multimodal Opinion Sentiment and Emotion Intensity (CMU MOSEI)](https://github.com/A2Zadeh/CMU-MultimodalSDK)** dataset, a benchmark for multimodal sentiment analysis. This dataset includes:
- **Audio**: Features such as pitch and tone for vocal analysis.
- **Video**: Facial expressions and gesture data.
- **Text**: Transcriptions of spoken content.

*Note: Ensure proper permissions and access to download and use the CMU MOSEI dataset.*

---

### Methodology
1. **Data Preprocessing**: 
   - **Text**: Tokenization, embedding generation.
   - **Audio**: Feature extraction for pitch and tone.
   - **Video**: Facial and gesture feature extraction.
   - **Data Synchronization**: Aligns all modalities based on timestamps.

2. **Model Design and Fusion**:
   - **Architecture**: LSTM or Transformer-based architectures for each modality.
   - **Fusion Techniques**: Early fusion (input level), late fusion (output level), and hybrid fusion.

3. **Training and Evaluation**:
   - **Hyperparameter Tuning**: Optimizes learning rate, batch size, etc.
   - **Performance Metrics**: Assesses accuracy, precision, recall, and F1 score.

---

### Installation
To set up the environment, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/multimodal-sentiment-analysis-HCI.git
   cd multimodal-sentiment-analysis-HCI

2. **Install Dependencies: Install required Python packages using requirements.txt:**:
   ```bash
   pip install -r requirements.txt

3. **Download Dataset:**:
   Follow instructions on CMU Multimodal SDK to download and prepare the CMU MOSEI dataset.


---
### Results
Key results include:

Multimodal Fusion Performance: The model demonstrated higher accuracy and F1 scores when combining all three modalities compared to individual modality performance.
Fusion Technique Comparison: Late fusion yielded the best results for this dataset, with hybrid fusion close behind.


---
### Future Scope
Future work could explore:

- Real-Time Processing: Adapting the model for real-time sentiment analysis in virtual meetings and customer support.
- Emotion Recognition: Expanding from sentiment to broader emotion categories for diverse applications.
- Domain-Specific Models: Tailoring the model to specific domains like education or telemedicine.
