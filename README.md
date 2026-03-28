# 🎭 Emotion_Detect_AI

> **Detect human emotions from text using Machine Learning · NLP · Gradio**

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Emotions Detected](#-emotions-detected)
- [ML Models & Performance](#-ml-models--performance)
- [Model Comparison](#-model-comparison)
- [Project Workflow](#-project-workflow)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Example Sentences to Try](#-example-sentences-to-try)
- [Final Conclusions](#-final-conclusions)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 About the Project

**Emotion Detector** is an NLP-powered machine learning project that identifies the **human emotion** behind any typed sentence. Given a sentence, the model predicts whether the emotion is **Joy, Sadness, Anger, Fear, Love, or Surprise** — and also shows the **confidence probability** for each emotion.

The project trains and compares **four classic ML classifiers** — Naive Bayes, Logistic Regression, SVM, and Random Forest — using text vectorization techniques like Bag-of-Words and TF-IDF with character n-grams. The best-performing model (Linear SVM with **89.66% test accuracy**) is deployed through a beautiful **Gradio web interface** with emoji-based result cards and probability bar charts.

---

## 🖼️ Demo
### 🌐 Live Demo
[![Open App](https://img.shields.io/badge/🚀%20Launch%20App-Gradio-blue?style=for-the-badge)](https://32d98bb328b2f61d3b.gradio.live/)
### 😡 Anger Detection
![Anger](https://github.com/user-attachments/assets/5dc6e62e-16fc-4936-8af2-8ecd015b07bf)

*Input: "I am absolutely livid and frustrated with this situation right now" → **ANGER** detected with 36.1% confidence*

---

### 😄 Joy Detection
![Joy](https://github.com/user-attachments/assets/4e64ac86-2f70-49ed-bafb-043a23d67b1c)

*Input: "I am so excited and thrilled about this wonderful news today" → **JOY** detected with 85.5% confidence*

---

### ❤️ Love Detection
![Love](https://github.com/user-attachments/assets/1aeede55-e59a-4eb5-88fa-2012e01daa29)

*Input: "I feel so warm and affectionate whenever I am with you" → **LOVE** detected with 32.4% confidence*

---

### 😢 Sadness Detection
![Sadness](https://github.com/user-attachments/assets/5eee6f97-64a8-4ad5-b08e-edf2d890f173)

*Input: "I feel so lonely and empty inside nothing makes me happy anymore" → **SADNESS** detected with 71.1% confidence*

---

### 😲 Surprise Detection
![Surprise](https://github.com/user-attachments/assets/64a0998a-2e61-410d-a546-55e5bbcd03f7)

*Input: "I am so shocked and stunned I never saw this coming at all" → **SURPRISE** detected with 81.7% confidence*

---

### 😨 Fear Detection
![Fear](https://github.com/user-attachments/assets/d1fc1ff8-a17e-4763-9659-a81e0e501479)

*Input: "I am so scared and anxious about my results tomorrow" → **FEAR** detected with 79.4% confidence*

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.x |
| **ML Models** | Naive Bayes · Logistic Regression · LinearSVC · Random Forest |
| **Text Vectorization** | CountVectorizer (BoW) · TF-IDF (char n-grams) |
| **ML Library** | Scikit-learn |
| **Data Processing** | Pandas · NumPy |
| **Visualization** | Matplotlib · Seaborn |
| **UI / Frontend** | Gradio |
| **Model Serialization** | Pickle |
| **Runtime** | Jupyter Notebook / Google Colab |

---

## ✨ Features

- 🎯 **6-Class Emotion Classification** — Detects Joy, Sadness, Anger, Fear, Love, and Surprise
- 📊 **Probability Bar Chart** — Shows confidence scores for all 6 emotions simultaneously
- 🃏 **Emotion Result Card** — Color-coded card with emoji, emotion name, description, and confidence %
- ⚡ **Quick Example Buttons** — Click pre-loaded sentences to try the model instantly
- 🤖 **4 ML Models Trained** — Naive Bayes, Logistic Regression, SVM, Random Forest
- 📈 **Full EDA Pipeline** — 13 engineered text features including char count, word count, vowel count, punctuation count and more
- 🔍 **Model Comparison** — Side-by-side performance comparison of all 4 models
- 📉 **Confusion Matrices** — Visual confusion matrices for each model
- 💾 **Model Export** — Best model (SVM) and vectorizer saved as `.pkl` files for deployment
- 🌐 **Gradio Web Interface** — Clean, shareable browser-based UI

---

## 😶‍🌫️ Emotions Detected

| Emoji | Emotion | Description |
|---|---|---|
| 😄 | **Joy** | Feeling happy, elated or content |
| 😢 | **Sadness** | Feeling down, sorrowful or blue |
| 😡 | **Anger** | Feeling frustrated, irritated or furious |
| 😨 | **Fear** | Feeling scared, anxious or worried |
| ❤️ | **Love** | Feeling affectionate or romantic |
| 😲 | **Surprise** | Feeling shocked or astonished |

---

## 🤖 ML Models & Performance

### 1. 🔵 Multinomial Naive Bayes
- **Vectorizer:** Bag-of-Words (CountVectorizer)
- **Training Accuracy:** ~98.4%
- **Test Accuracy:** ~94.3%
- **Strengths:** Extremely fast to train, great baseline model
- **Limitation:** Assumes feature independence; word-level BoW misses character patterns

### 2. 🟢 Logistic Regression
- **Vectorizer:** TF-IDF with character n-grams (1–3), max 50,000 features
- **Training Accuracy:** ~98.4%
- **Test Accuracy:** ~98.2%
- **Strengths:** Best balance between training and test accuracy — minimal overfitting
- **Note:** Most interpretable model with excellent generalisation

### 3. 🏆 Linear SVM *(Best Model — Used in App)*
- **Vectorizer:** TF-IDF with character n-grams (1–3), max 50,000 features
- **Training Accuracy:** ~99.7%
- **Test Accuracy:** ~98.6% *(Highest)*
- **Strengths:** Best overall accuracy; finds optimal decision boundaries in high-dimensional space
- **Deployed in Gradio UI with 89.66% reported test accuracy**

### 4. 🟠 Random Forest
- **Vectorizer:** TF-IDF word-level (max 1,000 features)
- **Training Accuracy:** ~95.1%
- **Test Accuracy:** ~92.8%
- **Strengths:** Stable, no overfitting, consistent results
- **Limitation:** Lower accuracy due to limited word-level features

---

## 📊 Model Comparison

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Naive Bayes | ~98.4% | ~94.3% | High | High | High |
| Logistic Regression | ~98.4% | ~98.2% | Very High | Very High | Very High |
| **Linear SVM** ⭐ | **~99.7%** | **~98.6%** | **Highest** | **Highest** | **Highest** |
| Random Forest | ~95.1% | ~92.8% | Good | Good | Good |

> ⭐ **Linear SVM** is the winning model — saved as `svm_model.pkl` and deployed in the Gradio app.

---

## 🔄 Project Workflow
```
Raw Dataset (language.csv)
         │
         ▼
┌─────────────────────────┐
│  Exploratory Data       │
│  Analysis (EDA)         │
│  13 Text Features       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Text Preprocessing     │
│  Lowercase + Remove     │
│  Punctuation            │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Train / Test Split     │
│  80% Train / 20% Test   │
│  Stratified             │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│         Vectorization                │
│  BoW (Naive Bayes)                   │
│  TF-IDF char n-grams (LR, SVM)       │
│  TF-IDF word-level (Random Forest)   │
└──────────┬───────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│  Train 4 ML Models      │
│  NB · LR · SVM · RF     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Evaluate & Compare     │
│  Accuracy · Precision   │
│  Recall · F1 · CM       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Save Best Model (SVM)  │
│  svm_model.pkl          │
│  tfidf_vectorizer.pkl   │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Gradio Web Interface   │
│  Live Emotion Detection │
└─────────────────────────┘
```

---

## 📂 Dataset

- 📥 **Download Dataset:** [Google Drive Link](https://drive.google.com/file/d/111VL-UBFyGvNxN5s5RVOV1qLixn61ugg/view?usp=drivesdk)
- **File name:** `language.csv`
- **Columns:** `Text` (input sentence) · `language` (emotion label)
- **Labels:** joy · sadness · anger · fear · love · surprise
- **Split:** 80% training / 20% testing (stratified)

> Place `language.csv` in the **same folder** as the notebook before running.

---

## 📁 Project Structure
```
Emotion_Detect_AI/
│
├── Language_Recognition_NLP.ipynb   # Main Jupyter Notebook (training + EDA)
├── language.csv                      # Dataset (download from Drive link above)
├── svm_model.pkl                     # Saved best model (Linear SVM)
├── tfidf_vectorizer.pkl              # Saved TF-IDF vectorizer
├── Anger.png                         # Screenshot — Anger detection
├── Joy.png                           # Screenshot — Joy detection
├── Love.png                          # Screenshot — Love detection
├── Sadness.png                       # Screenshot — Sadness detection
├── Surprise.png                      # Screenshot — Surprise detection
├── Fear.png                          # Screenshot — Fear detection
|
|
|
└── README.md                         # This file
```

---

## ✅ Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- The dataset file `language.csv` (link above)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AbhishekDhawan07/Emotion_Detect_AI.git
cd Emotion_Detect_AI
```

### 2. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn gradio
```

Or in a Jupyter / Colab cell:
```python
!pip install numpy pandas matplotlib seaborn scikit-learn gradio
```

### 3. Download the Dataset

Download `language.csv` from the [Google Drive link](https://drive.google.com/file/d/111VL-UBFyGvNxN5s5RVOV1qLixn61ugg/view?usp=drivesdk) and place it in the project root folder.

---

## ▶️ Running the Application

Open `Language_Recognition_NLP.ipynb` and run the cells in order:

| Cell Range | Purpose |
|---|---|
| **Cell 0** | Import all libraries |
| **Cell 1–4** | Load dataset, check nulls, explore labels |
| **Cell 5–6** | EDA — Engineer 13 text features |
| **Cell 7–8** | Text preprocessing (lowercase, remove punctuation) |
| **Cell 9–10** | Train/test split (80/20 stratified) |
| **Cell 11** | Build BoW and TF-IDF vectorizers |
| **Cell 12–19** | Train & evaluate Multinomial Naive Bayes |
| **Cell 20–27** | Train & evaluate Logistic Regression |
| **Cell 28–35** | Train & evaluate Linear SVM |
| **Cell 36–43** | Train & evaluate Random Forest |
| **Cell 44–46** | Model comparison — all metrics side by side |
| **Cell 47** | Final conclusions |
| **Cell 48** | Save SVM model & vectorizer as `.pkl` files |

> The **Gradio** launches automatically after the model is trained — a public shareable URL is printed in the output.

---

## 📱 Usage Guide

1. **Open the Gradio URL** in any browser after launching the app.
2. **Type any sentence** in the text box — e.g. *"I am so happy today!"*
3. Click **"Detect Emotion"** or press Enter.
4. View the **emotion result card** on the right — shows emotion name, emoji, description, and confidence %.
5. Scroll down to see the **probability bar chart** for all 6 emotions.
6. Use the **Quick Example buttons** below the text box to try pre-loaded sentences instantly.
7. Click **"Clear"** to reset and try a new sentence.

---

## 💬 Example Sentences to Try

| Sentence | Expected Emotion |
|---|---|
| I feel so happy and grateful today! | 😄 Joy |
| I am so sad and nothing feels right anymore. | 😢 Sadness |
| I am furious about what happened! | 😡 Anger |
| I am terrified about the exam tomorrow. | 😨 Fear |
| I love spending time with you so much. | ❤️ Love |
| I never expected that to happen at all! | 😲 Surprise |

---

## 🏁 Final Conclusions

| # | Finding |
|---|---|
| 🥇 | **Best Overall Model — Linear SVM** with ~98.6% test accuracy using character-level TF-IDF n-grams |
| 🥈 | **Best Balanced Model — Logistic Regression** with the smallest gap between train (~98.4%) and test (~98.2%) accuracy |
| 🥉 | **Moderate Performer — Naive Bayes** achieved ~94.3% test accuracy; fast baseline but limited by BoW features |
| 4️⃣ | **Most Stable — Random Forest** showed no overfitting but lower accuracy (~92.8%) due to limited word-level features |

> **Character-level TF-IDF n-grams (1–3)** proved to be the most effective feature representation, capturing subtle patterns in word morphology that distinguish emotions better than word-level BoW.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">
  Built with ❤️ using Scikit-learn · Gradio · Pandas · Matplotlib
</div>
