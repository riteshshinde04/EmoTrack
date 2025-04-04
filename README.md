# EmoTrack 🎭 — Real-Time Emotional Monitoring via Episodic Activity or Text

**EmoTrack** is a real-time emotion tracking application built using **Streamlit**. It allows users to input episodic activities or text and receive a breakdown of emotional levels such as anger, joy, sadness, fear, and more. This project focuses on user interaction, emotion stability, and tracking emotional trends over time.

---

## 🌟 Features

- 🧠 **Activity-Based Emotion Prediction**: Detect emotions based on user-defined activities.
- 📊 **Real-Time Analytics**: Visual feedback on emotion levels using Streamlit charts.
- 🗃️ **Logging & Trend Visualization**: Stores and displays user inputs for trend analysis.
- 🧘 **Emotion Stabilization**: Reduces fluctuation in emotion levels with smooth transitions.
- ⚡ **Lightweight & Fast**: Built using pure Python and Streamlit — no frontend coding required.

---

## 📁 Project Structure
EmoTrack/ │ ├── .streamlit/ # Streamlit configuration files │ ├── secrets.toml # Secret keys & authentication config │ ├── assets/ # Stores static assets (images, icons, etc.) │ ├── data/ # Dataset-related files │ ├── emotion_dataset_raw.csv # Raw dataset used for emotion training │ ├── model/ # Pre-trained ML models for emotion detection │ ├── bert_lstm.pkl # BERT-LSTM model for text-based emotion detection │ ├── text_emotion.pkl # General text emotion classifier │ ├── text_emotion_bert.pkl # BERT-based text emotion classifier │ ├── text_emotion_glove.pkl # GloVe-based emotion model │ ├── Emotion Detection.ipynb # Jupyter notebook for model training │ ├── static/ # Stores static files (if needed in future) │ ├── templates/ # Template files for UI components │ ├── account.py # Handles user authentication and account management ├── home.py # Manages the home page & dashboard ├── main.py # Entry point for Streamlit application ├── .gitignore # Ignore unnecessary files in version control ├── emotrack-ad232-fire... # Firebase authentication config (if applicable) ├── requirements.txt # Python dependencies for the project └── README.md # Project documentation


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/riteshshinde04/EmoTrack.git
cd EmoTrack
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas matplotlib torch transformers scikit-learn
```

3. Run the app
```bash
streamlit run main.py
```
Visit: http://localhost:8501/ in your browser.

## 🧠 How It Works
-User enters an activity or text.

-The system maps this activity to estimated emotional levels using:

-Predefined logic

-Machine Learning models (BERT, SVM, Logistic Regression, Random Forest Classifier)

-Emotion levels are stabilized and logged for trend analysis.

# 📊 Sample Output

## 🏆 Model Performance
Below is the performance of different machine learning models tested for emotion classification.

| Model                | Test Accuracy | Precision (Happy) | Precision (Sad) | Recall (Happy) | Recall (Sad) | F1-Score (Happy) | F1-Score (Sad) |
|----------------------|--------------|------------------|----------------|---------------|--------------|----------------|---------------|
| Logistic Regression | 0.8543       | 0.89             | 0.85           | 0.81          | 0.83         | 0.87           | 0.82          |
| Random Forest       | 0.8201       | 0.86             | 0.80           | 0.79          | 0.82         | 0.83           | 0.81          |
| SVC                 | 0.8650       | 0.91             | 0.83           | 0.86          | 0.85         | 0.88           | 0.84          |

## 📌 Future Enhancements
Adding support for more emotion categories.

Enhancing the dashboard with better visualizations.

Improving model accuracy with additional training data.

## 📜 License
This project is licensed under the MIT License.
