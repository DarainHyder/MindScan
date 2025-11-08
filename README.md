# 🧠 Mental Health Detection System

AI-powered depression detection using NLP and machine learning. Includes text analysis and PHQ-9 questionnaire scoring.

## ⚠️ Important Disclaimer

**This tool is for educational and research purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. If you or someone you know is experiencing mental health difficulties, please consult a licensed mental health professional.**

## 🎯 Features

- **Text Analysis**: Detects depression indicators from free-form text
- **PHQ-9 Questionnaire**: Standard clinical depression screening tool
- **Two Model Options**: 
  - Baseline: Fast TF-IDF + Logistic Regression
  - Advanced: Fine-tuned DistilBERT transformer
- **Risk Assessment**: Categorizes risk levels (low, moderate, high, very high)
- **Explainability**: Shows key features contributing to predictions
- **Web Interface**: User-friendly Flask web application
- **REST API**: Easy integration with other applications

## 📁 Project Structure

```
mental-health-ml/
├── data/
│   ├── raw/                    # Place downloaded datasets here
│   └── processed/              # Processed data (auto-generated)
├── src/
│   ├── preprocess.py          # Data cleaning and preprocessing
│   ├── train_baseline.py      # Train TF-IDF + LR model
│   ├── train_transformer.py   # Train DistilBERT model
│   └── inference.py           # Prediction wrapper
├── notebooks/
│   └── 01_eda.py             # Exploratory data analysis
├── models/                    # Saved models (auto-generated)
├── outputs/
│   └── figures/              # Visualizations (auto-generated)
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Start Guide

### Step 1: Clone and Setup

```bash
# Create project directory
mkdir mental-health-ml
cd mental-health-ml

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Download Dataset

**Kaggle Reddit Mental Health Dataset (Recommended for Quick Start)**

1. Go to: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
2. Click "Download" (requires free Kaggle account)
3. Extract the CSV file
4. Place it in: `data/raw/mental_health.csv`

### Step 3: Create Directory Structure

```bash
# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p outputs/figures
mkdir -p notebooks
mkdir -p src
```

### Step 4: Preprocess Data

```bash
python src/preprocess.py
```

This will:
- Load the raw dataset
- Clean and anonymize text
- Remove noise (URLs, usernames, etc.)
- Save processed data to `data/processed/`

### Step 5: Run Exploratory Data Analysis (Optional but Recommended)

```bash
python notebooks/01_eda.py
```

This generates visualizations in `outputs/figures/`:
- Label distribution
- Text length analysis
- Top n-grams per class
- Word frequency analysis

### Step 6: Train the Model

**Option A: Train Baseline Model (Fast, CPU-friendly)**

```bash
python src/train_baseline.py
```

Training time: ~5-10 minutes on CPU

**Option B: Train Transformer Model (Better accuracy, needs GPU)**

```bash
python src/train_transformer.py
```

Training time: ~30-60 minutes on GPU, several hours on CPU

⚠️ **Note**: If training transformer on CPU, you'll be prompted to confirm.

### Step 7: Test the Model

```bash
python src/inference.py
```

This will run test predictions on sample texts.

### Step 8: Run the Web Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## 📊 Dataset Details

### Kaggle Reddit Mental Health Dataset

- **Size**: ~10,000+ samples
- **Labels**: Binary (depression indicators vs normal)
- **Source**: Reddit posts from mental health subreddits
- **License**: Check Kaggle page for current terms

### DAIC-WOZ Dataset

- **Size**: 189 clinical interviews
- **Labels**: PHQ-8 scores (0-24)
- **Source**: USC Institute for Creative Technologies
- **License**: Requires application and agreement

## 🎓 Model Performance

### Baseline Model (TF-IDF + Logistic Regression)

- **Training Time**: 5-10 minutes (CPU)
- **Inference Speed**: <10ms per text
- **Expected Metrics**:
  - Precision: ~0.75-0.85
  - Recall: ~0.70-0.80
  - F1-Score: ~0.72-0.82
  - PR-AUC: ~0.80-0.88

### Transformer Model (DistilBERT)

- **Training Time**: 30-60 minutes (GPU)
- **Inference Speed**: ~50-100ms per text
- **Expected Metrics**:
  - Precision: ~0.80-0.90
  - Recall: ~0.75-0.85
  - F1-Score: ~0.78-0.87
  - PR-AUC: ~0.85-0.92

## 🔌 API Usage

### Text Analysis Endpoint

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so alone and worthless", "return_explanation": true}'
```

Response:
```json
{
  "label": 1,
  "probability": 0.873,
  "risk_level": "high",
  "model_type": "baseline",
  "explanation": [
    {"word": "alone", "contribution": 0.234},
    {"word": "worthless", "contribution": 0.198}
  ]
}
```

### PHQ-9 Scoring Endpoint

```bash
curl -X POST http://localhost:5000/api/phq9 \
  -H "Content-Type: application/json" \
  -d '{"responses": [2, 2, 1, 2, 1, 2, 1, 1, 0]}'
```

Response:
```json
{
  "total_score": 12,
  "severity": "moderate",
  "emergency_flag": false,
  "q9_score": 0
}
```

## 🛠️ Configuration

### Switching Between Models

Edit `app.py` line 12:

```python
MODEL_TYPE = 'baseline'  # or 'transformer'
```

### Adjusting Prediction Threshold

Edit `src/inference.py` in the `__init__` method:

```python
self.threshold = 0.5  # Lower = higher recall, higher = higher precision
```

### Customizing Training

Edit hyperparameters in:
- `src/train_baseline.py` - lines 38-42 (param_grid)
- `src/train_transformer.py` - lines 63-75 (TrainingArguments)

## 📈 Understanding Results

### Risk Levels

- **Low** (0-30%): Minimal indicators of depression
- **Moderate** (30-60%): Some indicators present
- **High** (60-80%): Significant indicators present
- **Very High** (80-100%): Strong indicators present

### PHQ-9 Severity Levels

- **Minimal** (0-4): Little to no depression
- **Mild** (5-9): Mild depression symptoms
- **Moderate** (10-14): Moderate depression
- **Moderately Severe** (15-19): Moderately severe depression
- **Severe** (20-27): Severe depression

### Question 9 Emergency Flag

If PHQ-9 Question 9 (thoughts of self-harm) scores > 0, an emergency flag is raised and crisis resources are displayed prominently.

## 🔒 Privacy & Ethics

### Data Handling

- **Anonymization**: All usernames, emails, and personal identifiers are removed
- **No Storage**: Web app doesn't store user inputs by default
- **Consent**: Always get explicit consent before storing any text data
- **Encryption**: Use HTTPS in production

### Ethical Considerations

1. **Not a Diagnosis**: Emphasize this is a screening tool, not diagnostic
2. **Professional Referral**: Always recommend consulting licensed professionals
3. **Crisis Response**: Implement immediate crisis resource display for high-risk cases
4. **Bias Awareness**: Monitor for demographic disparities in predictions
5. **Transparency**: Explain how the model works to users

## 🐛 Troubleshooting

### Issue: "No module named 'transformers'"

```bash
pip install transformers datasets torch
```

### Issue: "CUDA out of memory" during transformer training

Reduce batch size in `src/train_transformer.py`:

```python
per_device_train_batch_size=8,  # instead of 16
```

### Issue: Model file not found

Make sure you've trained a model first:

```bash
python src/train_baseline.py
```

### Issue: Dataset not loading

Check that your CSV is in the correct location:
- For Kaggle: `data/raw/mental_health.csv`
- For DAIC-WOZ: `data/raw/daic_woz/`


## 📚 Additional Resources

### Technical Documentation

- **PHQ-9 Information**: https://www.hiv.uw.edu/page/mental-health-screening/phq-9
- **Transformers Library**: https://huggingface.co/docs/transformers
- **Scikit-learn**: https://scikit-learn.org/stable/

### Datasets

- **DAIC-WOZ**: https://dcapswoz.ict.usc.edu/
- **Kaggle Datasets**: https://www.kaggle.com/datasets

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add LIME/SHAP explainability visualizations
- [ ] Implement model calibration
- [ ] Add demographic bias analysis
- [ ] Add unit tests
- [ ] Implement A/B testing framework
- [ ] Add model monitoring dashboard

## 📝 License

This project is for educational purposes. Please check individual dataset licenses before use.

---

Built with ❤️ for mental health awareness and education.

**Remember: It's okay to not be okay. Help is available.**