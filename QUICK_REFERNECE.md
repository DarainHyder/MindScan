# 📋 Quick Reference Guide

## 🎯 ONE-PAGE SETUP

### 1️⃣ Download Dataset (5 minutes)
- Go to: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
- Download CSV → Save as `data/raw/mental_health.csv`

### 2️⃣ Install & Run (15 minutes)
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
./run_all.sh       # Linux/Mac
run_all.bat       # Windows
```

### 3️⃣ Start Web App
```bash
python app.py
# Open: http://localhost:5000
```

---

## 📂 Where Files Go

```
mental-health-ml/
├── data/raw/mental_health.csv    ← PUT DATASET HERE
├── app.py                         ← Main web application
├── requirements.txt               ← Python packages
├── src/
│   ├── preprocess.py             ← Data cleaning
│   ├── train_baseline.py         ← Train fast model
│   ├── train_transformer.py      ← Train better model
│   └── inference.py              ← Make predictions
└── notebooks/
    └── 01_eda.py                 ← Data analysis
```

---

## ⚡ Quick Commands

| Task | Command |
|------|---------|
| **Clean data** | `python src/preprocess.py` |
| **Explore data** | `python notebooks/01_eda.py` |
| **Train fast model** | `python src/train_baseline.py` |
| **Train accurate model** | `python src/train_transformer.py` |
| **Test model** | `python src/inference.py` |
| **Start web app** | `python app.py` |
| **Full pipeline** | `./run_all.sh` or `run_all.bat` |

---

## 🔧 Common Fixes

| Problem | Solution |
|---------|----------|
| **Dataset not found** | Check file is at `data/raw/mental_health.csv` |
| **Port 5000 in use** | Change port in `app.py` line 461 to `5001` |
| **Out of memory** | Reduce batch size in training files |
| **Slow training** | Use baseline model, skip transformer |
| **Module not found** | Run `pip install -r requirements.txt` |

---

## 📊 What Each File Does

| File | Purpose | Time |
|------|---------|------|
| `preprocess.py` | Cleans text, removes noise | 2-3 min |
| `01_eda.py` | Creates charts and graphs | 3-5 min |
| `train_baseline.py` | Trains fast TF-IDF model | 5-10 min |
| `train_transformer.py` | Trains accurate BERT model | 30-60 min |
| `inference.py` | Tests predictions | 1 min |
| `app.py` | Web interface | Running |

---

## 🌐 API Quick Reference

### Analyze Text
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so hopeless", "return_explanation": true}'
```

### Score PHQ-9
```bash
curl -X POST http://localhost:5000/api/phq9 \
  -H "Content-Type: application/json" \
  -d '{"responses": [2,2,1,2,1,2,1,1,0]}'
```

---

## 🎨 Output Files

After running, check these locations:

- **📊 Visualizations**: `outputs/figures/`
  - `label_distribution.png` - Class balance
  - `text_length_analysis.png` - Text stats
  - `confusion_matrix_baseline.png` - Model performance
  - `pr_curve_baseline.png` - Precision-recall curve

- **🤖 Models**: `models/`
  - `tfidf_lr_baseline.joblib` - Trained model
  - `baseline_metadata.joblib` - Performance metrics
  - `distilbert_finetuned/` - Transformer model (if trained)

- **💾 Processed Data**: `data/processed/`
  - `mental_health_clean.parquet` - Cleaned dataset

---

## 🚨 Emergency Resources

**If Q9 > 0 or risk_level is "very_high":**

- 🇺🇸 **US**: Call/Text **988** (Suicide & Crisis Lifeline)
- 💬 **Text**: Send **HOME** to **741741** (Crisis Text Line)
- 🌍 **International**: https://findahelpline.com
- 🆘 **Emergency**: Call local emergency services

---

## 📈 Model Performance

| Model | Speed | Accuracy | When to Use |
|-------|-------|----------|-------------|
| **Baseline (TF-IDF)** | <10ms | ~78% F1 | Production, fast responses |
| **Transformer (BERT)** | ~100ms | ~85% F1 | High accuracy needed |

---

## 🔄 Typical Workflow

1. **First Time Setup** (20 min)
   - Download dataset
   - Install packages
   - Run `./run_all.sh`

2. **Use the Model**
   - Start `python app.py`
   - Open browser to http://localhost:5000
   - Test with sample texts
   - Try PHQ-9 questionnaire

3. **Improve the Model**
   - Review visualizations in `outputs/figures/`
   - Check metrics in console output
   - Optionally train transformer: `python src/train_transformer.py`
   - Adjust threshold in `src/inference.py`

4. **Deploy**
   - Switch to transformer model in `app.py` if trained
   - Add HTTPS for production
   - Set up proper database for logging
   - Implement user authentication

---

## 🎓 Understanding the Code

### Data Flow
```
Raw CSV → preprocess.py → Clean Parquet → train_*.py → Model → inference.py → app.py → Web UI
```

### Model Pipeline
```
Text Input → Clean Text → TF-IDF Vectorization → Logistic Regression → Probability → Risk Level
```

### PHQ-9 Scoring
```
9 Questions (0-3 each) → Total Score (0-27) → Severity Category + Emergency Flag
```

---

## 💡 Tips & Tricks

### Speed Up Training
```python
# In train_baseline.py, reduce param_grid
param_grid = {
    'tfidf__max_features': [10000],  # Just one value
    'clf__C': [1.0]                  # Just one value
}
```

### Increase Recall (Catch More Cases)
```python
# In src/inference.py, line ~30
self.threshold = 0.3  # Lower threshold = higher recall
```

### Reduce Memory Usage
```python
# In preprocess.py, add sampling
df = df.sample(n=5000, random_state=42)  # Use 5000 samples
```

### Test Without Training
```python
# Use the inference.py test examples
python src/inference.py
```

---

## 🔍 Debugging Checklist

**Model won't load?**
- ✅ Check `models/` directory exists
- ✅ Verify `tfidf_lr_baseline.joblib` is present
- ✅ Re-run `python src/train_baseline.py`

**Web app crashes?**
- ✅ Check port 5000 is free
- ✅ Verify model file exists
- ✅ Check console for error messages
- ✅ Try: `python app.py 2>&1 | tee app.log`

**Predictions seem wrong?**
- ✅ Check training metrics in console
- ✅ Review confusion matrix in `outputs/figures/`
- ✅ Test with obvious examples
- ✅ Verify data preprocessing is correct

**Out of memory?**
- ✅ Close other applications
- ✅ Reduce batch size
- ✅ Use smaller dataset sample
- ✅ Stick to baseline model (skip transformer)

---

## 📱 Integration Examples

### Python Script
```python
from src.inference import DepressionDetector

detector = DepressionDetector(model_type='baseline')
result = detector.predict("I feel so alone")
print(f"Risk: {result['risk_level']}, Prob: {result['probability']:.2f}")
```

### JavaScript (Frontend)
```javascript
fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: userInput})
})
.then(r => r.json())
.then(data => console.log(data.risk_level));
```

### cURL Test
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"test input","return_explanation":true}'
```

---

## 🎯 Key Metrics Explained

### Precision
- **What**: Of all "depression" predictions, how many were correct?
- **High Precision**: Few false alarms
- **Target**: >0.75

### Recall (Sensitivity)
- **What**: Of all actual depression cases, how many did we catch?
- **High Recall**: Don't miss cases (critical for mental health!)
- **Target**: >0.80

### F1-Score
- **What**: Balance between precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Target**: >0.75

### PR-AUC (Precision-Recall Area Under Curve)
- **What**: Overall model quality across all thresholds
- **Range**: 0 to 1 (higher is better)
- **Target**: >0.80

---

## 🔐 Security Best Practices

### For Production:
```python
# app.py modifications for production

# 1. Add input validation
@app.route('/api/predict', methods=['POST'])
def predict():
    text = request.get_json().get('text', '')
    if len(text) > 2000:  # Limit length
        return jsonify({'error': 'Text too long'}), 400
    # ... rest of code

# 2. Add rate limiting
from flask_limiter import Limiter
limiter = Limiter(app, default_limits=["100 per hour"])

# 3. Use HTTPS in production
# 4. Don't log sensitive user data
# 5. Implement user authentication
# 6. Set proper CORS policies
```

---

## 📚 Additional Resources

### Learn More About:
- **Mental Health Screening**: https://www.hiv.uw.edu/page/mental-health-screening/phq-9
- **NLP for Mental Health**: Search "ACL Workshop on Computational Linguistics and Clinical Psychology"
- **Transformers**: https://huggingface.co/course
- **Model Explainability**: https://shap.readthedocs.io/

### Datasets:
- **RSDD**: https://georgetown-ir-lab.github.io/
- **DAIC-WOZ**: https://dcapswoz.ict.usc.edu/
- **CLPsych Shared Tasks**: Search for annual shared tasks

---

## ✅ Final Checklist

Before deployment:
- [ ] Model trained and tested
- [ ] Metrics look reasonable (F1 > 0.70)
- [ ] Disclaimer clearly visible
- [ ] Crisis resources prominent
- [ ] PHQ-9 Q9 emergency handling works
- [ ] Input validation in place
- [ ] Rate limiting configured
- [ ] HTTPS enabled (production)
- [ ] Privacy policy written
- [ ] User consent flow implemented
- [ ] Model card documented
- [ ] Backup plan for model failures

---

## 🆘 Still Stuck?

### Check These Common Issues:

**1. Wrong directory?**
```bash
pwd  # Should show .../mental-health-ml
cd mental-health-ml  # Navigate to correct directory
```

**2. Python version?**
```bash
python --version  # Should be 3.8 or higher
python3 --version  # Try python3 if python doesn't work
```

**3. Dataset columns wrong?**
```bash
# Check what columns are in your CSV
python -c "import pandas as pd; print(pd.read_csv('data/raw/mental_health.csv').columns)"
```

**4. Dependencies conflict?**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## 🎉 Success Indicators

You'll know everything is working when:

✅ `run_all.sh` completes without errors  
✅ You see "Model loaded successfully!" when starting app.py  
✅ Browser shows the web interface at localhost:5000  
✅ Test predictions return reasonable probabilities  
✅ Confusion matrix shows diagonal values > 70%  
✅ PHQ-9 scoring works correctly  

---

## 📞 Contact & Support

**Remember**: This is an educational project. For real mental health concerns:
- **Call 988** (US Suicide & Crisis Lifeline)
- **Text HOME to 741741** (Crisis Text Line)
- **Visit findahelpline.com** for international resources

---

**Built with care for mental health awareness 💙**

**Last Updated**: 2025  
**Version**: 1.0  
**License**: Educational Use