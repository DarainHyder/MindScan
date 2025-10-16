# 🚀 Complete Setup Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

## Step-by-Step Installation

### 1. Create Project Structure

```bash
# Create main directory
mkdir mental-health-ml
cd mental-health-ml

# Create all subdirectories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p outputs/figures
mkdir -p notebooks
mkdir -p src
```

### 2. Save All Project Files

Copy all the provided code files into their respective locations:

```
mental-health-ml/
├── requirements.txt          # Save this file in root
├── app.py                   # Save this file in root
├── README.md                # Save this file in root
├── run_all.sh              # Save this file in root (Linux/Mac)
├── run_all.bat             # Save this file in root (Windows)
├── src/
│   ├── preprocess.py       # Save in src/
│   ├── train_baseline.py   # Save in src/
│   ├── train_transformer.py # Save in src/
│   └── inference.py        # Save in src/
└── notebooks/
    └── 01_eda.py          # Save in notebooks/
```

### 3. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes depending on your internet speed.

If you get permission errors, try:
```bash
pip install --user -r requirements.txt
```

### 4. Download Dataset

**Option 1: Kaggle Dataset (Recommended for Beginners)**

1. Go to https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
2. Click the "Download" button (you'll need to create a free Kaggle account)
3. Extract the downloaded ZIP file
4. Find the CSV file (usually named something like `Combined Data.csv` or `mental_health.csv`)
5. Rename it to `mental_health.csv` if needed
6. Move it to: `data/raw/mental_health.csv`

**Verify the file location:**
```bash
# On Linux/Mac
ls data/raw/mental_health.csv

# On Windows
dir data\raw\mental_health.csv
```

You should see the file listed.

**Option 2: DAIC-WOZ Dataset (Advanced - Clinical Data)**

1. Go to https://dcapswoz.ict.usc.edu/
2. Fill out the access request form
3. Wait for approval email (usually 1-2 days)
4. Download the dataset
5. Extract to `data/raw/daic_woz/`

### 5. Run the Complete Pipeline

**On Linux/Mac:**
```bash
# Make the script executable
chmod +x run_all.sh

# Run the pipeline
./run_all.sh
```

**On Windows:**
```batch
run_all.bat
```

**Or run steps manually:**

```bash
# Step 1: Preprocess data
python src/preprocess.py

# Step 2: Run EDA (optional)
python notebooks/01_eda.py

# Step 3: Train model
python src/train_baseline.py

# Step 4: Test predictions
python src/inference.py
```

### 6. Start the Web Application

```bash
python app.py
```

Open your web browser and go to: **http://localhost:5000**

You should see the Mental Health Detection interface!

## Troubleshooting Common Issues

### Issue 1: "pip: command not found"

**Solution:**
```bash
# Try pip3 instead
pip3 install -r requirements.txt

# Or use python -m pip
python -m pip install -r requirements.txt
```

### Issue 2: "No module named 'sklearn'"

**Solution:**
```bash
pip install scikit-learn
```

### Issue 3: Dataset file not found

**Error message:** `FileNotFoundError: data/raw/mental_health.csv not found`

**Solution:**
1. Verify you downloaded the dataset
2. Check the file is named exactly `mental_health.csv`
3. Check it's in the correct location: `data/raw/mental_health.csv`

**Check current directory:**
```bash
# On Linux/Mac
pwd

# On Windows
cd
```

Make sure you're in the `mental-health-ml` directory.

### Issue 4: "Port 5000 already in use"

**Solution:**

Edit `app.py` line 461 and change the port:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

Then access at http://localhost:5001

### Issue 5: Training takes too long

**For baseline model:**
- Should take 5-10 minutes on most computers
- If longer, reduce dataset size in `src/preprocess.py`

**For transformer model:**
- Will be SLOW on CPU (2-4 hours)
- Recommended to use Google Colab with free GPU
- Or skip transformer and use baseline model

### Issue 6: Memory errors during training

**Solution:**

Reduce batch size in training scripts:

For baseline (`src/train_baseline.py`), line 39:
```python
'tfidf__max_features': [5000, 10000],  # reduced from 15000
```

For transformer (`src/train_transformer.py`), line 68:
```python
per_device_train_batch_size=8,  # reduced from 16
```

### Issue 7: CSV parsing errors

**Error message:** `ParserError: Error tokenizing data`

**Solution:**

The CSV might have formatting issues. Try:

1. Open CSV in Excel or text editor
2. Save as UTF-8 CSV
3. Or modify `src/preprocess.py` line 25:
```python
df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
```

## Verifying Installation

Run this test to check everything is installed:

```python
# test_install.py
import sys
print(f"Python version: {sys.version}")

try:
    import pandas
    print("✓ pandas installed")
except:
    print("✗ pandas not installed")

try:
    import sklearn
    print("✓ scikit-learn installed")
except:
    print("✗ scikit-learn not installed")

try:
    import transformers
    print("✓ transformers installed")
except:
    print("✗ transformers not installed")

try:
    import flask
    print("✓ flask installed")
except:
    print("✗ flask not installed")

print("\nAll checks complete!")
```

Save this as `test_install.py` and run:
```bash
python test_install.py
```

## Quick Start Commands

```bash
# Full pipeline (after downloading dataset)
./run_all.sh          # Linux/Mac
run_all.bat          # Windows

# Or step by step
python src/preprocess.py       # Clean data
python notebooks/01_eda.py     # Explore data
python src/train_baseline.py   # Train model
python src/inference.py        # Test predictions
python app.py                  # Start web app
```

## File Size Reference

After setup, your project should look like:

```
mental-health-ml/              (~500MB total)
├── data/
│   ├── raw/
│   │   └── mental_health.csv  (~50-100MB)
│   └── processed/
│       └── *.parquet          (~30-50MB)
├── models/
│   └── *.joblib              (~50-100MB)
├── outputs/
│   └── figures/              (~5MB)
└── [other files]             (~10MB)
```

## Using Google Colab (Alternative)

If you have issues with local setup, you can use Google Colab:

1. Go to https://colab.research.google.com/
2. Upload all Python files
3. Upload dataset to Colab files
4. Run cells in order
5. Download trained model

## Next Steps

Once everything is running:

1. ✅ Test the web interface at http://localhost:5000
2. ✅ Try analyzing some sample texts
3. ✅ Fill out the PHQ-9 questionnaire
4. ✅ Check the visualizations in `outputs/figures/`
5. ✅ Review model performance metrics
6. 📚 Read the full README.md for advanced usage

## Getting Help

If you're still having issues:

1. Check the error message carefully
2. Search for the error online
3. Make sure Python 3.8+ is installed: `python --version`
4. Make sure all files are in the correct locations
5. Try deleting and reinstalling dependencies

## Summary Checklist

- [ ] Python 3.8+ installed
- [ ] All project files saved in correct locations
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded to `data/raw/mental_health.csv`
- [ ] Pipeline runs successfully (`./run_all.sh`)
- [ ] Web app starts (`python app.py`)
- [ ] Can access http://localhost:5000 in browser

If all checkboxes are complete, you're ready to go! 🎉