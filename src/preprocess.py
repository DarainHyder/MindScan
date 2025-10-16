import pandas as pd
import re
import html
import os

def clean_text(s):
    """Clean text for NLP processing"""
    if not isinstance(s, str): 
        return ""
    s = html.unescape(s)
    s = re.sub(r'http\S+', ' ', s)           
    s = re.sub(r'u/[A-Za-z0-9_-]+', ' ', s)  
    s = re.sub(r'r/[A-Za-z0-9_-]+', ' ', s)  
    s = re.sub(r'[^A-Za-z0-9\s\.\,\'\-\?!]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def load_kaggle_reddit_data(filepath):
    """Load Kaggle Reddit mental health dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Kaggle dataset typically has 'statement' and 'status' columns
    if 'statement' in df.columns:
        df = df.rename(columns={'statement': 'text'})
    
    # Map labels to binary (depression=1, normal=0)
    if 'status' in df.columns:
        label_map = {
            'depression': 1, 'Depression': 1,
            'suicidal': 1, 'Suicidal': 1,
            'anxiety': 1, 'Anxiety': 1,
            'stress': 1, 'Stress': 1,
            'bipolar': 1, 'Bipolar': 1,
            'personality disorder': 1,
            'normal': 0, 'Normal': 0
        }
        df['label'] = df['status'].map(label_map).fillna(0).astype(int)
    
    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.len() > 10]  # Remove very short texts
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df[['text', 'text_clean', 'label']]

def load_daic_woz_data(transcript_dir, labels_file):
    """Load DAIC-WOZ dataset (clinical interviews)"""
    print(f"Loading DAIC-WOZ data...")
    
    # Load PHQ-8 scores
    labels_df = pd.read_csv(labels_file)
    
    # Read all transcript files
    data = []
    for filename in os.listdir(transcript_dir):
        if filename.endswith('.txt') or filename.endswith('.csv'):
            participant_id = filename.split('_')[0]
            filepath = os.path.join(transcript_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Get PHQ score for this participant
            phq_score = labels_df[labels_df['Participant_ID'] == int(participant_id)]['PHQ8_Score'].values
            if len(phq_score) > 0:
                label = 1 if phq_score[0] >= 10 else 0  # PHQ-8 >= 10 indicates depression
                data.append({
                    'text': text,
                    'label': label,
                    'phq_score': phq_score[0]
                })
    
    df = pd.DataFrame(data)
    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.len() > 20]
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df

def save_processed_data(df, output_path):
    """Save processed data"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    # Example usage for Kaggle dataset
    if os.path.exists("data/raw/mental_health.csv"):
        df = load_kaggle_reddit_data("data/raw/mental_health.csv")
        save_processed_data(df, "data/processed/mental_health_clean.parquet")
    
    # Example usage for DAIC-WOZ
    elif os.path.exists("data/raw/daic_woz/transcripts"):
        df = load_daic_woz_data(
            "data/raw/daic_woz/transcripts",
            "data/raw/daic_woz/labels.csv"
        )
        save_processed_data(df, "data/processed/daic_woz_clean.parquet")
    
    else:
        print("No dataset found in data/raw/")
        print("Please download dataset and place in data/raw/")