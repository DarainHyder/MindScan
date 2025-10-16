import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def run_eda(data_path="data/processed/mental_health_clean.parquet"):
    """Run complete exploratory data analysis"""
    
    print("="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df = pd.read_parquet(data_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Basic statistics
    print("\n" + "="*60)
    print("1. BASIC STATISTICS")
    print("="*60)
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    
    # 2. Label distribution
    print("\n" + "="*60)
    print("2. LABEL DISTRIBUTION")
    print("="*60)
    label_counts = df['label'].value_counts()
    print(f"\n{label_counts}")
    print(f"\nClass balance: {label_counts[1]/label_counts[0]:.2f}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    label_counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Label Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Label (0=Normal, 1=Depression)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/label_distribution.png', dpi=300)
    print("Saved: outputs/figures/label_distribution.png")
    
    # 3. Text length analysis
    print("\n" + "="*60)
    print("3. TEXT LENGTH ANALYSIS")
    print("="*60)
    df['len_chars'] = df['text_clean'].str.len()
    df['len_words'] = df['text_clean'].str.split().str.len()
    
    print("\nCharacter length statistics:")
    print(df['len_chars'].describe())
    print("\nWord length statistics:")
    print(df['len_words'].describe())
    
    # Plot text length by label
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for label in [0, 1]:
        data = df[df['label'] == label]['len_words']
        axes[0].hist(data, bins=50, alpha=0.6, label=f'Label {label}')
    axes[0].set_xlabel('Number of Words')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Text Length Distribution by Label')
    axes[0].legend()
    axes[0].set_xlim(0, 500)
    
    df.boxplot(column='len_words', by='label', ax=axes[1])
    axes[1].set_xlabel('Label (0=Normal, 1=Depression)')
    axes[1].set_ylabel('Number of Words')
    axes[1].set_title('Word Count by Label')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/text_length_analysis.png', dpi=300)
    print("Saved: outputs/figures/text_length_analysis.png")
    
    # 4. Top n-grams per class
    print("\n" + "="*60)
    print("4. TOP N-GRAMS BY CLASS")
    print("="*60)
    
    tf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
    X = tf.fit_transform(df['text_clean'].fillna(''))
    feature_names = tf.get_feature_names_out()
    
    # Get top features per class
    for label in [0, 1]:
        print(f"\nTop 20 features for label {label}:")
        mask = np.array(df['label'] == label)
        X_label = X[mask]
        mean_tfidf = np.asarray(X_label.mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[-20:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        for i, feat in enumerate(top_features, 1):
            print(f"  {i:2d}. {feat:20s} (score: {mean_tfidf[top_idx[i-1]]:.4f})")
    
    # 5. Sample texts
    print("\n" + "="*60)
    print("5. SAMPLE TEXTS")
    print("="*60)
    
    print("\nSample normal texts (label=0):")
    for i, text in enumerate(df[df['label'] == 0]['text'].head(3), 1):
        print(f"\n{i}. {text[:200]}...")
    
    print("\n\nSample depression texts (label=1):")
    for i, text in enumerate(df[df['label'] == 1]['text'].head(3), 1):
        print(f"\n{i}. {text[:200]}...")
    
    # 6. Word frequency analysis
    print("\n" + "="*60)
    print("6. MOST COMMON WORDS BY CLASS")
    print("="*60)
    
    for label in [0, 1]:
        texts = ' '.join(df[df['label'] == label]['text_clean'].tolist())
        words = texts.split()
        common = Counter(words).most_common(30)
        print(f"\nTop 30 words for label {label}:")
        for word, count in common:
            if len(word) > 2:  # Skip very short words
                print(f"  {word:15s}: {count}")
    
    print("\n" + "="*60)
    print("EDA COMPLETE! Check outputs/figures/ for visualizations.")
    print("="*60)
    
    return df

if __name__ == "__main__":
    # Run EDA
    if os.path.exists("data/processed/mental_health_clean.parquet"):
        df = run_eda("data/processed/mental_health_clean.parquet")
    else:
        print("Error: Processed data not found!")
        print("Please run src/preprocess.py first")