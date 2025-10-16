import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import re
import html

class DepressionDetector:
    """Inference wrapper for depression detection models"""
    
    def __init__(self, model_type='baseline', model_path=None):
        """
        Initialize the detector
        
        Args:
            model_type: 'baseline' for TF-IDF+LR or 'transformer' for DistilBERT
            model_path: Path to saved model
        """
        self.model_type = model_type
        
        if model_type == 'baseline':
            if model_path is None:
                model_path = 'models/tfidf_lr_baseline.joblib'
            print(f"Loading baseline model from {model_path}...")
            self.model = joblib.load(model_path)
            try:
                self.metadata = joblib.load('models/baseline_metadata.joblib')
                self.threshold = self.metadata.get('optimal_threshold', 0.5)
            except:
                self.threshold = 0.5
                
        elif model_type == 'transformer':
            if model_path is None:
                model_path = 'models/distilbert_finetuned'
            print(f"Loading transformer model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.threshold = 0.5
        
        print(f"Model loaded successfully! (type: {model_type})")
    
    def clean_text(self, text):
        """Clean input text"""
        if not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'u/[A-Za-z0-9_-]+', ' ', text)
        text = re.sub(r'r/[A-Za-z0-9_-]+', ' ', text)
        text = re.sub(r'[^A-Za-z0-9\s\.\,\'\-\?!]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    
    def predict(self, text, return_explanation=False):
        """
        Predict depression likelihood
        
        Args:
            text: Input text
            return_explanation: If True, return feature explanations
            
        Returns:
            dict with keys: label, probability, risk_level, explanation (optional)
        """
        if not text or len(text.strip()) < 10:
            return {
                'error': 'Text too short (minimum 10 characters)',
                'label': 0,
                'probability': 0.0,
                'risk_level': 'unknown'
            }
        
        # Limit text length
        text = text[:2000]
        text_clean = self.clean_text(text)
        
        if self.model_type == 'baseline':
            return self._predict_baseline(text_clean, return_explanation)
        else:
            return self._predict_transformer(text, return_explanation)
    
    def _predict_baseline(self, text_clean, return_explanation):
        """Predict using baseline model"""
        proba = self.model.predict_proba([text_clean])[0]
        prob_depression = float(proba[1])
        label = int(prob_depression >= self.threshold)
        
        result = {
            'label': label,
            'probability': prob_depression,
            'risk_level': self._get_risk_level(prob_depression),
            'model_type': 'baseline'
        }
        
        if return_explanation:
            # Get top features contributing to prediction
            vectorizer = self.model.named_steps['tfidf']
            classifier = self.model.named_steps['clf']
            
            X = vectorizer.transform([text_clean])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get feature contributions
            feature_values = X.toarray()[0]
            coefficients = classifier.coef_[0]
            contributions = feature_values * coefficients
            
            # Get top positive contributors
            top_indices = np.argsort(contributions)[-10:][::-1]
            top_features = []
            for idx in top_indices:
                if feature_values[idx] > 0:
                    top_features.append({
                        'word': feature_names[idx],
                        'contribution': float(contributions[idx])
                    })
            
            result['explanation'] = top_features[:5]
        
        return result
    
    def _predict_transformer(self, text, return_explanation):
        """Predict using transformer model"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prob_depression = float(probs[0][1].cpu().numpy())
        
        label = int(prob_depression >= self.threshold)
        
        result = {
            'label': label,
            'probability': prob_depression,
            'risk_level': self._get_risk_level(prob_depression),
            'model_type': 'transformer'
        }
        
        if return_explanation:
            # Simple token-level explanation (attention-based would be better)
            tokens = self.tokenizer.tokenize(text)[:20]
            result['explanation'] = {
                'top_tokens': tokens,
                'note': 'Full attention-based explanation requires additional analysis'
            }
        
        return result
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return 'low'
        elif probability < 0.6:
            return 'moderate'
        elif probability < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def batch_predict(self, texts):
        """Predict on multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

# PHQ-9 Questionnaire Handler
class PHQ9Handler:
    """Handle PHQ-9 questionnaire scoring"""
    
    QUESTIONS = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling/staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or that you are a failure",
        "Trouble concentrating on things",
        "Moving or speaking slowly, or being fidgety/restless",
        "Thoughts that you would be better off dead or of hurting yourself"  # Q9 - critical
    ]
    
    @staticmethod
    def score_phq9(responses):
        """
        Score PHQ-9 responses
        
        Args:
            responses: List of 9 integers (0-3 for each question)
            
        Returns:
            dict with total_score, severity, emergency_flag
        """
        if len(responses) != 9:
            raise ValueError("PHQ-9 requires exactly 9 responses")
        
        if not all(0 <= r <= 3 for r in responses):
            raise ValueError("All responses must be between 0-3")
        
        total = sum(responses)
        q9_score = responses[8]  # Question 9 about self-harm
        
        # Determine severity
        if total <= 4:
            severity = 'minimal'
        elif total <= 9:
            severity = 'mild'
        elif total <= 14:
            severity = 'moderate'
        elif total <= 19:
            severity = 'moderately_severe'
        else:
            severity = 'severe'
        
        return {
            'total_score': total,
            'severity': severity,
            'emergency_flag': q9_score > 0,  # Any positive Q9 is emergency
            'q9_score': q9_score
        }

if __name__ == "__main__":
    # Test the detector
    print("Testing Depression Detector\n")
    
    # Test with baseline model
    if os.path.exists('models/tfidf_lr_baseline.joblib'):
        detector = DepressionDetector(model_type='baseline')
        
        test_texts = [
            "I had a great day today! Everything went perfectly and I feel amazing.",
            "I feel so alone and worthless. Nothing makes me happy anymore. I can't sleep.",
            "Just finished my project at work. Looking forward to the weekend."
        ]
        
        print("Testing predictions:")
        print("="*60)
        for i, text in enumerate(test_texts, 1):
            result = detector.predict(text, return_explanation=True)
            print(f"\nText {i}: {text[:60]}...")
            print(f"Label: {result['label']} (0=Normal, 1=Depression)")
            print(f"Probability: {result['probability']:.3f}")
            print(f"Risk Level: {result['risk_level']}")
            if 'explanation' in result:
                print("Top indicators:", [f['word'] for f in result['explanation'][:3]])
    
    # Test PHQ-9
    print("\n" + "="*60)
    print("Testing PHQ-9 Handler")
    print("="*60)
    
    test_responses = [2, 2, 1, 2, 1, 2, 1, 1, 0]  # Example responses
    result = PHQ9Handler.score_phq9(test_responses)
    print(f"\nPHQ-9 Score: {result['total_score']}")
    print(f"Severity: {result['severity']}")
    print(f"Emergency Flag (Q9): {result['emergency_flag']}")