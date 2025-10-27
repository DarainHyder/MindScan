from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys

# Add src to path
sys.path.append('src')
from inference import DepressionDetector, PHQ9Handler

app = Flask(__name__)
CORS(app)

MODEL_TYPE = 'baseline'  # or 'transformer'
detector = None

def init_model():
    """Initialize the model on first request"""
    global detector
    if detector is None:
        try:
            detector = DepressionDetector(model_type=MODEL_TYPE)
            print(f"Model initialized successfully: {MODEL_TYPE}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        init_model()
        data = request.get_json()
        text = data.get('text', '')
        return_explanation = data.get('return_explanation', False)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = detector.predict(text, return_explanation=return_explanation)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/phq9', methods=['POST'])
def phq9_score():
    try:
        data = request.get_json()
        responses = data.get('responses', [])
        if len(responses) != 9:
            return jsonify({'error': 'PHQ-9 requires exactly 9 responses'}), 400

        result = PHQ9Handler.score_phq9(responses)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_type': MODEL_TYPE,
        'model_loaded': detector is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Mental Health Detection API")
    print("="*60)
    print(f"Model type: {MODEL_TYPE}")
    print("Initializing model...")

    try:
        init_model()
        print("\n✓ Model loaded successfully!")
        print("\nServer running at: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have trained a model first!")


# from flask import Flask, request, jsonify, render_template_string
# from flask_cors import CORS
# import sys
# import os

# # Add src to path
# sys.path.append('src')
# from inference import DepressionDetector, PHQ9Handler

# app = Flask(__name__)
# CORS(app)

# # Initialize model (change to 'transformer' if you trained that model)
# MODEL_TYPE = 'baseline'  # or 'transformer'
# detector = None

# def init_model():
#     """Initialize the model on first request"""
#     global detector
#     if detector is None:
#         try:
#             detector = DepressionDetector(model_type=MODEL_TYPE)
#             print(f"Model initialized successfully: {MODEL_TYPE}")
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             raise

# # HTML Template for Web Interface
# HTML_TEMPLATE = '''
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Mental Health Detection System</title>
#     <style>
#         * {
#             margin: 0;
#             padding: 0;
#             box-sizing: border-box;
#         }
        
#         body {
#             font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             min-height: 100vh;
#             padding: 20px;
#         }
        
#         .container {
#             max-width: 800px;
#             margin: 0 auto;
#             background: white;
#             border-radius: 20px;
#             padding: 40px;
#             box-shadow: 0 20px 60px rgba(0,0,0,0.3);
#         }
        
#         h1 {
#             color: #333;
#             margin-bottom: 10px;
#             font-size: 2.5em;
#         }
        
#         .disclaimer {
#             background: #fff3cd;
#             border-left: 4px solid #ffc107;
#             padding: 15px;
#             margin: 20px 0;
#             border-radius: 5px;
#             font-size: 0.9em;
#         }
        
#         .crisis-card {
#             background: #f8d7da;
#             border: 2px solid #dc3545;
#             padding: 20px;
#             margin: 20px 0;
#             border-radius: 10px;
#             display: none;
#         }
        
#         .crisis-card.show {
#             display: block;
#         }
        
#         .crisis-card h3 {
#             color: #dc3545;
#             margin-bottom: 10px;
#         }
        
#         textarea {
#             width: 100%;
#             min-height: 150px;
#             padding: 15px;
#             border: 2px solid #ddd;
#             border-radius: 10px;
#             font-size: 16px;
#             font-family: inherit;
#             resize: vertical;
#             transition: border-color 0.3s;
#         }
        
#         textarea:focus {
#             outline: none;
#             border-color: #667eea;
#         }
        
#         button {
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#             color: white;
#             border: none;
#             padding: 15px 40px;
#             font-size: 18px;
#             border-radius: 10px;
#             cursor: pointer;
#             margin-top: 20px;
#             transition: transform 0.2s;
#         }
        
#         button:hover {
#             transform: translateY(-2px);
#         }
        
#         button:disabled {
#             background: #ccc;
#             cursor: not-allowed;
#             transform: none;
#         }
        
#         .result {
#             margin-top: 30px;
#             padding: 25px;
#             border-radius: 10px;
#             display: none;
#         }
        
#         .result.show {
#             display: block;
#         }
        
#         .result.low {
#             background: #d4edda;
#             border-left: 4px solid #28a745;
#         }
        
#         .result.moderate {
#             background: #fff3cd;
#             border-left: 4px solid #ffc107;
#         }
        
#         .result.high {
#             background: #f8d7da;
#             border-left: 4px solid #dc3545;
#         }
        
#         .result h3 {
#             margin-bottom: 15px;
#         }
        
#         .metric {
#             margin: 10px 0;
#             font-size: 1.1em;
#         }
        
#         .metric strong {
#             color: #333;
#         }
        
#         .explanation {
#             margin-top: 15px;
#             padding: 15px;
#             background: rgba(0,0,0,0.05);
#             border-radius: 5px;
#         }
        
#         .loader {
#             display: none;
#             margin: 20px auto;
#             border: 4px solid #f3f3f3;
#             border-top: 4px solid #667eea;
#             border-radius: 50%;
#             width: 40px;
#             height: 40px;
#             animation: spin 1s linear infinite;
#         }
        
#         .loader.show {
#             display: block;
#         }
        
#         @keyframes spin {
#             0% { transform: rotate(0deg); }
#             100% { transform: rotate(360deg); }
#         }
        
#         .phq9-section {
#             margin-top: 40px;
#             padding-top: 40px;
#             border-top: 2px solid #eee;
#         }
        
#         .phq9-question {
#             margin: 15px 0;
#             padding: 15px;
#             background: #f8f9fa;
#             border-radius: 5px;
#         }
        
#         .phq9-options {
#             margin-top: 10px;
#         }
        
#         .phq9-options label {
#             margin-right: 20px;
#             cursor: pointer;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>🧠 Mental Health Detection</h1>
#         <p style="color: #666; margin-bottom: 20px;">AI-powered text analysis for depression indicators</p>
        
#         <div class="disclaimer">
#             <strong>⚠️ Important Disclaimer:</strong> This tool is NOT a medical diagnosis. 
#             It's an AI-based screening tool for educational purposes only. 
#             Please consult a licensed mental health professional for proper evaluation and treatment.
#         </div>
        
#         <div class="crisis-card" id="crisisCard">
#             <h3>🚨 Crisis Resources</h3>
#             <p><strong>If you're having thoughts of self-harm, please reach out immediately:</strong></p>
#             <ul style="margin-top: 10px; margin-left: 20px;">
#                 <li>National Suicide Prevention Lifeline: 988 (US)</li>
#                 <li>Crisis Text Line: Text HOME to 741741</li>
#                 <li>International: <a href="https://findahelpline.com">findahelpline.com</a></li>
#             </ul>
#         </div>
        
#         <h2 style="margin-top: 30px; color: #333;">Text Analysis</h2>
#         <textarea id="textInput" placeholder="Share your thoughts or feelings here... (minimum 10 characters)"></textarea>
        
#         <button onclick="analyzeText()" id="analyzeBtn">Analyze Text</button>
        
#         <div class="loader" id="loader"></div>
        
#         <div class="result" id="result"></div>
        
#         <!-- PHQ-9 Section -->
#         <div class="phq9-section">
#             <h2 style="color: #333;">PHQ-9 Questionnaire</h2>
#             <p style="color: #666; margin: 10px 0;">Over the last 2 weeks, how often have you been bothered by the following?</p>
#             <p style="color: #888; font-size: 0.9em; margin-bottom: 20px;">
#                 0 = Not at all | 1 = Several days | 2 = More than half the days | 3 = Nearly every day
#             </p>
            
#             <div id="phq9Questions"></div>
            
#             <button onclick="scorePHQ9()" id="phq9Btn" style="background: #28a745;">Score PHQ-9</button>
            
#             <div class="result" id="phq9Result"></div>
#         </div>
#     </div>
    
#     <script>
#         // Initialize PHQ-9 questions
#         const phq9Questions = [
#             "Little interest or pleasure in doing things",
#             "Feeling down, depressed, or hopeless",
#             "Trouble falling/staying asleep, or sleeping too much",
#             "Feeling tired or having little energy",
#             "Poor appetite or overeating",
#             "Feeling bad about yourself or that you are a failure",
#             "Trouble concentrating on things",
#             "Moving or speaking slowly, or being fidgety/restless",
#             "Thoughts that you would be better off dead or of hurting yourself"
#         ];
        
#         const questionsContainer = document.getElementById('phq9Questions');
#         phq9Questions.forEach((q, i) => {
#             const div = document.createElement('div');
#             div.className = 'phq9-question';
#             div.innerHTML = `
#                 <div style="font-weight: 500; margin-bottom: 10px;">${i + 1}. ${q}</div>
#                 <div class="phq9-options">
#                     <label><input type="radio" name="q${i}" value="0" required> 0</label>
#                     <label><input type="radio" name="q${i}" value="1"> 1</label>
#                     <label><input type="radio" name="q${i}" value="2"> 2</label>
#                     <label><input type="radio" name="q${i}" value="3"> 3</label>
#                 </div>
#             `;
#             questionsContainer.appendChild(div);
#         });
        
#         async function analyzeText() {
#             const text = document.getElementById('textInput').value;
#             const resultDiv = document.getElementById('result');
#             const loader = document.getElementById('loader');
#             const btn = document.getElementById('analyzeBtn');
            
#             if (text.length < 10) {
#                 alert('Please enter at least 10 characters');
#                 return;
#             }
            
#             // Show loader
#             loader.classList.add('show');
#             btn.disabled = true;
#             resultDiv.classList.remove('show');
            
#             try {
#                 const response = await fetch('/api/predict', {
#                     method: 'POST',
#                     headers: {'Content-Type': 'application/json'},
#                     body: JSON.stringify({text: text, return_explanation: true})
#                 });
                
#                 const data = await response.json();
                
#                 // Hide loader
#                 loader.classList.remove('show');
#                 btn.disabled = false;
                
#                 if (data.error) {
#                     alert('Error: ' + data.error);
#                     return;
#                 }
                
#                 // Show crisis card if high risk
#                 const crisisCard = document.getElementById('crisisCard');
#                 if (data.risk_level === 'high' || data.risk_level === 'very_high') {
#                     crisisCard.classList.add('show');
#                 } else {
#                     crisisCard.classList.remove('show');
#                 }
                
#                 // Display results
#                 resultDiv.className = 'result show ' + data.risk_level;
                
#                 let html = '<h3>Analysis Results</h3>';
#                 html += `<div class="metric"><strong>Depression Indicator:</strong> ${data.label === 1 ? 'Yes' : 'No'}</div>`;
#                 html += `<div class="metric"><strong>Probability:</strong> ${(data.probability * 100).toFixed(1)}%</div>`;
#                 html += `<div class="metric"><strong>Risk Level:</strong> ${data.risk_level.replace('_', ' ').toUpperCase()}</div>`;
                
#                 if (data.explanation && data.explanation.length > 0) {
#                     html += '<div class="explanation">';
#                     html += '<strong>Key Indicators:</strong> ';
#                     html += data.explanation.map(e => e.word).join(', ');
#                     html += '</div>';
#                 }
                
#                 resultDiv.innerHTML = html;
                
#             } catch (error) {
#                 loader.classList.remove('show');
#                 btn.disabled = false;
#                 alert('Error analyzing text: ' + error.message);
#             }
#         }
        
#         async function scorePHQ9() {
#             const responses = [];
#             let allAnswered = true;
            
#             for (let i = 0; i < 9; i++) {
#                 const selected = document.querySelector(`input[name="q${i}"]:checked`);
#                 if (!selected) {
#                     allAnswered = false;
#                     break;
#                 }
#                 responses.push(parseInt(selected.value));
#             }
            
#             if (!allAnswered) {
#                 alert('Please answer all 9 questions');
#                 return;
#             }
            
#             try {
#                 const response = await fetch('/api/phq9', {
#                     method: 'POST',
#                     headers: {'Content-Type': 'application/json'},
#                     body: JSON.stringify({responses: responses})
#                 });
                
#                 const data = await response.json();
                
#                 // Show crisis card if Q9 > 0
#                 const crisisCard = document.getElementById('crisisCard');
#                 if (data.emergency_flag) {
#                     crisisCard.classList.add('show');
#                     window.scrollTo({top: 0, behavior: 'smooth'});
#                 }
                
#                 // Display results
#                 const resultDiv = document.getElementById('phq9Result');
#                 let riskClass = 'low';
#                 if (data.total_score >= 15) riskClass = 'high';
#                 else if (data.total_score >= 10) riskClass = 'moderate';
                
#                 resultDiv.className = 'result show ' + riskClass;
                
#                 let html = '<h3>PHQ-9 Results</h3>';
#                 html += `<div class="metric"><strong>Total Score:</strong> ${data.total_score} / 27</div>`;
#                 html += `<div class="metric"><strong>Severity:</strong> ${data.severity.replace('_', ' ').toUpperCase()}</div>`;
                
#                 if (data.emergency_flag) {
#                     html += '<div class="metric" style="color: #dc3545;"><strong>⚠️ Emergency Flag:</strong> Please seek immediate help</div>';
#                 }
                
#                 html += '<div class="explanation">';
#                 html += '<strong>Interpretation:</strong><br>';
#                 if (data.total_score <= 4) html += 'Minimal or no depression';
#                 else if (data.total_score <= 9) html += 'Mild depression';
#                 else if (data.total_score <= 14) html += 'Moderate depression';
#                 else if (data.total_score <= 19) html += 'Moderately severe depression';
#                 else html += 'Severe depression';
#                 html += '<br><br><em>Consider consulting a mental health professional for proper evaluation.</em>';
#                 html += '</div>';
                
#                 resultDiv.innerHTML = html;
                
#             } catch (error) {
#                 alert('Error scoring PHQ-9: ' + error.message);
#             }
#         }
#     </script>
# </body>
# </html>
# '''

# @app.route('/')
# def index():
#     """Serve the web interface"""
#     return render_template_string(HTML_TEMPLATE)

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     """API endpoint for text prediction"""
#     try:
#         init_model()
        
#         data = request.get_json()
#         text = data.get('text', '')
#         return_explanation = data.get('return_explanation', False)
        
#         if not text:
#             return jsonify({'error': 'No text provided'}), 400
        
#         result = detector.predict(text, return_explanation=return_explanation)
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/phq9', methods=['POST'])
# def phq9_score():
#     """API endpoint for PHQ-9 scoring"""
#     try:
#         data = request.get_json()
#         responses = data.get('responses', [])
        
#         if len(responses) != 9:
#             return jsonify({'error': 'PHQ-9 requires exactly 9 responses'}), 400
        
#         result = PHQ9Handler.score_phq9(responses)
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/health', methods=['GET'])
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'model_type': MODEL_TYPE,
#         'model_loaded': detector is not None
#     })

# if __name__ == '__main__':
#     print("\n" + "="*60)
#     print("Starting Mental Health Detection API")
#     print("="*60)
#     print(f"Model type: {MODEL_TYPE}")
#     print("Initializing model...")
    
#     try:
#         init_model()
#         print("\n✓ Model loaded successfully!")
#         print("\nServer starting on http://localhost:5000")
#         print("Open your browser and go to: http://localhost:5000")
#         print("\nPress Ctrl+C to stop the server")
#         print("="*60 + "\n")
        
#         app.run(debug=True, host='0.0.0.0', port=5000)
#     except Exception as e:
#         print(f"\n✗ Error: {e}")
#         print("\nMake sure you have trained a model first!")
#         print("Run: python src/train_baseline.py")