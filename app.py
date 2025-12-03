from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- 1. FLASK SETUP ---
app = Flask(__name__)

# --- 2. MODEL LOADING ---
# Ensure the 'model/' directory contains your saved pytorch_model.bin and tokenizer files
model_path = "model/"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Set model to evaluation mode
    model.eval()
    print("Model and Tokenizer loaded successfully from:", model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    # In a production environment, you might stop the app or serve an error page.

# --- 3. PREDICTION FUNCTION ---

def predict_fake_news(text):
    """
    Tokenizes input text, runs inference on the loaded model, and returns the prediction string.
    
    Based on the training map:
    0 is Mapped to FAKE
    1 is Mapped to REAL
    """
    # 1. Tokenize the combined text
    # Use torch.no_grad() for inference to save memory and computations
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # 2. Run inference
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    # 3. Interpret the prediction
    if prediction == 1:
        return "REAL NEWS"
    else:
        return "FAKE NEWS"

# --- 4. FLASK ROUTES ---

@app.route("/")
def home():
    """Renders the main page with the input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles the POST request from the form, combines headline and body,
    and returns the prediction.
    """
    if request.method == "POST":
        
        # Get the two separate inputs from the HTML form (matching the 'name' attributes)
        headline = request.form.get("headline", "")
        body_snippet = request.form.get("body_snippet", "")

        # CRITICAL: Combine them exactly as trained: Headline [SEP] Body
        input_text = headline + " [SEP] " + body_snippet
        
        print(f"Processing combined input: {input_text[:50]}...")
        
        # Run prediction
        try:
            result = predict_fake_news(input_text)
        except Exception as e:
            result = f"Prediction Error: {e}"
            print(result)
            
        # Pass the result back to the template
        return render_template("index.html", result=result)

if __name__ == "__main__":
    # Ensure you are running this from the directory containing the 'model' folder
    # and the 'templates/index.html' file.
    app.run(debug=True)