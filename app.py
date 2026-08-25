import pickle
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from keras.models import load_model

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Model Initialization
# ---------------------------------------------------------------------------
# Attempts to load the trained RNN model via pickle or Keras load_model
try:
    with open("RNN.pkl", "rb") as f:
        model = pickle.load(f)
except Exception:
    model = load_model("RNN.pkl")

# Hyper-parameters matching your model's input layer specifications
MAX_LEN = 50  # Fixed sequence length expected by the model input [null, 50]
VOCAB_SIZE = 5000  # Vocabulary bound for the Embedding layer


# ---------------------------------------------------------------------------
# 2. Text Preprocessing Pipeline
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> np.ndarray:
    """Preprocesses raw input text into a padded integer tensor of shape (1, 50).

    Replace the placeholder sequence mapping below with your saved Keras
    Tokenizer (tokenizer.texts_to_sequences) if one was saved during training.
    """
    words = text.lower().split()

    # Map words to token IDs bounded by VOCAB_SIZE
    sequence = [abs(hash(w)) % (VOCAB_SIZE - 1) + 1 for w in words]

    # Pre-pad or truncate sequence to MAX_LEN (50)
    if len(sequence) < MAX_LEN:
        padded_sequence = [0] * (MAX_LEN - len(sequence)) + sequence
    else:
        padded_sequence = sequence[-MAX_LEN:]

    return np.array([padded_sequence], dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. HTML Interface Template (Embedded)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RNN Model Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f7f6; margin: 0; padding: 40px; }
        .container { max-width: 600px; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 0 auto; }
        h2 { color: #333; margin-top: 0; }
        textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        button { background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; display: none; }
        .success { background-color: #e2f0d9; border: 1px solid #b7ddb0; color: #276a20; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h2>RNN Text Classification</h2>
        <form id="predictionForm">
            <textarea id="inputText" placeholder="Enter text to classify..." required></textarea><br>
            <button type="submit">Predict</button>
        </form>
        <div id="resultBox" class="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const text = document.getElementById('inputText').value;
            const resultBox = document.getElementById('resultBox');

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const result = await response.json();
            resultBox.style.display = 'block';

            if (result.status === 'success') {
                resultBox.className = 'result success';
                resultBox.innerHTML = `<strong>Label:</strong> ${result.label}<br><strong>Probability:</strong> ${result.probability}`;
            } else {
                resultBox.className = 'result error';
                resultBox.innerHTML = `<strong>Error:</strong> ${result.message}`;
            }
        });
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 4. Flask Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Renders the embedded web interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """Processes input text and returns JSON predictions."""
    try:
        # Accept JSON payloads or standard form inputs
        data = request.get_json(silent=True)
        if data and "text" in data:
            user_input = data["text"]
        else:
            user_input = request.form.get("text", "")

        if not user_input.strip():
            return (
                jsonify(
                    {"status": "error", "message": "Input text cannot be empty."}
                ),
                400,
            )

        # Preprocess text to tensor shape (1, 50)
        processed_input = preprocess_text(user_input)

        # Run inference using the loaded model
        prediction_prob = float(model.predict(processed_input, verbose=0)[0][0])
        prediction_label = "Positive" if prediction_prob >= 0.5 else "Negative"

        return jsonify(
            {
                "status": "success",
                "input_text": user_input,
                "probability": round(prediction_prob, 4),
                "label": prediction_label,
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# 5. Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
