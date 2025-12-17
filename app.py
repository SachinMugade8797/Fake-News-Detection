import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("news_text")

    if not text:
        return render_template("index.html", prediction="No text received")

    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]

    # Convert numeric output to label
    if pred == 1:
        result = "Real News"
    else:
        result = "Fake News"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run()

