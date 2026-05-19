from flask import Flask, request, jsonify
from gemini_rag import rag_query
import os

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "question field required"}), 400

    question = data["question"]
    top_k = data.get("top_k", 3)

    results = rag_query(question, top_k=top_k)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)