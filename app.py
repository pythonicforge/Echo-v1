from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = Flask(__name__)

# Load the model and data
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
facts = [
    "Hardik graduated from high school.",
    "Hardik is interested in AI and ML.",
    "Hardik's favorite online resources include freeCodeCamp and Coursera.",
    "Hardik is based in Kolkata, India.",
    "Hardik likes creating interactive and visually appealing websites."
]

embeddings = model.encode(facts)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query")
    query_embedding = model.encode(user_query)

    # Find the closest matching fact
    distances = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    closest_idx = np.argmax(distances)

    response = facts[closest_idx]
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
