from flask import Flask, request, render_template, jsonify
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load SHL catalog JSON
with open("shl_product_catalog_sample.json", "r") as f:
    catalog = json.load(f)

# Prepare embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = [item["Summary"] for item in catalog]
desc_embeddings = model.encode(descriptions, convert_to_tensor=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, desc_embeddings)[0]

        top_indices = np.argsort(-scores)[:3]  # Top 3 matches
        recommendations = [
            {
                "Assessment Name": catalog[i]["Assessment Name"],
                "Summary": catalog[i]["Summary"],
                "Duration": catalog[i].get("Duration", "Unknown"),
                "URL": catalog[i]["URL"]
            }
            for i in top_indices
        ]

        return render_template("index.html", results=recommendations, query=query)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
