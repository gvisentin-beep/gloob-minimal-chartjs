from __future__ import annotations

from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    # Dati finti, sempre uguali: solo per testare fetch + JSON
    payload = {
        "labels": [
            "2021-02", "2021-06", "2021-12",
            "2022-06", "2022-12",
            "2023-06", "2023-12",
            "2024-06", "2024-12",
            "2025-06", "2026-01"
        ],
        "values": [100, 110, 105, 120, 115, 130, 140, 160, 170, 180, 195]
    }
    return jsonify(payload)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
