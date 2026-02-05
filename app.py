from flask import Flask, jsonify, request
import pandas as pd
import os

app = Flask(__name__)

DATA_DIR = "data"   # cartella dove stanno i CSV


# ---------- Utility ----------

def read_asset(asset: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{asset}.csv")

    if not os.path.exists(path):
        raise ValueError(f"CSV {asset} non trovato")

    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"CSV {asset} deve avere almeno 2 colonne (data, valore)")

    # usa solo le prime due colonne
    df = df.iloc[:, :2]
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna()

    if df.empty:
        raise ValueError(f"CSV {asset} non contiene dati validi")

    df = df.sort_values("date").set_index("date")
    return df


def resample_series(df: pd.DataFrame, freq: str) -> pd.Series:
    rule = {
        "monthly": "ME",
        "quarterly": "QE",
        "yearly": "YE"
    }.get(freq, "ME")

    return df["value"].resample(rule).last()


def normalize_100(series: pd.Series) -> list:
    base = series.iloc[0]
    return (series / base * 100).round(2).tolist()


# ---------- API ----------

@app.route("/")
def home():
    return "gloob-minimal-chartjs API is running"


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset")
    freq = request.args.get("freq", "monthly")

    try:
        s = resample_series(read_asset(asset), freq)

        return jsonify({
            "asset": asset,
            "base_date": s.index[0].strftime("%Y-%m-%d"),
            "freq": freq,
            "labels": s.index.strftime("%Y-%m").tolist(),
            "points": len(s),
            "values": normalize_100(s)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/combined")
def api_combined():
    freq = request.args.get("freq", "monthly")

    try:
        benchmark = normalize_100(
            resample_series(read_asset("ls80"), freq)
        )
        portfolio = normalize_100(
            resample_series(read_asset("portfolio"), freq)
        )

        labels = (
            resample_series(read_asset("ls80"), freq)
            .index.strftime("%Y-%m")
            .tolist()
        )

        return jsonify({
            "base_date": labels[0] + "-01",
            "freq": freq,
            "labels": labels,
            "points": len(labels),
            "series": {
                "benchmark": benchmark,
                "portfolio": portfolio
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Entrypoint ----------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
