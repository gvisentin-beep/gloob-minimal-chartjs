from flask import Flask, jsonify, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

DATA_DIR = Path("data")


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def _read_asset(asset: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}.csv"

    if not path.exists():
        raise ValueError(f"Asset '{asset}' non trovato")

    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError(f"CSV {asset} deve avere almeno 2 colonne")

    df = df.iloc[:, :2]
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna()
    df = df.sort_values("date")
    df = df.set_index("date")

    return df


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "monthly":
        rule = "ME"   # Month End (pandas 2.x)
    elif freq == "daily":
        rule = "D"
    else:
        raise ValueError("freq deve essere 'daily' o 'monthly'")

    return df.resample(rule).last().dropna()


def _normalize_100(df: pd.DataFrame) -> pd.Series:
    base = df.iloc[0, 0]
    return (df.iloc[:, 0] / base * 100).round(2)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():
    return "gloob-minimal-chartjs API is running"


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset")
    freq = request.args.get("freq", "monthly")

    if not asset:
        return jsonify({"error": "asset mancante"}), 400

    try:
        df = _read_asset(asset)
        df = _resample(df, freq)
        series = _normalize_100(df)

        return jsonify({
            "asset": asset,
            "freq": freq,
            "labels": series.index.strftime("%Y-%m").tolist(),
            "values": series.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/combined")
def api_combined():
    freq = request.args.get("freq", "monthly")

    try:
        benchmark = _normalize_100(
            _resample(_read_asset("ls80"), freq)
        )
        portfolio = _normalize_100(
            _resample(_read_asset("portfolio"), freq)
        )

        labels = benchmark.index.strftime("%Y-%m").tolist()

        return jsonify({
            "base_date": labels[0],
            "freq": freq,
            "labels": labels,
            "points": len(labels),
            "series": {
                "benchmark": benchmark.tolist(),
                "portfolio": portfolio.tolist()
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------------------------------
# Render / Gunicorn entrypoint
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
