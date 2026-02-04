from flask import Flask, jsonify, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc": DATA_DIR / "btc.csv",
}

# Pesi "Pigro" (modifica qui se vuoi)
WEIGHTS = {"ls80": 0.80, "gold": 0.10, "btc": 0.10}


def load_series(asset: str) -> pd.Series:
    """
    Carica un CSV separato da ';' e restituisce una serie prezzi con indice datetime.
    Gestisce:
    - btc.csv: Data;Close;;;;;
    - gold/ls80: Date;Close
    """
    path = FILES[asset]
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    # Legge solo le prime 2 colonne (le altre in btc sono vuote)
    df = pd.read_csv(path, sep=";", usecols=[0, 1], engine="python")

    # Normalizza nomi colonne
    df.columns = ["date", "close"]

    # Pulisce e converte
    df["date"] = (
        df["date"]
        .astype(str)
        .str.strip()
        .str.replace(".", "/", regex=False)  # BTC usa i punti
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date")
    s = df.set_index("date")["close"]
    s.name = asset
    return s


def to_frequency(s: pd.Series, freq: str) -> pd.Series:
    """
    freq:
      - "daily"   = serie giornaliera (così com'è)
      - "monthly" = ultimo valore di ogni mese (più leggero per Chart.js)
    """
    if freq == "daily":
        return s
    # monthly: ultimo valore del mese
    return s.resample("M").last().dropna()


def series_to_payload(s: pd.Series, monthly_labels: bool) -> dict:
    if monthly_labels:
        labels = [d.strftime("%Y-%m") for d in s.index]
    else:
        labels = [d.strftime("%Y-%m-%d") for d in s.index]
    values = [float(v) for v in s.values]
    return {"labels": labels, "values": values}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset", "portfolio").lower()
    freq = request.args.get("freq", "monthly").lower()

    if freq not in ("monthly", "daily"):
        return jsonify({"error": "freq deve essere 'monthly' o 'daily'"}), 400

    try:
        if asset in ("ls80", "gold", "btc"):
            s = load_series(asset)
            s = to_frequency(s, freq)
            return jsonify(series_to_payload(s, monthly_labels=(freq == "monthly")))

        if asset == "portfolio":
            s_ls80 = load_series("ls80")
            s_gold = load_series("gold")
            s_btc = load_series("btc")

            # Allinea sulle date comuni (inner join)
            df = pd.concat([s_ls80, s_gold, s_btc], axis=1).dropna()

            # Indice 100 al primo valore: portfolio = somma pesata dei prezzi normalizzati
            base = df.iloc[0]
            norm = df / base * 100.0
            port = (
                norm["ls80"] * WEIGHTS["ls80"]
                + norm["gold"] * WEIGHTS["gold"]
                + norm["btc"] * WEIGHTS["btc"]
            )
            port.name = "portfolio"

            port = to_frequency(port, freq)

            return jsonify(series_to_payload(port, monthly_labels=(freq == "monthly")))

        return jsonify({"error": "asset deve essere ls80, gold, btc, oppure portfolio"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
