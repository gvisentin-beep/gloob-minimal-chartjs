from flask import Flask, jsonify, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc":  DATA_DIR / "btc.csv",
}

# Pesi del "Pigro" (modifica qui se vuoi)
WEIGHTS = {"ls80": 0.80, "gold": 0.10, "btc": 0.10}


def _read_asset(asset: str) -> pd.Series:
    """
    Legge CSV separato da ';' e ritorna una Serie prezzi con indice datetime.
    Gestisce formati data:
      - btc.csv:  21.01.2026 (dd.mm.yyyy)
      - gold/ls80: 21/01/2026 (dd/mm/yyyy)
    Colonne attese: Date/Close oppure Data/Close
    """
    path = FILES[asset]
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    # btc ha molte colonne vuote: prendiamo solo le prime 2
    df = pd.read_csv(path, sep=";", usecols=[0, 1], engine="python")
    df.columns = ["date", "close"]

    # pulizia
    df["date"] = df["date"].astype(str).str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # parsing date robusto (prima prova dd/mm/yyyy, poi dd.mm.yyyy)
    d1 = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, format=None)
    # se restano NaT, riprova sostituendo '.' con '/'
    mask_nat = d1.isna()
    if mask_nat.any():
        d2 = pd.to_datetime(df.loc[mask_nat, "date"].str.replace(".", "/", regex=False),
                            errors="coerce", dayfirst=True, format=None)
        d1.loc[mask_nat] = d2

    df["date"] = d1
    df = df.dropna(subset=["date", "close"])

    # spesso sono in ordine decrescente: riordina
    df = df.sort_values("date")

    s = df.set_index("date")["close"]
    s = s[~s.index.duplicated(keep="last")]
    return s


def _resample(s: pd.Series, freq: str) -> pd.Series:
    freq = (freq or "daily").lower()

    if freq == "daily":
        return s

    if freq == "weekly":
        # ultimo valore della settimana (venerdì)
        return s.resample("W-FRI").last().dropna()

    if freq == "monthly":
        # ultimo valore del mese
        return s.resample("M").last().dropna()

    # fallback
    return s


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    asset = request.args.get("asset", "ls80").lower()
    freq = request.args.get("freq", "monthly").lower()

    if asset not in FILES:
        return jsonify({"error": f"asset non valido: {asset}"}), 400

    s = _read_asset(asset)
    s = _resample(s, freq)

    # normalizza a 100
    base = float(s.iloc[0])
    idx = (s / base) * 100.0

    labels = [d.strftime("%Y-%m-%d") for d in idx.index]
    values = [round(float(v), 2) for v in idx.values]

    return jsonify({
        "asset": asset,
        "freq": freq,
        "base_date": idx.index[0].strftime("%Y-%m-%d"),
        "points": len(values),
        "labels": labels,
        "values": values,
    })


@app.route("/api/combined")
def api_combined():
    """
    Ritorna due serie NORMALIZZATE a 100:
      - B: Portafoglio (LS80 80% + Gold 10% + BTC 10%)
      - C: LS80 (benchmark semplice)
    """
    freq = request.args.get("freq", "monthly").lower()

    # carica e resample
    s_ls80 = _resample(_read_asset("ls80"), freq)
    s_gold = _resample(_read_asset("gold"), freq)
    s_btc  = _resample(_read_asset("btc"),  freq)

    # allinea sulle date comuni
    df = pd.concat(
        {"ls80": s_ls80, "gold": s_gold, "btc": s_btc},
        axis=1
    ).dropna()

    if df.empty or len(df) < 2:
        return jsonify({"error": "Dati insufficienti dopo l'allineamento (date comuni troppo poche)."}), 400

    base_date = df.index[0]

    # normalizza ogni asset a 100
    norm = (df / df.iloc[0]) * 100.0

    # Portafoglio B (somma pesata degli indici normalizzati)
    w = WEIGHTS
    portfolio = (
        norm["ls80"] * w["ls80"] +
        norm["gold"] * w["gold"] +
        norm["btc"]  * w["btc"]
    )

    # Serie C: LS80 normalizzato
    series_c = norm["ls80"]

    labels = [d.strftime("%Y-%m-%d") for d in df.index]
    out_b = [round(float(v), 2) for v in portfolio.values]
    out_c = [round(float(v), 2) for v in series_c.values]

    return jsonify({
        "freq": freq,
        "base_date": base_date.strftime("%Y-%m-%d"),
        "points": len(labels),
        "weights": w,
        "labels": labels,
        "portfolio": out_b,   # Grafico B
        "ls80": out_c         # Grafico C
    })


if __name__ == "__main__":
    app.run(debug=True)
