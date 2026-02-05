from flask import Flask, jsonify, render_template, request
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# File attesi dentro /data
FILES = {
    "ls80": DATA_DIR / "ls80.csv",
    "gold": DATA_DIR / "gold.csv",
    "btc":  DATA_DIR / "btc.csv",
}

# Pesi portafoglio "B" (modifica qui se vuoi)
WEIGHTS = {"ls80": 0.80, "gold": 0.10, "btc": 0.10}

# Frequenze pandas (NOTA: 'M' non è più accettato -> usare 'ME')
FREQ_MAP = {
    "daily": "D",
    "day": "D",
    "weekly": "W-FRI",
    "week": "W-FRI",
    "monthly": "ME",     # Month End
    "month": "ME",
    "quarterly": "QE",   # Quarter End
    "quarter": "QE",
    "yearly": "YE",      # Year End
    "year": "YE",
}

def _read_asset(asset: str) -> pd.Series:
    """
    Legge un CSV in modo robusto e restituisce una Series indicizzata per data.
    Attende almeno 2 colonne (data, valore). Accetta separatori , o ;.
    """
    if asset not in FILES:
        raise ValueError(f"Asset non valido: {asset}. Validi: {list(FILES.keys())}")

    path = FILES[asset]
    if not path.exists():
        raise ValueError(f"File mancante: {path}")

    # 1) Prova autodetect separatore
    df = None
    errors = []
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        errors.append(f"auto-sep: {e}")

    # 2) Fallback espliciti
    if df is None or df.shape[1] < 2:
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep)
                if df.shape[1] >= 2:
                    break
            except Exception as e:
                errors.append(f"sep={sep}: {e}")

    if df is None or df.shape[1] < 2:
        raise ValueError(
            f"CSV '{asset}' deve avere almeno 2 colonne (data, valore). "
            f"Letto: {0 if df is None else df.shape[1]} colonne. "
            f"Dettagli: {errors}"
        )

    # Cerca colonne “sensate”, altrimenti usa le prime due
    cols = [c.strip().lower() for c in df.columns.astype(str)]
    df.columns = cols

    # Colonna data
    date_col = None
    for c in cols:
        if c in ("date", "data", "datetime", "time"):
            date_col = c
            break
    if date_col is None:
        date_col = cols[0]

    # Colonna valore
    value_col = None
    for c in cols:
        if c in ("value", "valore", "close", "price", "prezzo", "index", "nav"):
            value_col = c
            break
    if value_col is None:
        value_col = cols[1]

    # Parsing date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    # Parsing numeri (gestisce "1.234,56" -> "1234.56")
    s = df[value_col].astype(str).str.replace(" ", "", regex=False)
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    df[value_col] = pd.to_numeric(s, errors="coerce")

    df = df[[date_col, value_col]].dropna()
    if df.empty:
        raise ValueError(f"CSV '{asset}' letto ma senza righe valide (date/valori non parsati).")

    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col], keep="last")

    series = pd.Series(df[value_col].values, index=df[date_col], name=asset).sort_index()
    return series


def _resample(series: pd.Series, freq_key: str) -> pd.Series:
    """
    Resampling; se daily -> lascia com’è; altrimenti usa l’ultimo valore del periodo.
    """
    fk = (freq_key or "monthly").strip().lower()
    if fk not in FREQ_MAP:
        fk = "monthly"
    rule = FREQ_MAP[fk]

    if rule == "D":
        return series.dropna()

    # ultimo valore del periodo
    return series.resample(rule).last().dropna()


def _normalize_to_100(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    base = float(series.iloc[0])
    if base == 0:
        raise ValueError("Valore base = 0, impossibile normalizzare.")
    return (series / base) * 100.0


def _label_format(freq_key: str) -> str:
    fk = (freq_key or "monthly").strip().lower()
    if fk in ("daily", "day"):
        return "%Y-%m-%d"
    if fk in ("weekly", "week"):
        return "%Y-%m-%d"
    if fk in ("yearly", "year"):
        return "%Y"
    # monthly / quarterly
    return "%Y-%m"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    """
    Esempio:
      /api/data?asset=ls80&freq=monthly
    Ritorna serie NORMALIZZATA a 100.
    """
    try:
        asset = request.args.get("asset", "ls80").lower()
        freq = request.args.get("freq", "monthly").lower()

        s = _read_asset(asset)
        s = _resample(s, freq)
        s = _normalize_to_100(s)

        fmt = _label_format(freq)
        labels = [d.strftime(fmt) for d in s.index]
        values = [round(float(v), 2) for v in s.values]

        return jsonify({
            "asset": asset,
            "freq": freq,
            "base_date": s.index[0].strftime("%Y-%m-%d") if len(s) else None,
            "points": len(values),
            "labels": labels,
            "values": values,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/combined")
def api_combined():
    """
    Ritorna due serie NORMALIZZATE a 100:
      - B: Portafoglio (ls80*0.8 + gold*0.1 + btc*0.1)
      - C: ls80 (benchmark)
    Esempio:
      /api/combined?freq=monthly
    """
    try:
        freq = request.args.get("freq", "monthly").lower()

        s_ls80 = _resample(_read_asset("ls80"), freq)
        s_gold = _resample(_read_asset("gold"), freq)
        s_btc  = _resample(_read_asset("btc"),  freq)

        # allinea alle date comuni
        df = pd.concat(
            {"ls80": s_ls80, "gold": s_gold, "btc": s_btc},
            axis=1
        ).dropna()

        if df.empty:
            raise ValueError("Nessuna data comune tra ls80/gold/btc dopo resample. Controlla i CSV (date).")

        # Portafoglio B (valori “grezzi”)
        port = (
            df["ls80"] * WEIGHTS["ls80"] +
            df["gold"] * WEIGHTS["gold"] +
            df["btc"]  * WEIGHTS["btc"]
        )

        # normalizza entrambi a 100 sulla stessa base (prima data comune)
        port_n = _normalize_to_100(port)
        ls80_n = _normalize_to_100(df["ls80"])

        fmt = _label_format(freq)
        labels = [d.strftime(fmt) for d in df.index]

        return jsonify({
            "freq": freq,
            "base_date": df.index[0].strftime("%Y-%m-%d"),
            "points": int(len(df)),
            "weights": WEIGHTS,
            "labels": labels,
            "portfolio": [round(float(v), 2) for v in port_n.values],  # B
            "ls80":      [round(float(v), 2) for v in ls80_n.values],  # C
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    app.run(debug=True)
