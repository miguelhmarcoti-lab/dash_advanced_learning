"""
=============================================================
  War Conflict Prediction Dashboard — app.py
=============================================================
Flask app com 4 páginas:
  /map        → Mapa-múndi com probabilidades
  /table      → Tabela de países com bandeiras
  /model      → Arquitetura e resultados da Rede Neural
  /evaluation → Comparação de todos os modelos

Como rodar:
  cd app
  pip install -r ../requirements.txt
  python app.py

Garanta que model/model.py foi executado antes:
  cd ../model && python model.py
"""

import os
import json
import sys

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")
DATA_DIR  = os.path.join(ROOT_DIR, "data")

app = Flask(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — Carregar artefatos do modelo
# ══════════════════════════════════════════════════════════════════════════════

def load_json(filename: str, default=None):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def get_predictions() -> dict:
    return load_json("predictions.json", {})


def get_all_results() -> dict:
    return load_json("all_results.json", {})


def get_baseline_results() -> dict:
    return load_json("baseline_results.json", {})


def get_nn_results() -> dict:
    return load_json("nn_results.json", {})


def get_model_config() -> dict:
    return load_json("model_config.json", {})


def get_nn_info() -> dict:
    results = get_nn_results() or {}
    if "Neural Network" in results:
        return results["Neural Network"]
    for value in results.values():
        if isinstance(value, dict) and value.get("type") == "neural_network":
            return value
    return {}


# ISO3 → nome completo do país
ISO3_NAMES = {
    "AFG":"Afghanistan","AGO":"Angola","ALB":"Albania","ARG":"Argentina",
    "AUS":"Australia","AZE":"Azerbaijan","BDI":"Burundi","BEN":"Benin",
    "BFA":"Burkina Faso","BGD":"Bangladesh","BIH":"Bosnia & Herzegovina",
    "BOL":"Bolivia","BRN":"Brunei","CAF":"Central African Republic",
    "CHN":"China","CHL":"Chile","CIV":"Côte d'Ivoire","CMR":"Cameroon",
    "COD":"DR Congo","COG":"Congo","COL":"Colombia","COM":"Comoros",
    "CRI":"Costa Rica","CUB":"Cuba","CYP":"Cyprus","DJI":"Djibouti",
    "DZA":"Algeria","DOM":"Dominican Republic","ECU":"Ecuador",
    "EGY":"Egypt","ERI":"Eritrea","ESP":"Spain","ETH":"Ethiopia",
    "FRA":"France","GAB":"Gabon","GBR":"United Kingdom","GEO":"Georgia",
    "GHA":"Ghana","GMB":"Gambia","GNB":"Guinea-Bissau","GRC":"Greece",
    "GRD":"Grenada","GTM":"Guatemala","GIN":"Guinea","HND":"Honduras",
    "HRV":"Croatia","HTI":"Haiti","HUN":"Hungary","IDN":"Indonesia",
    "IND":"India","IRN":"Iran","IRQ":"Iraq","ISR":"Israel",
    "JOR":"Jordan","KEN":"Kenya","KGZ":"Kyrgyzstan","KHM":"Cambodia",
    "KOR":"South Korea","KWT":"Kuwait","LAO":"Laos","LBN":"Lebanon",
    "LBR":"Liberia","LBY":"Libya","LKA":"Sri Lanka","LSO":"Lesotho",
    "MAR":"Morocco","MDG":"Madagascar","MEX":"Mexico","MLI":"Mali",
    "MKD":"North Macedonia","MDA":"Moldova","MMR":"Myanmar","MOZ":"Mozambique",
    "MRT":"Mauritania","MYS":"Malaysia","NER":"Niger","NGA":"Nigeria",
    "NIC":"Nicaragua","NLD":"Netherlands","NPL":"Nepal","OMN":"Oman",
    "PAK":"Pakistan","PAN":"Panama","PER":"Peru","PHL":"Philippines",
    "PNG":"Papua New Guinea","PRK":"North Korea","PRY":"Paraguay",
    "ROU":"Romania","RUS":"Russia","RWA":"Rwanda","SAU":"Saudi Arabia",
    "SDN":"Sudan","SEN":"Senegal","SLE":"Sierra Leone","SLV":"El Salvador",
    "SOM":"Somalia","SRB":"Serbia","SSD":"South Sudan","SUR":"Suriname",
    "SYR":"Syria","TCD":"Chad","TGO":"Togo","THA":"Thailand",
    "TJK":"Tajikistan","TTO":"Trinidad & Tobago","TUN":"Tunisia",
    "TUR":"Turkey","TWN":"Taiwan","TZA":"Tanzania","UGA":"Uganda",
    "UKR":"Ukraine","URY":"Uruguay","USA":"United States","UZB":"Uzbekistan",
    "VEN":"Venezuela","VNM":"Vietnam","YEM":"Yemen","ZAF":"South Africa",
    "ZWE":"Zimbabwe","ALG":"Algeria","BGD":"Bangladesh",
}

# ISO3 → ISO2 para bandeiras emoji
ISO3_TO_ISO2 = {
    "AFG":"AF","AGO":"AO","ALB":"AL","ARG":"AR","AUS":"AU","AZE":"AZ",
    "BDI":"BI","BEN":"BJ","BFA":"BF","BGD":"BD","BIH":"BA","BOL":"BO",
    "BRN":"BN","CAF":"CF","CHN":"CN","CHL":"CL","CIV":"CI","CMR":"CM",
    "COD":"CD","COG":"CG","COL":"CO","COM":"KM","CRI":"CR","CUB":"CU",
    "CYP":"CY","DJI":"DJ","DZA":"DZ","DOM":"DO","ECU":"EC","EGY":"EG",
    "ERI":"ER","ESP":"ES","ETH":"ET","FRA":"FR","GAB":"GA","GBR":"GB",
    "GEO":"GE","GHA":"GH","GMB":"GM","GNB":"GW","GRC":"GR","GRD":"GD",
    "GTM":"GT","GIN":"GN","HND":"HN","HRV":"HR","HTI":"HT","HUN":"HU",
    "IDN":"ID","IND":"IN","IRN":"IR","IRQ":"IQ","ISR":"IL","JOR":"JO",
    "KEN":"KE","KGZ":"KG","KHM":"KH","KOR":"KR","KWT":"KW","LAO":"LA",
    "LBN":"LB","LBR":"LR","LBY":"LY","LKA":"LK","LSO":"LS","MAR":"MA",
    "MDG":"MG","MEX":"MX","MLI":"ML","MKD":"MK","MDA":"MD","MMR":"MM",
    "MOZ":"MZ","MRT":"MR","MYS":"MY","NER":"NE","NGA":"NG","NIC":"NI",
    "NLD":"NL","NPL":"NP","OMN":"OM","PAK":"PK","PAN":"PA","PER":"PE",
    "PHL":"PH","PNG":"PG","PRK":"KP","PRY":"PY","ROU":"RO","RUS":"RU",
    "RWA":"RW","SAU":"SA","SDN":"SD","SEN":"SN","SLE":"SL","SLV":"SV",
    "SOM":"SO","SRB":"RS","SSD":"SS","SUR":"SR","SYR":"SY","TCD":"TD",
    "TGO":"TG","THA":"TH","TJK":"TJ","TTO":"TT","TUN":"TN","TUR":"TR",
    "TWN":"TW","TZA":"TZ","UGA":"UG","UKR":"UA","URY":"UY","USA":"US",
    "UZB":"UZ","VEN":"VE","VNM":"VN","YEM":"YE","ZAF":"ZA","ZWE":"ZW",
}


def iso2_to_flag(iso2: str) -> str:
    """Converte ISO2 em emoji de bandeira."""
    if not iso2 or len(iso2) != 2:
        return "🏳"
    return "".join(chr(ord(c) + 127397) for c in iso2.upper())


def build_countries_list() -> list:
    """Monta lista de países com nome, bandeira e probabilidade."""
    predictions = get_predictions()
    countries = []
    for iso3, prob in predictions.items():
        iso2  = ISO3_TO_ISO2.get(iso3, "")
        flag  = iso2_to_flag(iso2)
        name  = ISO3_NAMES.get(iso3, iso3)
        pct   = round(prob * 100, 1)
        countries.append({
            "iso3" : iso3,
            "iso2" : iso2,
            "flag" : flag,
            "name" : name,
            "prob" : prob,
            "pct"  : pct,
        })
    countries.sort(key=lambda x: x["prob"], reverse=True)
    return countries


def model_is_trained() -> bool:
    return os.path.exists(os.path.join(MODEL_DIR, "predictions.json"))


# ══════════════════════════════════════════════════════════════════════════════
# ROTAS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return map_page()


@app.route("/map")
def map_page():
    predictions = get_predictions()
    countries   = build_countries_list()
    trained     = model_is_trained()

    # Preparar dados para Plotly Choropleth
    map_data = [
        {
            "iso3" : c["iso3"],
            "name" : c["name"],
            "prob" : c["prob"],
            "pct"  : c["pct"],
        }
        for c in countries
    ]
    return render_template(
        "map.html",
        map_data_json=json.dumps(map_data),
        trained=trained,
        active="map",
    )


@app.route("/table")
def table_page():
    countries = build_countries_list()
    trained   = model_is_trained()
    return render_template(
        "table.html",
        countries=countries,
        trained=trained,
        active="table",
    )


@app.route("/model")
def model_page():
    config     = get_model_config()
    trained    = model_is_trained()

    nn_info = get_nn_info()
    history = nn_info.get("history", {})

    return render_template(
        "model.html",
        config=config,
        nn_info=nn_info,
        history_json=json.dumps(history),
        trained=trained,
        active="model",
    )


@app.route("/evaluation")
def evaluation_page():
    all_results = get_all_results()
    trained     = model_is_trained()

    # Separar baseline vs rede neural
    baseline = {k: v for k, v in all_results.items() if v.get("type") == "baseline"}
    nn       = {k: v for k, v in all_results.items() if v.get("type") == "neural_network"}

    # Todos os modelos para o radar/bar chart
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    chart_data = []
    for name, m in all_results.items():
        chart_data.append({
            "name"      : name,
            "type"      : m.get("type", "baseline"),
            "accuracy"  : m.get("accuracy", 0),
            "precision" : m.get("precision", 0),
            "recall"    : m.get("recall", 0),
            "f1"        : m.get("f1", 0),
            "auc_roc"   : m.get("auc_roc", 0),
        })
    # Ordenar por AUC-ROC
    chart_data.sort(key=lambda x: x["auc_roc"], reverse=True)

    return render_template(
        "evaluation.html",
        all_results=all_results,
        baseline=baseline,
        nn=nn,
        chart_data_json=json.dumps(chart_data),
        metrics=metrics,
        trained=trained,
        active="evaluation",
    )


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route("/api/predictions")
def api_predictions():
    return jsonify(get_predictions())


@app.route("/api/results")
def api_results():
    return jsonify(get_all_results())


@app.route("/api/config")
def api_config():
    return jsonify(get_model_config())


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not model_is_trained():
        print("\n⚠️  AVISO: model/predictions.json não encontrado.")
        print("   Execute primeiro: cd ../model && python model.py\n")
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
