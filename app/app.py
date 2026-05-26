"""
=============================================================
  War Conflict Prediction Dashboard — app.py  (v2)
=============================================================
Flask app com 4 páginas:
  /map        → Mapa-múndi com seletor de variante de modelo
  /table      → Tabela de países com seletor de variante
  /model      → Arquitetura, SHAP e resultados por variante
  /evaluation → Comparação de todos os modelos por variante

Como rodar:
  cd app && python app.py
  (garanta que model/model.py foi executado antes)
"""

import os, json
from flask import Flask, render_template, jsonify

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")

app = Flask(__name__)

VARIANT_LABELS = {
    "base":          "Modelo Base",
    "sem_lag1":      "Sem Conflito Anterior",
    "sem_historico": "Conflitos Novos",
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_json(filename, default=None):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_predictions_all():
    return load_json("predictions_all.json", {"variants": {}, "prediction_year": None})

def get_all_results():
    return load_json("all_results.json", {})

def get_model_config():
    return load_json("model_config.json", {})

def get_shap_importances():
    return load_json("shap_importances.json", {})

def get_prediction_year():
    return get_predictions_all().get("prediction_year")

def normalize_all_results(raw):
    """Garante formato {variant: {model: metrics}}.
    Se for legado {model: metrics}, envolve em {"base": ...}."""
    if not raw:
        return {}
    first_val = next(iter(raw.values()), {})
    if isinstance(first_val, dict):
        inner = next(iter(first_val.values()), None)
        if isinstance(inner, dict) and "auc_roc" in inner:
            return raw
    return {"base": raw}

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
    "ZWE":"Zimbabwe",
}
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

def iso2_to_flag(iso2):
    if not iso2 or len(iso2) != 2:
        return "🏳"
    return "".join(chr(ord(c) + 127397) for c in iso2.upper())

def build_countries_for_predictions(predictions):
    countries = []
    for iso3, prob in predictions.items():
        iso2 = ISO3_TO_ISO2.get(iso3, "")
        countries.append({
            "iso3": iso3, "iso2": iso2,
            "flag": iso2_to_flag(iso2),
            "name": ISO3_NAMES.get(iso3, iso3),
            "prob": round(float(prob), 4),
            "pct":  round(float(prob) * 100, 1),
        })
    countries.sort(key=lambda x: x["prob"], reverse=True)
    return countries

def get_all_variants_countries():
    preds_all = get_predictions_all()
    result = {}
    for vkey, preds in preds_all.get("variants", {}).items():
        result[vkey] = build_countries_for_predictions(preds)
    if not result:
        old = load_json("predictions.json", {})
        result["base"] = build_countries_for_predictions(old)
    return result

def model_is_trained():
    return (
        os.path.exists(os.path.join(MODEL_DIR, "predictions_all.json")) or
        os.path.exists(os.path.join(MODEL_DIR, "predictions.json"))
    )

# ══════════════════════════════════════════════════════════════════════════════
# ROTAS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return map_page()

@app.route("/map")
def map_page():
    trained         = model_is_trained()
    all_vc          = get_all_variants_countries()
    prediction_year = get_prediction_year()

    all_variants_map = {
        vkey: [{"iso3": c["iso3"], "name": c["name"],
                "prob": c["prob"],  "pct":  c["pct"]}
               for c in countries]
        for vkey, countries in all_vc.items()
    }
    return render_template("map.html",
        all_variants_map_json = json.dumps(all_variants_map),
        variant_labels_json   = json.dumps(VARIANT_LABELS),
        prediction_year       = prediction_year,
        trained=trained, active="map",
    )

@app.route("/table")
def table_page():
    trained         = model_is_trained()
    all_vc          = get_all_variants_countries()
    prediction_year = get_prediction_year()

    return render_template("table.html",
        all_variants_countries_json = json.dumps(all_vc),
        variant_labels_json         = json.dumps(VARIANT_LABELS),
        prediction_year             = prediction_year,
        trained=trained, active="table",
        countries=all_vc.get("base", []),  # para server-side summary cards
    )

@app.route("/model")
def model_page():
    trained         = model_is_trained()
    all_configs     = get_model_config()
    shap_data       = get_shap_importances()
    prediction_year = get_prediction_year()
    all_results     = normalize_all_results(get_all_results())

    all_nn_info = {
        vkey: vresults.get("NeuralNetwork", {})
        for vkey, vresults in all_results.items()
    }
    base_config  = all_configs.get("base", all_configs) if all_configs else {}
    base_nn_info = all_nn_info.get("base", {})

    return render_template("model.html",
        all_configs_json    = json.dumps(all_configs),
        all_nn_info_json    = json.dumps(all_nn_info),
        shap_data_json      = json.dumps(shap_data),
        variant_labels_json = json.dumps(VARIANT_LABELS),
        prediction_year     = prediction_year,
        trained=trained, active="model",
        config=base_config, nn_info=base_nn_info,
        history_json=json.dumps(base_nn_info.get("history", {})),
    )

@app.route("/evaluation")
def evaluation_page():
    trained         = model_is_trained()
    prediction_year = get_prediction_year()
    all_results     = normalize_all_results(get_all_results())

    MODEL_DISPLAY = {
        "LogisticRegression":         "Logistic Regression",
        "DecisionTreeClassifier":     "Decision Tree",
        "RandomForestClassifier":     "Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting",
        "NeuralNetwork":              "Neural Network",
        "Logistic Regression":        "Logistic Regression",
        "Decision Tree":              "Decision Tree",
        "Random Forest":              "Random Forest",
        "Gradient Boosting":          "Gradient Boosting",
        "Neural Network":             "Neural Network",
    }
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]

    chart_data_by_variant = {}
    for vkey, vresults in all_results.items():
        rows = []
        for name, m in vresults.items():
            rows.append({
                "name":   MODEL_DISPLAY.get(name, name),
                "type":   m.get("type", "classical"),
                "split":  m.get("split", "temporal (treino ≤ 2010 / teste > 2010)"),
                **{k: round(float(m.get(k, 0)), 4) for k in metrics},
            })
        chart_data_by_variant[vkey] = sorted(rows, key=lambda x: x["auc_roc"], reverse=True)

    base_results = all_results.get("base", {})

    return render_template("evaluation.html",
        all_results_json           = json.dumps(all_results),
        chart_data_by_variant_json = json.dumps(chart_data_by_variant),
        variant_labels_json        = json.dumps(VARIANT_LABELS),
        prediction_year            = prediction_year,
        trained=trained, active="evaluation",
        all_results=base_results,
        chart_data_json=json.dumps(chart_data_by_variant.get("base", [])),
        metrics=metrics,
    )

# ── API ───────────────────────────────────────────────────────────────────────
@app.route("/api/predictions")
def api_predictions():
    return jsonify(get_predictions_all())

@app.route("/api/results")
def api_results():
    return jsonify(normalize_all_results(get_all_results()))

@app.route("/api/config")
def api_config():
    return jsonify(get_model_config())

@app.route("/api/shap")
def api_shap():
    return jsonify(get_shap_importances())

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not model_is_trained():
        print("\n⚠️  AVISO: modelo não treinado.")
        print("   Execute: cd ../model && python model.py\n")
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)