"""
=============================================================
  War Conflict Prediction Model — model.py  (v3)
=============================================================
3 variantes:
  - base          : reproduz comportamento original
                    Clássicos: drop conflict (usa log_gdp + lag1 + 5y_sum)
                    NN:        MANTÉM conflict (usa gdp + conflict + lag1 + 5y_sum)
                    → mesma configuração que gerava recall alto na NN
  - sem_lag1      : remove conflict_lag1 e conflict; mantém conflict_5y_sum
  - sem_historico : remove todo histórico de conflito (Conflitos Novos)

SHAP para TODOS os modelos:
  - Clássicos: TreeExplainer (árvores) / LinearExplainer (logística)
  - NN:        DeepExplainer (Keras/TensorFlow)

Saídas em model/:
  - all_results.json        → {variant: {model: métricas}}
  - predictions_all.json    → {variants: {variant: {iso3: prob}}, prediction_year}
  - shap_importances.json   → {variant: {model: {feature: importance}}}
  - model_config.json       → {variant: {features_classical, features_nn, ...}}
  - nn_model_{variant}.keras + scaler_{variant}.joblib
  - {model}_{variant}.joblib

Como rodar:
  pip install shap tensorflow scikit-learn pandas joblib
  cd model && python model.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import shap

from sklearn.base           import clone
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import tensorflow as tf

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(MODEL_DIR, "..", "data", "panel_final.csv")
CUTOFF_YEAR = 2010


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO DAS VARIANTES
# Cada variante define quais colunas dropar para modelos clássicos e para a NN
# separadamente — preservando o comportamento original onde NN base usa 'conflict'
# ══════════════════════════════════════════════════════════════════════════════

VARIANT_CONFIG = {
    "base": {
        "label":       "Modelo Base",
        "description": (
            "Reproduz o modelo original: clássicos sem 'conflict' atual (usa log_gdp + lag1 + 5y_sum); "
            "NN mantém 'conflict' (preditor forte, maior recall)."
        ),
        # Clássicos: sem conflict atual, sem gdp bruto → usa log_gdp
        "classical_drop": ["iso3", "year", "gdp_per_capita", "conflict"],
        # NN: sem gdp bruto, sem conflict atual → usa log_gdp + lag1 + 5y_sum (igual aos clássicos)
        "nn_drop":        ["iso3", "year", "gdp_per_capita", "conflict"],
    },
    "sem_lag1": {
        "label":       "Sem Conflito Anterior",
        "description": (
            "Remove conflict_lag1 e conflict atual; mantém conflict_5y_sum. "
            "Reduz viés de 'guerra já em curso'."
        ),
        "classical_drop": ["iso3", "year", "gdp_per_capita", "conflict", "conflict_lag1"],
        "nn_drop":        ["iso3", "year", "gdp_per_capita", "conflict", "conflict_lag1"],
    },
    "sem_historico": {
        "label":       "Conflitos Novos",
        "description": (
            "Remove todo o histórico de conflito (lag1 + 5y_sum + conflict). "
            "Prevê apenas via fatores estruturais: economia, democracia, gastos militares."
        ),
        "classical_drop": ["iso3", "year", "gdp_per_capita", "conflict", "conflict_lag1", "conflict_5y_sum"],
        "nn_drop":        ["iso3", "year", "gdp_per_capita", "conflict", "conflict_lag1", "conflict_5y_sum"],
    },
}

# ── Modelos clássicos ─────────────────────────────────────────────────────────
CLASSICAL_MODELS = [
    ("LogisticRegression", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(
        min_samples_leaf=20, min_samples_split=30,
        class_weight="balanced", max_depth=5, random_state=42)),
    ("RandomForestClassifier", RandomForestClassifier(
        max_depth=10, min_samples_leaf=20, max_features="sqrt",
        class_weight="balanced", n_estimators=200, n_jobs=-1, random_state=42)),
    ("GradientBoostingClassifier", GradientBoostingClassifier(
        learning_rate=0.01, n_estimators=100, max_depth=3, random_state=42)),
]
TREE_MODELS = {"DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"}

# ── Config da rede neural (mesma arquitetura para todas as variantes) ─────────
NN_CONFIG = {
    "layers": [
        {"units": 128, "activation": "relu", "dropout": 0.3},
        {"units": 2,   "activation": "relu", "dropout": None},
    ],
    "output_activation"      : "sigmoid",
    "optimizer"              : "adam",
    "loss"                   : "binary_crossentropy",
    "epochs"                 : 100,
    "batch_size"             : 32,
    "validation_split"       : 0.2,
    "early_stopping_patience": 10,
    "reduce_lr_patience"     : 5,
    "reduce_lr_factor"       : 0.2,
    "min_lr"                 : 1e-5,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[PRÉ] Limpeza de dados...")
    df = df.drop_duplicates()
    pct = df.isnull().sum() / len(df)
    drop_cols = pct[pct > 0.3].index.tolist()
    if drop_cols:
        print(f"  Removidas por >30% NaN: {drop_cols}")
        df = df.drop(columns=drop_cols)
    for col in df.columns:
        if df[col].dtype in ["int64", "float64", "Int8"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    keep = df.columns[df.nunique() > 1]
    df = df[keep]
    print(f"  Shape final: {df.shape}")
    return df


def build_features(df: pd.DataFrame, variant_key: str,
                   model_type: str = "classical",
                   target: str = "conflict_next_year"):
    """
    Monta X, y respeitando os drop_cols específicos de cada variante e tipo de modelo.
    model_type: 'classical' | 'nn'
    """
    cfg      = VARIANT_CONFIG[variant_key]
    drop_key = "classical_drop" if model_type == "classical" else "nn_drop"
    all_drop = [c for c in cfg[drop_key] if c in df.columns]
    feat     = df.drop(columns=all_drop)
    X        = feat.drop(columns=[target])
    y        = feat[target].astype(int)
    return X, y


def temporal_split(X, y, year_col):
    tr = year_col.values <= CUTOFF_YEAR
    te = year_col.values >  CUTOFF_YEAR
    return (
        X[tr].reset_index(drop=True), X[te].reset_index(drop=True),
        y[tr].reset_index(drop=True), y[te].reset_index(drop=True),
        tr, te,
    )


def metrics_dict(y_test, y_pred, y_prob,
                 variant_key, model_type, feature_names) -> dict:
    return {
        "accuracy" : round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall"   : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1"       : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc"  : round(float(roc_auc_score(y_test, y_prob)), 4),
        "split"    : f"temporal (treino ≤ {CUTOFF_YEAR} / teste > {CUTOFF_YEAR})",
        "type"     : model_type,
        "variant"  : variant_key,
        "features" : feature_names,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. SHAP — MODELOS CLÁSSICOS
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_classical(model, model_name: str,
                           X_train: pd.DataFrame,
                           X_test:  pd.DataFrame,
                           feature_names: list) -> dict:
    """TreeExplainer para árvores, LinearExplainer para LogisticRegression."""
    try:
        if model_name in TREE_MODELS:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test)
            if isinstance(shap_vals, list):
                sv = shap_vals[1]
            elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                sv = shap_vals[:, :, 1] if shap_vals.shape[-1] == 2 else shap_vals[:, 1, :]
            else:
                sv = shap_vals
        else:
            explainer = shap.LinearExplainer(
                model, X_train, feature_perturbation="interventional")
            sv = explainer.shap_values(X_test)
            if isinstance(sv, list):
                sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        return {f: round(float(v), 6) for f, v in zip(feature_names, mean_abs)}
    except Exception as e:
        print(f"    [SHAP clássico aviso] {model_name}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# 3. SHAP — REDE NEURAL (DeepExplainer)
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_nn(nn_model: tf.keras.Model,
                    X_train_scaled: np.ndarray,
                    X_test_scaled:  np.ndarray,
                    feature_names:  list,
                    n_background:   int = 100,
                    n_test:         int = 200) -> dict:
    """
    DeepExplainer para modelos Keras com saída sigmoid binária.
    Usa subconjunto do treino como background e subconjunto do teste para calcular SHAP.
    """
    try:
        print("    Calculando SHAP para NeuralNetwork (DeepExplainer)...", flush=True)
        bg    = X_train_scaled[:n_background]
        Xtest = X_test_scaled[:n_test]

        explainer = shap.DeepExplainer(nn_model, bg)
        shap_vals = explainer.shap_values(Xtest)

        # shap_values pode retornar lista (por classe) ou array único
        if isinstance(shap_vals, list):
            sv = shap_vals[0]         # saída sigmoid: classe positiva
        else:
            sv = shap_vals

        # Algumas versões retornam 3D (n_samples, n_features, 1)
        if sv.ndim == 3:
            sv = sv[:, :, 0]

        mean_abs = np.abs(sv).mean(axis=0)
        result = {f: round(float(v), 6) for f, v in zip(feature_names, mean_abs)}
        print(f"    SHAP NN OK — top feature: {max(result, key=result.get)}")
        return result

    except Exception as e:
        print(f"    [SHAP NN aviso] DeepExplainer falhou: {e}")
        print("    Tentando KernelExplainer (mais lento)...")
        try:
            bg_df     = X_train_scaled[:50]
            Xtest_df  = X_test_scaled[:50]
            predict_fn = lambda x: nn_model.predict(x, verbose=0).ravel()
            explainer  = shap.KernelExplainer(predict_fn, bg_df)
            sv         = explainer.shap_values(Xtest_df, nsamples=100)
            if isinstance(sv, list):
                sv = sv[0]
            mean_abs = np.abs(sv).mean(axis=0)
            return {f: round(float(v), 6) for f, v in zip(feature_names, mean_abs)}
        except Exception as e2:
            print(f"    [SHAP NN aviso] KernelExplainer também falhou: {e2}")
            return {}


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODELOS CLÁSSICOS
# ══════════════════════════════════════════════════════════════════════════════

def train_classical_variant(df_clean: pd.DataFrame, variant_key: str):
    print(f"\n  [Classical | {variant_key}]")
    year_col      = df_clean["year"].reset_index(drop=True)
    X, y          = build_features(df_clean.reset_index(drop=True), variant_key, "classical")
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test, tr_mask, te_mask = temporal_split(X, y, year_col)

    print(f"    Treino: {tr_mask.sum():,} obs | Teste: {te_mask.sum():,} obs")
    print(f"    Features ({len(feature_names)}): {feature_names}")

    # Pesos de amostra para modelos que não suportam class_weight nativamente
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    results  = {}
    shap_out = {}

    for name, proto in CLASSICAL_MODELS:
        m = clone(proto)
        print(f"    → {name}", flush=True)
        # GradientBoostingClassifier não suporta class_weight → passa sample_weight no fit
        if name == "GradientBoostingClassifier":
            m.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1]

        results[name] = {
            **metrics_dict(y_test, y_pred, y_prob, variant_key, "classical", feature_names),
            "model_class": "classical",
        }
        print(f"      AUC={results[name]['auc_roc']}  Recall={results[name]['recall']}  F1={results[name]['f1']}")

        shap_out[name] = compute_shap_classical(m, name, X_train, X_test, feature_names)

        joblib.dump(m, os.path.join(MODEL_DIR, f"{name.lower()}_{variant_key}.joblib"))

    return results, shap_out, X_train, X_test, y_train, y_test, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# 5. REDE NEURAL
# ══════════════════════════════════════════════════════════════════════════════

def build_nn(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(name="war_predictor_nn")
    for i, lc in enumerate(NN_CONFIG["layers"]):
        kw = {"activation": lc["activation"]}
        if i == 0:
            kw["input_shape"] = (input_dim,)
        model.add(tf.keras.layers.Dense(lc["units"], **kw))
        if lc.get("dropout"):
            model.add(tf.keras.layers.Dropout(lc["dropout"]))
    model.add(tf.keras.layers.Dense(1, activation=NN_CONFIG["output_activation"]))
    model.compile(
        optimizer=NN_CONFIG["optimizer"],
        loss=NN_CONFIG["loss"],
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def train_nn_variant(df_clean: pd.DataFrame, variant_key: str):
    print(f"\n  [NN | {variant_key}]")
    year_col      = df_clean["year"].reset_index(drop=True)
    X, y          = build_features(df_clean.reset_index(drop=True), variant_key, "nn")
    feature_names = X.columns.tolist()
    X_train_raw, X_test_raw, y_train, y_test, tr_mask, te_mask = temporal_split(X, y, year_col)

    print(f"    Treino: {tr_mask.sum():,} obs | Teste: {te_mask.sum():,} obs")
    print(f"    Features NN ({len(feature_names)}): {feature_names}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    neg, pos = np.bincount(y_train.values)
    total    = neg + pos
    cw       = {0: (1/neg)*(total/2.0), 1: (1/pos)*(total/2.0)}
    print(f"    Peso classe 0: {cw[0]:.2f} | Peso classe 1: {cw[1]:.2f}")

    nn = build_nn(X_train.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=NN_CONFIG["early_stopping_patience"],
            restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=NN_CONFIG["reduce_lr_factor"],
            patience=NN_CONFIG["reduce_lr_patience"], min_lr=NN_CONFIG["min_lr"]),
    ]
    history = nn.fit(
        X_train, y_train,
        epochs=NN_CONFIG["epochs"],
        batch_size=NN_CONFIG["batch_size"],
        validation_split=NN_CONFIG["validation_split"],
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc, prec, rec, auc = nn.evaluate(X_test, y_test, verbose=0)
    y_prob = nn.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    print(f"    AUC={auc:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

    nn.save(os.path.join(MODEL_DIR, f"nn_model_{variant_key}.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{variant_key}.joblib"))

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # ── SHAP para NN ──────────────────────────────────────────────────────────
    nn_shap = compute_shap_nn(nn, X_train, X_test, feature_names)

    result = {
        "NeuralNetwork": {
            **metrics_dict(y_test, y_pred, y_prob, variant_key, "neural_network", feature_names),
            "loss"   : round(float(loss), 4),
            "history": hist_dict,
        }
    }
    return result, nn, scaler, feature_names, nn_shap


# ══════════════════════════════════════════════════════════════════════════════
# 6. PREDIÇÕES POR PAÍS
# ══════════════════════════════════════════════════════════════════════════════

def generate_predictions(df_raw: pd.DataFrame, nn_model, scaler, feature_names) -> dict:
    df_latest = df_raw.sort_values("year").groupby("iso3").last().reset_index()
    for col in feature_names:
        if col in df_latest.columns:
            df_latest[col] = df_latest[col].fillna(df_latest[col].median())
        else:
            df_latest[col] = 0.0
    X_pred   = df_latest[feature_names].values
    X_scaled = scaler.transform(X_pred)
    probs    = nn_model.predict(X_scaled, verbose=0).ravel()
    return {iso3: round(float(p), 4) for iso3, p in zip(df_latest["iso3"], probs)}


# ══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def save_json(obj, fname):
    path = os.path.join(MODEL_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    print(f"  ✓ {fname}")


def run_pipeline():
    print("=" * 65)
    print("  PIPELINE — WAR CONFLICT PREDICTION (v3 | 3 variantes + SHAP)")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    print(f"\nDados: {len(df):,} linhas | {df['iso3'].nunique()} países | "
          f"{df['year'].min()}–{df['year'].max()}")

    df_clean = clean_dataframe(df.copy())

    all_results     = {}
    all_shap        = {}
    all_predictions = {}
    all_configs     = {}

    for variant_key, vcfg in VARIANT_CONFIG.items():
        label = vcfg["label"]
        print(f"\n{'='*65}")
        print(f"  VARIANTE: {label}  [{variant_key}]")
        print(f"  {vcfg['description']}")
        print(f"{'='*65}")

        # Clássicos
        c_results, c_shap, X_tr, X_te, y_tr, y_te, feat_c = \
            train_classical_variant(df_clean, variant_key)

        # NN
        nn_result, nn_model, scaler, feat_nn, nn_shap = \
            train_nn_variant(df_clean, variant_key)

        # Combinar
        variant_results = {**c_results, **nn_result}
        all_results[variant_key] = variant_results

        # SHAP: clássicos + NN juntos
        all_shap[variant_key] = {**c_shap, "NeuralNetwork": nn_shap}

        # Predições usando NN da variante
        all_predictions[variant_key] = generate_predictions(df, nn_model, scaler, feat_nn)

        # Config por variante (inclui feature sets separados para clássico e NN)
        all_configs[variant_key] = {
            "label"             : label,
            "description"       : vcfg["description"],
            "classical_drop"    : vcfg["classical_drop"],
            "nn_drop"           : vcfg["nn_drop"],
            "features_classical": feat_c,
            "features_nn"       : feat_nn,
            # Para template backwards compat
            "features"          : feat_nn,
            "feature_names"     : feat_nn,
            "input_dim"         : len(feat_nn),
            "nn_architecture"   : NN_CONFIG["layers"],
        }

        # Sumário
        print(f"\n  Resultados — {label}:")
        rows = sorted(variant_results.items(), key=lambda x: x[1]["auc_roc"], reverse=True)
        for name, m in rows:
            print(f"    {name:<35} AUC={m['auc_roc']:.4f}  "
                  f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}")

    # ── Salvar ────────────────────────────────────────────────────────────────
    prediction_year  = int(df["year"].max()) + 1
    latest_data_year = int(df["year"].max())
    print(f"\n[SAVE] Ano de predição: {prediction_year}  (último dado: {latest_data_year})")
    print("[SAVE] Salvando artefatos...\n")

    save_json(all_results, "all_results.json")
    save_json(all_shap,    "shap_importances.json")
    save_json({
        "variants"       : all_predictions,
        "prediction_year": prediction_year,
        "data_year"      : latest_data_year,
    }, "predictions_all.json")
    save_json(all_configs, "model_config.json")

    # Retrocompatibilidade
    base_results = all_results.get("base", {})
    save_json(base_results, "baseline_results.json")
    save_json({"Neural Network": base_results.get("NeuralNetwork", {})}, "nn_results.json")
    save_json(all_predictions.get("base", {}), "predictions.json")

    # ── Sumário final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMÁRIO FINAL")
    print("=" * 65)
    for variant_key, variant_results in all_results.items():
        label = VARIANT_CONFIG[variant_key]["label"]
        print(f"\n  ▶ {label} ({variant_key})")
        print(f"    {'Modelo':<35} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7}")
        print("    " + "-" * 62)
        rows = sorted(variant_results.items(), key=lambda x: x[1]["auc_roc"], reverse=True)
        for name, m in rows:
            print(f"    {name:<35} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
                  f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['auc_roc']:>7.4f}")

    print(f"\n  Previsão para: {prediction_year}  |  Artefatos salvos em model/\n")


if __name__ == "__main__":
    run_pipeline()