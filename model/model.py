"""
=============================================================
  War Conflict Prediction Model — model_tensorflow.py
=============================================================
Versão com TensorFlow/Keras.
(versão sklearn sem deadlock: model.py)
 
Saídas em model/:
  - nn_model.keras         → modelo Keras treinado
  - scaler.joblib          → StandardScaler
  - baseline_results.json  → métricas dos modelos clássicos
  - nn_results.json        → métricas + histórico da rede
  - predictions.json       → {iso3: probability}
  - model_config.json      → arquitetura e parâmetros
  - all_results.json       → tudo combinado
 
Como rodar:
  cd model
  python model_tensorflow.py
"""
 
import os, json, warnings
import numpy as np
import pandas as pd
import joblib
 
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
# 1. PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════════════════
 
def remove_duplicate(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicatas removidas: {before - len(df)}")
    return df
 
def remove_constant_column(df, num=1):
    keep = df.columns[df.nunique() > num]
    dropped = list(df.columns.difference(keep))
    if dropped:
        print(f"  Colunas constantes removidas: {dropped}")
    return df[keep]
 
def missing_data(df, method="impute", perc_missing=0.3):
    df = df.copy()
    pct = df.isnull().sum() / len(df)
    drop_cols = pct[pct > perc_missing].index.tolist()
    if drop_cols:
        print(f"  Removidas por >{perc_missing*100:.0f}% NaN: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)
    if method == "impute":
        for col in df.columns:
            if df[col].dtype in ["int64", "float64", "Int8"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df
 
def clean_dataframe(df):
    print("\n[1] Limpeza de dados...")
    df = remove_duplicate(df)
    df = missing_data(df, method="impute", perc_missing=0.3)
    df = remove_constant_column(df, num=1)
    print("  Limpeza concluída.")
    return df
 
def build_features_baseline(df, target="conflict_next_year"):
    drop_cols = [c for c in ["iso3", "year", "gdp_per_capita", "conflict"] if c in df.columns]
    feat = df.drop(columns=drop_cols)
    return feat.drop(columns=[target]), feat[target].astype(int)
 
def build_features_nn(df, target="conflict_next_year"):
    drop_cols = [c for c in ["iso3", "year", "log_gdp_per_capita"] if c in df.columns]
    feat = df.drop(columns=drop_cols)
    return feat.drop(columns=[target]), feat[target].astype(int)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. MODELOS CLÁSSICOS DE BASELINE
# ══════════════════════════════════════════════════════════════════════════════
 
BASELINE_MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        min_samples_leaf=20, min_samples_split=30,
        class_weight="balanced", max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        max_depth=10, min_samples_leaf=20, max_features="sqrt",
        class_weight="balanced", n_estimators=200, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        learning_rate=0.01, n_estimators=100, max_depth=3, random_state=42),
}
 
def train_baseline_models(df_clean):
    print("\n[2] Treinando modelos clássicos de baseline...")
    year_col = df_clean["year"].reset_index(drop=True)
    X, y = build_features_baseline(df_clean.reset_index(drop=True))
 
    train_mask = year_col.values <= CUTOFF_YEAR
    test_mask  = year_col.values >  CUTOFF_YEAR
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
 
    print(f"  Treino: {train_mask.sum():,} | Teste: {test_mask.sum():,}")
    results = {}
 
    for name, model in BASELINE_MODELS.items():
        print(f"\n  → {name}", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy" : round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall"   : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1"       : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc_roc"  : round(float(roc_auc_score(y_test, y_prob)), 4),
            "split"    : "temporal (treino <= 2010 / teste > 2010)",
            "type"     : "baseline",
        }
        print(f"    AUC={results[name]['auc_roc']}  Recall={results[name]['recall']}  F1={results[name]['f1']}")
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name.replace(' ','_').lower()}_model.joblib"))
 
    return results
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. REDE NEURAL (TensorFlow / Keras)
#    ← EDITE AQUI PARA MUDAR O MODELO
# ══════════════════════════════════════════════════════════════════════════════
 
NN_CONFIG = {
    "layers": [
        {"units": 128, "activation": "relu", "dropout": 0.3},
        {"units": 2,   "activation": "relu", "dropout": None},
    ],
    "output_activation": "sigmoid",
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.2,
    "min_lr": 1e-5,
    "cutoff_year": CUTOFF_YEAR,
}
 
def build_nn_model(input_dim):
    model = tf.keras.Sequential(name="war_predictor_nn")
    for i, layer in enumerate(NN_CONFIG["layers"]):
        kwargs = {"activation": layer["activation"]}
        if i == 0:
            kwargs["input_shape"] = (input_dim,)
        model.add(tf.keras.layers.Dense(layer["units"], **kwargs))
        if layer["dropout"]:
            model.add(tf.keras.layers.Dropout(layer["dropout"]))
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
 
def train_neural_network(df_clean):
    print("\n[3] Treinando Rede Neural (TensorFlow/Keras)...")
    year_col = df_clean["year"].reset_index(drop=True)
    X, y = build_features_nn(df_clean.reset_index(drop=True))
    feature_names = X.columns.tolist()
 
    train_mask = year_col.values <= NN_CONFIG["cutoff_year"]
    test_mask  = year_col.values >  NN_CONFIG["cutoff_year"]
 
    X_train_raw = X[train_mask].values
    X_test_raw  = X[test_mask].values
    y_train     = y[train_mask].values
    y_test      = y[test_mask].values
 
    print(f"  Treino: {train_mask.sum():,} obs | Teste: {test_mask.sum():,} obs")
 
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
 
    # Pesos de classe
    neg, pos = np.bincount(y_train)
    total = neg + pos
    class_weight = {0: (1/neg)*(total/2.0), 1: (1/pos)*(total/2.0)}
    print(f"  Peso classe 0: {class_weight[0]:.2f} | Peso classe 1: {class_weight[1]:.2f}")
 
    model = build_nn_model(X_train.shape[1])
    model.summary()
 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=NN_CONFIG["early_stopping_patience"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=NN_CONFIG["reduce_lr_factor"],
            patience=NN_CONFIG["reduce_lr_patience"],
            min_lr=NN_CONFIG["min_lr"],
        ),
    ]
 
    history = model.fit(
        X_train, y_train,
        epochs=NN_CONFIG["epochs"],
        batch_size=NN_CONFIG["batch_size"],
        validation_split=NN_CONFIG["validation_split"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
 
    loss, acc, prec, rec, auc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    print(f"\n  AUC-ROC={auc:.4f}  Recall={rec:.4f}  F1={f1_score(y_test, y_pred, zero_division=0):.4f}")
 
    model.save(os.path.join(MODEL_DIR, "nn_model.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    print("  Salvo: nn_model.keras + scaler.joblib")
 
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
 
    config_out = dict(NN_CONFIG)
    config_out.update({
        "feature_names"   : feature_names,
        "input_dim"       : int(X_train.shape[1]),
        "training_samples": int(train_mask.sum()),
        "test_samples"    : int(test_mask.sum()),
        "framework"       : "TensorFlow/Keras",
    })
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
        json.dump(config_out, f, indent=4)
 
    nn_results = {
        "Neural Network": {
            "accuracy" : round(float(acc),  4),
            "precision": round(float(prec), 4),
            "recall"   : round(float(rec),  4),
            "f1"       : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "auc_roc"  : round(float(auc),  4),
            "loss"     : round(float(loss), 4),
            "split"    : "temporal (treino <= 2010 / teste > 2010)",
            "type"     : "neural_network",
            "history"  : hist_dict,
        }
    }
    with open(os.path.join(MODEL_DIR, "nn_results.json"), "w") as f:
        json.dump(nn_results, f, indent=4)
 
    return nn_results, scaler, model, feature_names
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDIÇÕES POR PAÍS
# ══════════════════════════════════════════════════════════════════════════════
 
def generate_country_predictions(df_raw, model, scaler, feature_names):
    print("\n[4] Gerando predições por país...")
    df_latest = df_raw.sort_values("year").groupby("iso3").last().reset_index()
 
    for col in feature_names:
        if col in df_latest.columns:
            df_latest[col] = df_latest[col].fillna(df_latest[col].median())
        else:
            df_latest[col] = 0.0
 
    X_pred   = df_latest[feature_names].values
    X_scaled = scaler.transform(X_pred)
    probs    = model.predict(X_scaled, verbose=0).ravel()
 
    predictions = {
        iso3: round(float(p), 4)
        for iso3, p in zip(df_latest["iso3"], probs)
    }
    print(f"  {len(predictions)} países processados.")
    return predictions
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5. PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
 
def run_pipeline():
    print("=" * 60)
    print(" PIPELINE — WAR CONFLICT PREDICTION (TensorFlow)")
    print("=" * 60)
 
    df = pd.read_csv(DATA_PATH)
    print(f"\nDados: {len(df):,} linhas | {df['iso3'].nunique()} países | "
          f"{df['year'].min()}–{df['year'].max()}")
 
    df_clean = clean_dataframe(df.copy())
 
    baseline_results = train_baseline_models(df_clean)
    with open(os.path.join(MODEL_DIR, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=4)
 
    nn_results, scaler, nn_model, feature_names = train_neural_network(df_clean)
 
    predictions = generate_country_predictions(df, nn_model, scaler, feature_names)
    with open(os.path.join(MODEL_DIR, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=4)
 
    all_results = {**baseline_results, **nn_results}
    with open(os.path.join(MODEL_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=4)
 
    print("\n" + "=" * 60)
    rows = sorted(all_results.items(), key=lambda x: x[1]["auc_roc"], reverse=True)
    header = f"{'Modelo':<30} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<30} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['f1']:>6.4f} {m['auc_roc']:>7.4f}")
 
    print("\n Pronto! Artefatos salvos em model/")
 
 
if __name__ == "__main__":
    run_pipeline()
