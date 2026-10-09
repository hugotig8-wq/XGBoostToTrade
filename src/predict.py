"""
Módulo de Inferencia / Predicción para Producción.
Carga los artefactos guardados en .joblib y realiza predicciones
sobre nuevos datos ingresados.
"""

import joblib
import polars as pl


def predict_latest_opportunity(data_path: str = "data/parquet/ml_ready_dataset_enhanced.parquet"):
    # Cargar los artefactos desde .joblib
    clf_artifact = joblib.load("models/xgboost_classifier.joblib")
    reg_artifact = joblib.load("models/xgboost_regressor.joblib")

    clf_model = clf_artifact["model"]
    reg_model = reg_artifact["model"]
    features = clf_artifact["features"]

    # Cargar los últimos registros del dataset (Simulando datos de hoy)
    df = pl.read_parquet(data_path).sort("date")
    latest_data = df.tail(5)

    X_latest = latest_data.select(features).to_pandas()

    # Predicciones
    probs = clf_model.predict_proba(X_latest)[:, 1]
    signals = clf_model.predict(X_latest)
    predicted_profits = reg_model.predict(X_latest)

    # Combinar resultados
    results = latest_data.select(["date", "ticker", "close"]).with_columns([
        pl.Series("prob_success", probs).round(4),
        pl.Series("signal_buy", signals),
        pl.Series("est_max_profit_%", (predicted_profits * 100)).round(2)
    ])

    print("🚀 --- PREDICCIONES GENERADAS EN TIEMPO REAL ---")
    print(results)


if __name__ == "__main__":
    predict_latest_opportunity()
  
