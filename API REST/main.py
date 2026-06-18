# ════════════════════════════════════════════════════════════════════════
#  InvestAI — API REST con FastAPI
#  Sistema Inteligente de Apoyo en Decisiones de Inversión
#
#  Universidad Nacional Mayor de San Marcos (UNMSM) — Facultad de Ingeniería
#  de Sistemas e Informática (FISI)
#  Curso: Introducción al Desarrollo de Software (iDeSo) — Ciclo 2026-II
#  Grupo 10 — Prof. E. D. Cancho-Rodríguez, MBA (GWU)
#
#  Este backend expone, mediante endpoints HTTP documentados con Swagger UI,
#  la MISMA lógica de ingesta de datos y modelos de Machine Learning / Deep
#  Learning desarrollada en los notebooks de investigación del proyecto:
#
#    1. Ingesta_de_Datos_Reales_con_yfinance.ipynb   → /api/mercado/{ticker}
#    2. Clasificador_SVC.ipynb                       → /api/svc/{ticker}
#    3. Clasificadores_RNN.ipynb                     → /api/rnns/{ticker}
#    4. Regresor_LSTM.ipynb                          → /api/lstm/{ticker}
#
#  Diseñado para ejecutarse en Google Colab (gratuito) y exponerse a
#  Internet mediante ngrok.
# ════════════════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings("ignore")

import os
import traceback
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


# ════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN GLOBAL Y REPRODUCIBILIDAD
# ════════════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Metadatos de los activos del estudio doctoral (mineras con operaciones
# en Perú). Si el ticker solicitado no está en este diccionario, la API
# igual intenta descargarlo desde Yahoo Finance, usando el propio símbolo
# como nombre y "USD" como moneda por defecto.
EMPRESAS_META: Dict[str, Dict[str, str]] = {
    "FSM":         {"nombre": "Fortuna Silver Mines Inc.",            "moneda": "USD"},
    "VOLCABC1.LM": {"nombre": "Volcan Compañía Minera S.A.A.",        "moneda": "PEN"},
    "ABX.TO":      {"nombre": "Barrick Gold Corporation",             "moneda": "CAD"},
    "BVN":         {"nombre": "Compañía de Minas Buenaventura S.A.A.", "moneda": "USD"},
    "BHP":         {"nombre": "BHP Group Limited",                    "moneda": "USD"},
}


def meta_empresa(ticker: str) -> Dict[str, str]:
    """Devuelve nombre/moneda conocidos del ticker, o un valor por defecto."""
    return EMPRESAS_META.get(ticker, {"nombre": ticker, "moneda": "USD"})


# ════════════════════════════════════════════════════════════════════════
# 2. CACHÉ EN MEMORIA
# ════════════════════════════════════════════════════════════════════════
# El entrenamiento del SVC (GridSearchCV), de los 4 clasificadores RNN y
# del Regresor LSTM es computacionalmente costoso (puede tomar desde
# varios segundos hasta varios minutos por ticker). Para que la API sea
# utilizable en la práctica, los resultados de cada módulo se cachean en
# memoria por ticker durante el tiempo de vida del proceso. Esto respeta
# exactamente la misma lógica/arquitectura de los notebooks; solo evita
# repetir el entrenamiento en cada solicitud HTTP.
_cache_mercado: Dict[str, Any] = {}
_cache_svc: Dict[str, Any] = {}
_cache_rnns: Dict[str, Any] = {}
_cache_lstm: Dict[str, Any] = {}   # guarda modelo + scaler + métricas (sin horizonte)


# ════════════════════════════════════════════════════════════════════════
# 3. UTILIDADES COMUNES
# ════════════════════════════════════════════════════════════════════════

def redondear(valor) -> Optional[float]:
    """Redondea un valor numérico a 4 decimales; maneja NaN / None."""
    try:
        if valor is None or pd.isna(valor):
            return None
        return round(float(valor), 4)
    except Exception:
        return None


def descargar_ohlcv(ticker: str, fecha_inicio: str, fecha_fin: str,
                     auto_adjust: bool = True) -> pd.DataFrame:
    """
    Descarga datos OHLCV reales desde Yahoo Finance usando yfinance,
    replicando el manejo de errores y el aplanado de columnas
    MultiIndex realizado en los notebooks originales.

    Lanza HTTPException(404) si el ticker no existe o no tiene datos.
    """
    try:
        df = yf.download(
            ticker, start=fecha_inicio, end=fecha_fin,
            auto_adjust=auto_adjust, progress=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al consultar Yahoo Finance para '{ticker}': {e}",
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron datos para el ticker '{ticker}'. "
                   f"Verifique que el símbolo sea válido en Yahoo Finance.",
        )

    # Aplanar columnas MultiIndex si yfinance las genera
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ════════════════════════════════════════════════════════════════════════
# 4. MÓDULO 1 — DATOS DE MERCADO E INDICADORES TÉCNICOS
#    (replica 1_Ingesta_de_Datos_Reales_con_yfinance.ipynb)
# ════════════════════════════════════════════════════════════════════════

def construir_datos_mercado(ticker: str) -> Dict[str, Any]:
    """Descarga OHLCV (2 años) y calcula SMA/EMA/RSI/MACD/Bollinger,
    igual que el Notebook 1, devolviendo la misma estructura JSON que
    consume el dashboard de InvestAI."""

    fecha_fin = datetime.today()
    fecha_inicio = fecha_fin - timedelta(days=730)
    fecha_inicio_str = fecha_inicio.strftime("%Y-%m-%d")
    fecha_fin_str = fecha_fin.strftime("%Y-%m-%d")

    df = descargar_ohlcv(ticker, fecha_inicio_str, fecha_fin_str, auto_adjust=False)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    df = df.dropna()

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos suficientes para '{ticker}'.")

    d = df.copy()

    # ── Medias móviles ──────────────────────────────────────────────
    d["sma_20"] = ta.trend.sma_indicator(d["close"], window=20)
    d["sma_50"] = ta.trend.sma_indicator(d["close"], window=50)
    d["ema_12"] = ta.trend.ema_indicator(d["close"], window=12)
    d["ema_26"] = ta.trend.ema_indicator(d["close"], window=26)

    # ── RSI ──────────────────────────────────────────────────────────
    d["rsi_14"] = ta.momentum.rsi(d["close"], window=14)

    # ── MACD ─────────────────────────────────────────────────────────
    macd_obj = ta.trend.MACD(d["close"], window_slow=26, window_fast=12, window_sign=9)
    d["macd"] = macd_obj.macd()
    d["macd_signal"] = macd_obj.macd_signal()
    d["macd_hist"] = macd_obj.macd_diff()

    # ── Bandas de Bollinger ─────────────────────────────────────────
    bb_obj = ta.volatility.BollingerBands(d["close"], window=20, window_dev=2)
    d["bb_upper"] = bb_obj.bollinger_hband()
    d["bb_middle"] = bb_obj.bollinger_mavg()
    d["bb_lower"] = bb_obj.bollinger_lband()

    # ── Señal BUY/SELL/HOLD según cruce SMA-20 / SMA-50 ──────────────
    d["signal"] = "HOLD"
    d.loc[d["sma_20"] > d["sma_50"], "signal"] = "BUY"
    d.loc[d["sma_20"] < d["sma_50"], "signal"] = "SELL"

    d = d.dropna()
    if d.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos suficientes tras calcular indicadores para '{ticker}'.")

    meta = meta_empresa(ticker)

    precio_actual = redondear(d["close"].iloc[-1])
    precio_anterior = redondear(d["close"].iloc[-2]) if len(d) > 1 else None
    variacion_dia = redondear(precio_actual - precio_anterior) if (precio_actual is not None and precio_anterior is not None) else None
    variacion_pct = redondear((variacion_dia / precio_anterior) * 100) if (variacion_dia is not None and precio_anterior) else None

    serie = []
    for fecha, fila in d.iterrows():
        serie.append({
            "fecha": fecha.strftime("%Y-%m-%d"),
            "open": redondear(fila["open"]),
            "high": redondear(fila["high"]),
            "low": redondear(fila["low"]),
            "close": redondear(fila["close"]),
            "adj_close": redondear(fila.get("adj_close")),
            "volume": int(fila["volume"]) if not pd.isna(fila["volume"]) else None,
            "sma_20": redondear(fila["sma_20"]),
            "sma_50": redondear(fila["sma_50"]),
            "ema_12": redondear(fila["ema_12"]),
            "ema_26": redondear(fila["ema_26"]),
            "rsi_14": redondear(fila["rsi_14"]),
            "macd": redondear(fila["macd"]),
            "macd_signal": redondear(fila["macd_signal"]),
            "macd_hist": redondear(fila["macd_hist"]),
            "bb_upper": redondear(fila["bb_upper"]),
            "bb_middle": redondear(fila["bb_middle"]),
            "bb_lower": redondear(fila["bb_lower"]),
            "signal": fila["signal"],
        })

    resultado = {
        "metadata": {
            "generado_en": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "periodo_inicio": fecha_inicio_str,
            "periodo_fin": fecha_fin_str,
            "fuente": "Yahoo Finance (yfinance)",
            "ticker": ticker,
        },
        "nombre": meta["nombre"],
        "ticker": ticker,
        "precio_actual": precio_actual,
        "variacion_dia": variacion_dia,
        "variacion_pct": variacion_pct,
        "signal_actual": d["signal"].iloc[-1],
        "rsi_actual": redondear(d["rsi_14"].iloc[-1]),
        "macd_actual": redondear(d["macd"].iloc[-1]),
        "volumen_actual": int(d["volume"].iloc[-1]),
        "total_registros": len(d),
        "serie": serie,
    }
    return resultado


# ════════════════════════════════════════════════════════════════════════
# 5. MÓDULO 2 — CLASIFICADOR SVC
#    (replica 2_Clasificador_SVC.ipynb)
# ════════════════════════════════════════════════════════════════════════

HORIZONTE_PREDICCION_SVC = 5      # días hábiles de horizonte de la tendencia
RATIO_TRAIN_SVC = 0.80
PARAM_GRID_SVC = {
    "svc__kernel": ["linear", "rbf", "poly", "sigmoid"],
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": ["scale", "auto"],
}
CV_FOLDS_SVC = 5

COLUMNAS_EXCLUIR_SVC = [
    "Open", "High", "Low", "Close", "Volume", "Trend",
    "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "bb_upper", "bb_lower", "bb_middle",
    "macd", "macd_signal",
    "atr_14",
    "momentum_5", "momentum_10",
]


def calcular_caracteristicas_svc(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de características técnicas + variable objetivo 'Trend'
    (1=BUY, 0=SELL), idéntica a la del Notebook 2. No usa valores futuros
    para calcular las features; sólo 'Trend' mira hacia adelante."""
    df = df.copy()

    df["retorno_log"] = np.log(df["Close"] / df["Close"].shift(1))
    df["retorno_1d"] = df["Close"].pct_change(1)
    df["retorno_3d"] = df["Close"].pct_change(3)
    df["retorno_5d"] = df["Close"].pct_change(5)
    df["retorno_10d"] = df["Close"].pct_change(10)

    for ventana in [10, 20, 50]:
        sma = SMAIndicator(close=df["Close"], window=ventana)
        df[f"sma_{ventana}"] = sma.sma_indicator()
        df[f"precio_sma_{ventana}"] = df["Close"] / df[f"sma_{ventana}"] - 1

    for ventana in [12, 26]:
        ema = EMAIndicator(close=df["Close"], window=ventana)
        df[f"ema_{ventana}"] = ema.ema_indicator()
    df["ema_cruce"] = df["ema_12"] / df["ema_26"] - 1

    macd_ind = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()

    rsi_ind = RSIIndicator(close=df["Close"], window=14)
    df["rsi_14"] = rsi_ind.rsi()
    df["rsi_zona_alta"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_zona_baja"] = (df["rsi_14"] < 30).astype(int)

    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct_b"] = bb.bollinger_pband()

    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["atr_14"] = atr.average_true_range()
    df["atr_pct"] = df["atr_14"] / df["Close"]

    df["volatilidad_10d"] = df["retorno_log"].rolling(10).std() * np.sqrt(252)
    df["volatilidad_20d"] = df["retorno_log"].rolling(20).std() * np.sqrt(252)

    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["roc_5"] = (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5) * 100
    df["roc_10"] = (df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10) * 100

    df["rango_hl"] = (df["High"] - df["Low"]) / df["Close"]
    df["rango_hl_5d"] = df["rango_hl"].rolling(5).mean()

    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["vol_cambio"] = df["Volume"].pct_change()

    # Variable objetivo: Trend (1=BUY, 0=SELL) — horizonte de 5 días hábiles
    precio_futuro = df["Close"].shift(-HORIZONTE_PREDICCION_SVC)
    df["Trend"] = (precio_futuro > df["Close"]).astype(int)

    return df


def entrenar_svc(ticker: str) -> Dict[str, Any]:
    """Pipeline completo del Notebook 2 para UN ticker: descarga, features,
    partición temporal 80/20, GridSearchCV (StandardScaler+SVC) con
    TimeSeriesSplit, métricas y construcción del JSON de salida."""

    fecha_inicio = "2020-01-01"
    fecha_fin = datetime.today().strftime("%Y-%m-%d")

    df_raw = descargar_ohlcv(ticker, fecha_inicio, fecha_fin, auto_adjust=True)
    df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df_raw = df_raw[df_raw["Close"] > 0].dropna(subset=["Close"])

    if len(df_raw) < 200:
        raise HTTPException(
            status_code=404,
            detail=f"Datos insuficientes para entrenar el SVC de '{ticker}' "
                   f"({len(df_raw)} registros, se requieren ≥200).",
        )

    df_feat = calcular_caracteristicas_svc(df_raw).dropna()
    if df_feat.empty:
        raise HTTPException(status_code=404, detail=f"Sin datos suficientes tras la ingeniería de características para '{ticker}'.")

    ticker_ref_cols = df_feat.columns
    features = [c for c in ticker_ref_cols if c not in COLUMNAS_EXCLUIR_SVC]

    n = len(df_feat)
    corte = int(n * RATIO_TRAIN_SVC)
    df_train = df_feat.iloc[:corte].copy()
    df_test = df_feat.iloc[corte:].copy()

    X_train = df_train[features].values
    y_train = df_train["Trend"].values
    X_test = df_test[features].values
    y_test = df_test["Trend"].values
    fecha_corte = df_train.index[-1].strftime("%Y-%m-%d")

    # ── Limpieza de valores no finitos (idéntica al notebook) ──────────
    def limpiar(X, y):
        X = X.copy()
        if np.isinf(X).any() or np.isnan(X).any():
            X[np.isinf(X)] = np.nan
            combinado = np.hstack((X, y.reshape(-1, 1)))
            mask = ~np.isnan(combinado).any(axis=1)
            limpio = combinado[mask]
            return limpio[:, :-1], limpio[:, -1], mask
        return X, y, np.ones(len(X), dtype=bool)

    X_train, y_train, _ = limpiar(X_train, y_train)
    X_test, y_test, mask_test = limpiar(X_test, y_test)

    if len(X_train) == 0:
        raise HTTPException(status_code=500, detail=f"'{ticker}': el conjunto de entrenamiento quedó vacío tras la limpieza de datos.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=SEED, cache_size=500)),
    ])

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS_SVC)
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=PARAM_GRID_SVC, cv=tscv,
        scoring="f1_macro", n_jobs=-1, verbose=0, refit=True,
    )

    try:
        grid_search.fit(X_train, y_train)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Error durante GridSearchCV para '{ticker}': {e}")

    mejor_modelo = grid_search.best_estimator_
    mejores_params = grid_search.best_params_
    mejor_cv_score = grid_search.best_score_

    if len(X_test) == 0:
        raise HTTPException(status_code=404, detail=f"Sin datos de prueba disponibles para '{ticker}' tras la limpieza.")

    y_pred = mejor_modelo.predict(X_test)
    y_prob = mejor_modelo.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    df_test_sig = df_test[mask_test].copy()
    df_test_sig["prediccion"] = y_pred
    df_test_sig["probabilidad"] = y_prob
    df_test_sig["senal"] = df_test_sig["prediccion"].map({1: "BUY", 0: "SELL"})

    meta = meta_empresa(ticker)
    fechas = [d.strftime("%Y-%m-%d") for d in df_test_sig.index]
    precios = [round(float(p), 4) for p in df_test_sig["Close"]]
    senales = list(df_test_sig["senal"])

    ultima_senal = senales[-1]
    ultima_prob = float(y_prob[-1])
    confianza = ultima_prob if ultima_senal == "BUY" else (1.0 - ultima_prob)

    tn, fp, fn, tp = cm.ravel()
    matriz_html = [[int(tp), int(fp)], [int(fn), int(tn)]]

    n_buy = int((df_test_sig["senal"] == "BUY").sum())
    n_sell = int((df_test_sig["senal"] == "SELL").sum())

    senales_detalle = [
        {"fecha": f, "precio": p, "senal": s, "prob": round(float(pb), 4)}
        for f, p, s, pb in zip(fechas, precios, senales, y_prob.tolist())
    ]

    resultado = {
        "ticker": ticker,
        "nombre": meta["nombre"],
        "moneda": meta["moneda"],
        "generado_en": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "horizonte_dias": HORIZONTE_PREDICCION_SVC,
        "ratio_train": RATIO_TRAIN_SVC,
        "cv_folds": CV_FOLDS_SVC,
        "fechas": fechas,
        "precios": precios,
        "senales": senales,
        "senal": ultima_senal,
        "conf": round(confianza, 4),
        "metricas": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
        },
        "matriz": matriz_html,
        "clases": {"buy": n_buy, "sell": n_sell},
        "kernel": {
            "tipo": mejores_params["svc__kernel"],
            "C": mejores_params["svc__C"],
            "gamma": mejores_params["svc__gamma"],
        },
        "senales_detalle": senales_detalle,
        "modelo_meta": {
            "n_train": len(df_train),
            "n_test": len(df_test),
            "fecha_corte": fecha_corte,
            "cv_f1_macro": round(float(mejor_cv_score), 4),
            "n_features": len(features),
        },
    }
    return resultado


# ════════════════════════════════════════════════════════════════════════
# 6. MÓDULO 3 — CLASIFICADORES RNN (LSTM · BiLSTM · GRU · SimpleRNN)
#    (replica 3_Clasificadores_RNN.ipynb)
# ════════════════════════════════════════════════════════════════════════

VENTANA_RNN = 20
EPOCAS_RNN = 80
BATCH_RNN = 64
LR_RNN = 0.001
SPLIT_RNN = 0.80

MODELOS_CONFIG_RNN = [
    {"id": "LSTM", "arq": "LSTM(260)→Drop(0.2)→LSTM(130)→Dense(1)", "color": "#1565c0"},
    {"id": "BiLSTM", "arq": "BiLSTM(200)→Drop(0.3)→BiLSTM(100)→Dense(1)", "color": "#6a1b9a"},
    {"id": "GRU", "arq": "GRU(280)→GRU(140)→Dense(1)", "color": "#e65100"},
    {"id": "SimpleRNN", "arq": "RNN(180)→RNN(90)→Dense(1)", "color": "#00838f"},
]

FEATURES_RNN = [
    "cierre", "apertura", "maximo", "minimo", "volumen",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_superior", "bb_media", "bb_inferior",
    "atr_14", "retorno", "retorno_log",
]


def calcular_indicadores_rnn(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Indicadores técnicos + variable objetivo binaria 'trend', idéntico
    al Notebook 3 (Celda 4)."""
    df = df.copy()

    df["sma_20"] = ta.trend.sma_indicator(df["cierre"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["cierre"], window=50)
    df["ema_12"] = ta.trend.ema_indicator(df["cierre"], window=12)
    df["ema_26"] = ta.trend.ema_indicator(df["cierre"], window=26)
    df["rsi_14"] = ta.momentum.rsi(df["cierre"], window=14)

    macd_obj = ta.trend.MACD(df["cierre"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    bb = ta.volatility.BollingerBands(df["cierre"], window=20, window_dev=2)
    df["bb_superior"] = bb.bollinger_hband()
    df["bb_media"] = bb.bollinger_mavg()
    df["bb_inferior"] = bb.bollinger_lband()

    df["atr_14"] = ta.volatility.average_true_range(df["maximo"], df["minimo"], df["cierre"], window=14)

    df["retorno"] = df["cierre"].pct_change()
    df["retorno_log"] = np.log(df["cierre"] / df["cierre"].shift(1))

    df["trend"] = (df["cierre"].shift(-1) > df["cierre"]).astype(int)

    df = df.dropna()
    df = df.iloc[:-1]   # la tendencia del último día es desconocida

    if len(df) < VENTANA_RNN * 2:
        return None
    return df


def crear_secuencias_rnn(X: np.ndarray, y: np.ndarray, ventana: int):
    X_seq, y_seq = [], []
    for i in range(ventana, len(X)):
        X_seq.append(X[i - ventana:i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def construir_lstm(n_features: int) -> keras.Model:
    modelo = keras.Sequential([
        keras.Input(shape=(VENTANA_RNN, n_features)),
        layers.LSTM(260, return_sequences=True, name="lstm_1"),
        layers.Dropout(0.2, name="dropout_1"),
        layers.LSTM(130, return_sequences=False, name="lstm_2"),
        layers.Dense(1, activation="sigmoid", name="salida"),
    ], name="LSTM_Clasificador")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_RNN),
                    loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


def construir_bilstm(n_features: int) -> keras.Model:
    modelo = keras.Sequential([
        keras.Input(shape=(VENTANA_RNN, n_features)),
        layers.Bidirectional(layers.LSTM(200, return_sequences=True), name="bilstm_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Bidirectional(layers.LSTM(100, return_sequences=False), name="bilstm_2"),
        layers.Dense(1, activation="sigmoid", name="salida"),
    ], name="BiLSTM_Clasificador")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_RNN),
                    loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


def construir_gru(n_features: int) -> keras.Model:
    modelo = keras.Sequential([
        keras.Input(shape=(VENTANA_RNN, n_features)),
        layers.GRU(280, return_sequences=True, name="gru_1"),
        layers.GRU(140, return_sequences=False, name="gru_2"),
        layers.Dense(1, activation="sigmoid", name="salida"),
    ], name="GRU_Clasificador")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_RNN),
                    loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


def construir_simple_rnn(n_features: int) -> keras.Model:
    modelo = keras.Sequential([
        keras.Input(shape=(VENTANA_RNN, n_features)),
        layers.SimpleRNN(180, return_sequences=True, name="rnn_1"),
        layers.SimpleRNN(90, return_sequences=False, name="rnn_2"),
        layers.Dense(1, activation="sigmoid", name="salida"),
    ], name="SimpleRNN_Clasificador")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_RNN),
                    loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


CONSTRUCTORES_RNN = {
    "LSTM": construir_lstm,
    "BiLSTM": construir_bilstm,
    "GRU": construir_gru,
    "SimpleRNN": construir_simple_rnn,
}


def entrenar_rnns(ticker: str) -> Dict[str, Any]:
    """Entrena los 4 clasificadores RNN (LSTM, BiLSTM, GRU, SimpleRNN)
    para un ticker, replicando exactamente la Celda 7 del Notebook 3, y
    construye la misma estructura JSON de la Celda 9."""

    fecha_fin = datetime.today().strftime("%Y-%m-%d")
    fecha_inicio = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    df_raw = descargar_ohlcv(ticker, fecha_inicio, fecha_fin, auto_adjust=True)
    if len(df_raw) < VENTANA_RNN * 3:
        raise HTTPException(status_code=404, detail=f"Datos insuficientes para '{ticker}' ({len(df_raw)} filas).")

    df_raw = df_raw.rename(columns={
        "Open": "apertura", "High": "maximo", "Low": "minimo",
        "Close": "cierre", "Volume": "volumen",
    })
    df_raw = df_raw.dropna(subset=["cierre"])

    df_proc = calcular_indicadores_rnn(df_raw)
    if df_proc is None:
        raise HTTPException(status_code=404, detail=f"Datos insuficientes tras calcular indicadores para '{ticker}'.")

    X_raw = df_proc[FEATURES_RNN].values
    y_raw = df_proc["trend"].values

    corte = int(len(X_raw) * SPLIT_RNN)
    X_train_raw, X_test_raw = X_raw[:corte], X_raw[corte:]
    y_train_raw, y_test_raw = y_raw[:corte], y_raw[corte:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_test_norm = scaler.transform(X_test_raw)

    X_train, y_train = crear_secuencias_rnn(X_train_norm, y_train_raw, VENTANA_RNN)
    X_test, y_test = crear_secuencias_rnn(X_test_norm, y_test_raw, VENTANA_RNN)

    if len(X_train) == 0 or len(X_test) == 0:
        raise HTTPException(status_code=404, detail=f"Secuencias insuficientes para entrenar/evaluar RNNs en '{ticker}'.")

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=0,
    )

    resultados_modelos: Dict[str, Any] = {}

    for cfg in MODELOS_CONFIG_RNN:
        modelo_id = cfg["id"]

        tf.random.set_seed(SEED)
        np.random.seed(SEED)

        modelo = CONSTRUCTORES_RNN[modelo_id](X_train.shape[2])

        hist = modelo.fit(
            X_train, y_train, epochs=EPOCAS_RNN, batch_size=BATCH_RNN,
            validation_split=0.15, callbacks=[early_stopping], verbose=0,
        )

        y_prob = modelo.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
        confianza_ultima = float(y_prob[-1])
        senal_ultima = "BUY" if y_pred[-1] == 1 else "SELL"

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        acc_hist = [round(float(a), 4) for a in hist.history["accuracy"]]
        if len(acc_hist) < EPOCAS_RNN:
            acc_hist += [acc_hist[-1]] * (EPOCAS_RNN - len(acc_hist))

        senales_pred = ["BUY" if p == 1 else "SELL" for p in y_pred]

        resultados_modelos[modelo_id] = {
            "acc": round(float(acc), 4), "prec": round(float(prec), 4),
            "rec": round(float(rec), 4), "f1": round(float(f1), 4),
            "senal": senal_ultima, "conf": round(confianza_ultima, 2),
            "accHist": acc_hist, "senales": senales_pred,
        }

        keras.backend.clear_session()
        del modelo

    # ── Construcción del JSON final (Celda 9) ──────────────────────────
    df_test = df_proc.iloc[corte:]
    df_senales = df_test.iloc[VENTANA_RNN:].copy()

    fechas = [str(d.date()) for d in df_senales.index]
    precios = [round(float(v), 4) for v in df_senales["cierre"].values]

    lista_modelos = []
    for cfg in MODELOS_CONFIG_RNN:
        mid = cfg["id"]
        r = resultados_modelos[mid]
        senales_ajustadas = r["senales"][:len(fechas)]
        if len(senales_ajustadas) < len(fechas):
            senales_ajustadas += [r["senal"]] * (len(fechas) - len(senales_ajustadas))

        lista_modelos.append({
            "id": mid, "arq": cfg["arq"], "color": cfg["color"],
            "acc": r["acc"], "prec": r["prec"], "rec": r["rec"], "f1": r["f1"],
            "accHist": r["accHist"], "senales": senales_ajustadas,
            "senal": r["senal"], "conf": r["conf"],
        })

    meta = meta_empresa(ticker)
    resultado = {
        "ticker": ticker,
        "nombre": meta["nombre"],
        "fecha_generado": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "periodo": f"{fecha_inicio} → {fecha_fin}",
        "ventana": VENTANA_RNN,
        "epocas": EPOCAS_RNN,
        "fechas": fechas,
        "precios": precios,
        "modelos": lista_modelos,
    }
    return resultado


# ════════════════════════════════════════════════════════════════════════
# 7. MÓDULO 4 — REGRESOR LSTM DE PRECIOS
#    (replica 4__Regresor_LSTM.ipynb)
# ════════════════════════════════════════════════════════════════════════

VENTANA_LSTM = 60          # ventana de entrada FIJA (no modificable)
EPOCAS_LSTM = 100
BATCH_SIZE_LSTM = 32
LEARNING_RATE_LSTM = 0.001
VALIDACION_LSTM = 0.1
TEST_SPLIT_LSTM = 0.15
Z_CONFIANZA_LSTM = 1.96


def construir_modelo_lstm_regresor(ventana: int, lr: float = LEARNING_RATE_LSTM) -> keras.Model:
    """LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, relu)
    → Dense(1, linear), idéntico al Notebook 4 (Sección 5)."""
    modelo = keras.Sequential(name="LSTM_Regressor_Precios", layers=[
        layers.LSTM(64, return_sequences=True, input_shape=(ventana, 1), name="lstm_64"),
        layers.Dropout(0.2, name="dropout_1"),
        layers.LSTM(32, return_sequences=False, name="lstm_32"),
        layers.Dropout(0.2, name="dropout_2"),
        layers.Dense(16, activation="relu", name="dense_16"),
        layers.Dense(1, activation="linear", name="output_precio"),
    ])
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss="mean_squared_error", metrics=["mae"])
    return modelo


def crear_ventanas_deslizantes_lstm(serie_normalizada: np.ndarray, ventana: int):
    X, y = [], []
    for i in range(ventana, len(serie_normalizada)):
        X.append(serie_normalizada[i - ventana:i, 0])
        y.append(serie_normalizada[i, 0])
    return np.array(X).reshape(-1, ventana, 1), np.array(y)


def preprocesar_ticker_lstm(serie: pd.Series, ventana: int, test_split: float):
    valores = serie.values.reshape(-1, 1)
    corte = int(len(valores) * (1 - test_split))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(valores[:corte])
    valores_norm = scaler.transform(valores)

    X_all, y_all = crear_ventanas_deslizantes_lstm(valores_norm, ventana)
    corte_ventanas = corte - ventana

    X_train, y_train = X_all[:corte_ventanas], y_all[:corte_ventanas]
    X_test, y_test = X_all[corte_ventanas:], y_all[corte_ventanas:]
    return X_train, y_train, X_test, y_test, scaler, valores


def calcular_metricas_lstm(y_real: np.ndarray, y_pred: np.ndarray, scaler: MinMaxScaler) -> Dict[str, Any]:
    y_real_usd = scaler.inverse_transform(y_real.reshape(-1, 1)).flatten()
    y_pred_usd = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    rmse_usd = np.sqrt(mean_squared_error(y_real_usd, y_pred_usd))
    precio_med = np.mean(y_real_usd)
    rmse_pct = (rmse_usd / precio_med) * 100 if precio_med != 0 else 0
    mae_usd = mean_absolute_error(y_real_usd, y_pred_usd)
    r2 = r2_score(y_real_usd, y_pred_usd)

    return {
        "rmse_usd": round(float(rmse_usd), 4), "rmse_pct": round(float(rmse_pct), 4),
        "mae_usd": round(float(mae_usd), 4), "r2": round(float(r2), 6),
        "precio_med": round(float(precio_med), 4),
        "y_real_usd": y_real_usd.tolist(), "y_pred_usd": y_pred_usd.tolist(),
    }


def predecir_horizonte_lstm(modelo, serie_norm: np.ndarray, ventana: int,
                              horizonte: int, scaler: MinMaxScaler) -> List[float]:
    """Predicción multi-paso iterativa (random-walk forward): usa cada
    predicción como entrada para predecir el siguiente paso."""
    ventana_actual = serie_norm[-ventana:].copy().tolist()
    predicciones_norm = []

    for _ in range(horizonte):
        entrada = np.array(ventana_actual[-ventana:]).reshape(1, ventana, 1)
        pred = modelo.predict(entrada, verbose=0)[0, 0]
        predicciones_norm.append(pred)
        ventana_actual.append([pred])

    preds_array = np.array(predicciones_norm).reshape(-1, 1)
    return scaler.inverse_transform(preds_array).flatten().tolist()


def calcular_bandas_lstm(predicciones_usd: List[float], rmse_usd: float, z: float = Z_CONFIANZA_LSTM):
    bandas_inf, bandas_sup = [], []
    for i, pred in enumerate(predicciones_usd):
        error = z * rmse_usd * np.sqrt(i + 1)
        bandas_inf.append(round(pred - error, 4))
        bandas_sup.append(round(pred + error, 4))
    return bandas_inf, bandas_sup


def entrenar_lstm_regresor(ticker: str) -> Dict[str, Any]:
    """Entrena (una sola vez, cacheado) el LSTM Regressor de precios para
    un ticker, replicando la Sección 8 del Notebook 4. Devuelve el modelo
    entrenado junto con el scaler, métricas e histórico — la predicción
    por horizonte se calcula aparte (es barata: solo inferencia)."""

    fecha_fin = datetime.today().strftime("%Y-%m-%d")
    fecha_inicio = (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    df = descargar_ohlcv(ticker, fecha_inicio, fecha_fin, auto_adjust=True)
    if len(df) < VENTANA_LSTM * 2:
        raise HTTPException(status_code=404, detail=f"Datos insuficientes para '{ticker}' ({len(df)} filas, se requieren ≥{VENTANA_LSTM*2}).")

    serie = df["Close"].dropna()
    if hasattr(serie, "columns"):
        serie = serie.iloc[:, 0]

    X_train, y_train, X_test, y_test, scaler, valores = preprocesar_ticker_lstm(
        serie, VENTANA_LSTM, TEST_SPLIT_LSTM,
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise HTTPException(status_code=404, detail=f"Ventanas insuficientes para entrenar/evaluar el LSTM Regressor de '{ticker}'.")

    modelo = construir_modelo_lstm_regresor(VENTANA_LSTM, LEARNING_RATE_LSTM)

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]
    modelo.fit(
        X_train, y_train, epochs=EPOCAS_LSTM, batch_size=BATCH_SIZE_LSTM,
        validation_split=VALIDACION_LSTM, callbacks=cbs, verbose=0,
    )

    y_pred_test = modelo.predict(X_test, verbose=0).flatten()
    metricas = calcular_metricas_lstm(y_test, y_pred_test, scaler)

    serie_norm_completa = scaler.transform(valores)
    fecha_ultimo = pd.to_datetime(serie.index[-1])

    meta = meta_empresa(ticker)

    cache_entry = {
        "ticker": ticker,
        "nombre": meta["nombre"],
        "modelo": modelo,
        "scaler": scaler,
        "serie_norm_completa": serie_norm_completa,
        "metricas": metricas,
        "ultimo_precio": round(float(serie.iloc[-1]), 4),
        "fecha_ultimo": str(fecha_ultimo.date()),
        "test_real": metricas["y_real_usd"][-30:],
        "test_pred": metricas["y_pred_usd"][-30:],
        "historico_fechas": [str(d.date()) for d in pd.to_datetime(serie.index)[-90:]],
        "historico_precios": serie.values[-90:].tolist(),
    }
    return cache_entry


def construir_respuesta_lstm(ticker: str, horizonte: int) -> Dict[str, Any]:
    """Usa el modelo cacheado (entrenándolo si es la primera vez) y calcula
    las predicciones futuras para el horizonte solicitado por el usuario."""

    if ticker not in _cache_lstm:
        _cache_lstm[ticker] = entrenar_lstm_regresor(ticker)

    entry = _cache_lstm[ticker]
    modelo = entry["modelo"]
    scaler = entry["scaler"]
    metricas = entry["metricas"]

    preds_usd = predecir_horizonte_lstm(
        modelo, entry["serie_norm_completa"], VENTANA_LSTM, horizonte, scaler,
    )
    b_inf, b_sup = calcular_bandas_lstm(preds_usd, metricas["rmse_usd"], Z_CONFIANZA_LSTM)
    predicciones = [round(p, 4) for p in preds_usd]

    resultado = {
        "ticker": ticker,
        "nombre": entry["nombre"],
        "modelo": {
            "tipo": "LSTM Regressor",
            "arquitectura": "LSTM(64)->Dropout(0.2)->LSTM(32)->Dropout(0.2)->Dense(16,relu)->Dense(1,linear)",
            "ventana_dias": VENTANA_LSTM,
            "epocas": EPOCAS_LSTM,
            "batch_size": BATCH_SIZE_LSTM,
            "optimizador": "Adam",
            "learning_rate": LEARNING_RATE_LSTM,
            "nivel_confianza": "95% (±1.96 * RMSE)",
            "generado_en": datetime.now().isoformat(),
        },
        "ultimo_precio": entry["ultimo_precio"],
        "fecha_ultimo": entry["fecha_ultimo"],
        "metricas": metricas_sin_arrays(metricas),
        "test_comparison": {"real": entry["test_real"], "pred": entry["test_pred"]},
        "historico": {"fechas": entry["historico_fechas"], "precios": entry["historico_precios"]},
        "horizonte_dias": horizonte,
        "prediccion": {
            "dias": horizonte,
            "predicciones": predicciones,
            "banda_inf": b_inf,
            "banda_sup": b_sup,
            "precio_final": predicciones[-1],
            "ic_inf_final": b_inf[-1],
            "ic_sup_final": b_sup[-1],
        },
    }
    return resultado


def metricas_sin_arrays(metricas: Dict[str, Any]) -> Dict[str, Any]:
    """Devuelve las métricas de evaluación sin los arrays largos
    (y_real_usd / y_pred_usd), que ya se exponen en 'test_comparison'."""
    return {k: v for k, v in metricas.items() if k not in ("y_real_usd", "y_pred_usd")}


# ════════════════════════════════════════════════════════════════════════
# 8. APLICACIÓN FASTAPI
# ════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="InvestAI API",
    description=(
        "API REST del Sistema InvestAI — Sistema Inteligente de Apoyo en "
        "Decisiones de Inversión. Expone datos de mercado reales y "
        "predicciones de los modelos de IA (SVC, RNN, LSTM) desarrollados "
        "en los notebooks de investigación del Grupo 10 (iDeSo - UNMSM FISI)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS habilitado para cualquier origen (permite conexión desde el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/salud", tags=["Salud"], summary="Estado de salud del servidor")
def api_salud():
    """Health check del servidor: confirma que la API está activa y
    reporta el estado de las cachés de modelos en memoria."""
    return {
        "status": "ok",
        "servicio": "InvestAI API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "cache": {
            "mercado": list(_cache_mercado.keys()),
            "svc": list(_cache_svc.keys()),
            "rnns": list(_cache_rnns.keys()),
            "lstm": list(_cache_lstm.keys()),
        },
    }


@app.get("/api/mercado/{ticker}", tags=["Mercado"],
         summary="Datos OHLCV con indicadores técnicos")
def api_mercado(ticker: str = Path(..., description="Símbolo bursátil en Yahoo Finance, ej. 'BVN'")):
    """Devuelve la serie histórica OHLCV (2 años) del ticker junto con
    indicadores técnicos (SMA-20/50, EMA-12/26, RSI-14, MACD, Bandas de
    Bollinger) y la señal BUY/SELL/HOLD basada en el cruce de medias
    móviles. Replica la lógica del Notebook 1."""
    ticker = ticker.strip().upper()
    try:
        if ticker not in _cache_mercado:
            _cache_mercado[ticker] = construir_datos_mercado(ticker)
        return _cache_mercado[ticker]
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno procesando '{ticker}': {e}")


@app.get("/api/svc/{ticker}", tags=["Modelos IA"],
         summary="Señales y métricas del clasificador SVC")
def api_svc(ticker: str = Path(..., description="Símbolo bursátil en Yahoo Finance, ej. 'BVN'")):
    """Entrena (o recupera de la caché) el clasificador SVC con
    optimización GridSearchCV + TimeSeriesSplit y devuelve señales
    BUY/SELL, métricas (accuracy/precision/recall/F1), matriz de
    confusión y los hiperparámetros ganadores. Replica la lógica del
    Notebook 2. La primera solicitud por ticker puede tardar varios
    segundos debido al entrenamiento."""
    ticker = ticker.strip().upper()
    try:
        if ticker not in _cache_svc:
            _cache_svc[ticker] = entrenar_svc(ticker)
        return _cache_svc[ticker]
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno entrenando SVC para '{ticker}': {e}")


@app.get("/api/rnns/{ticker}", tags=["Modelos IA"],
         summary="Señales y métricas de los 4 clasificadores RNN")
def api_rnns(ticker: str = Path(..., description="Símbolo bursátil en Yahoo Finance, ej. 'BVN'")):
    """Entrena (o recupera de la caché) los 4 clasificadores de
    tendencia basados en redes recurrentes (LSTM, BiLSTM, GRU,
    SimpleRNN) y devuelve, por cada uno, accuracy/precision/recall/F1,
    historial de accuracy por época, señales BUY/SELL y la señal/
    confianza más reciente. Replica la lógica del Notebook 3.
    ⚠️ La primera solicitud por ticker puede tardar varios minutos en
    CPU (se recomienda activar GPU en el entorno de ejecución de Colab)."""
    ticker = ticker.strip().upper()
    try:
        if ticker not in _cache_rnns:
            _cache_rnns[ticker] = entrenar_rnns(ticker)
        return _cache_rnns[ticker]
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno entrenando RNNs para '{ticker}': {e}")


@app.get("/api/lstm/{ticker}", tags=["Modelos IA"],
         summary="Predicciones de precios del regresor LSTM")
def api_lstm(
    ticker: str = Path(..., description="Símbolo bursátil en Yahoo Finance, ej. 'BVN'"),
    horizonte: int = Query(30, ge=1, le=180, description="Días futuros a predecir"),
):
    """Entrena (o recupera de la caché) el Regresor LSTM de precios
    (ventana fija de 60 días) y devuelve la predicción de precio para
    el horizonte solicitado, junto con bandas de confianza del 95%,
    métricas de evaluación (RMSE/MAE/R²) e histórico reciente. Replica
    la lógica del Notebook 4. La primera solicitud por ticker entrena
    el modelo; las solicitudes posteriores con otro 'horizonte' sólo
    ejecutan inferencia (son inmediatas)."""
    ticker = ticker.strip().upper()
    try:
        return construir_respuesta_lstm(ticker, horizonte)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno en LSTM Regressor para '{ticker}': {e}")


@app.get("/", tags=["Salud"], include_in_schema=False)
def root():
    return {
        "mensaje": "InvestAI API — visite /docs para la documentación interactiva (Swagger UI)",
    }
