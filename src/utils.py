"""Funciones compartidas por los cuadernos del proyecto SIPSA-A."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "book" / "figuras"

SEMILLA = 42

# Partición cronológica: el conjunto de prueba son las semanas objetivo desde 2025.
# Se deja un hueco de 8 semanas (la longitud de la ventana de referencia) para que
# ninguna ventana de prueba comparta semanas con la última semana objetivo de entrenamiento.
FIN_TRAIN = pd.Timestamp("2024-10-27")   # última semana t+1 (inicio de semana) en entrenamiento
INICIO_TEST = pd.Timestamp("2025-01-06")  # primera semana t+1 en prueba

VAR_OBJETIVO = "interrupcion_t1"

NUMERICAS = ["log_base8", "ratio_t", "ratio_t_1", "cv8", "pend8_rel", "activas8", "activas26",
             "n_envios8", "n_alimentos8", "antig_sem", "dist_km", "frac_caida_mercado_t",
             "frac_caida_nacional_t", "festivos_t1", "sem_sin", "sem_cos",
             "sh_verduras", "sh_frutas", "sh_tuberculos", "sh_granos", "sh_lacteos",
             "sh_carnes", "sh_pescados", "sh_procesados"]
BINARIAS = ["mismo_depto", "local", "pdet_origen"]
CATEGORICAS = ["mercado", "depto_origen", "grupo_dom"]


def estilo():
    plt.rcParams.update({
        "figure.figsize": (9, 4.5), "figure.dpi": 110, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 10,
    })


def cargar_panel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "panel_corredores.parquet")


def particion(p: pd.DataFrame):
    """Devuelve (train, test) según la semana objetivo t+1."""
    train = p[p["semana_t1"] <= FIN_TRAIN].copy()
    test = p[p["semana_t1"] >= INICIO_TEST].copy()
    return train, test


# ---------------------------------------------------------------- espacial
def pesos_knn(lat, lon, k=8):
    """Matriz de pesos k vecinos más cercanos (haversine), estandarizada por filas."""
    X = np.radians(np.column_stack([lat, lon]))
    tree = BallTree(X, metric="haversine")
    _, idx = tree.query(X, k=k + 1)
    idx = idx[:, 1:]
    n = len(lat)
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), idx.ravel()] = 1.0 / k
    return W


def moran_i(y, W, permutaciones=999, semilla=SEMILLA):
    """I de Moran global con prueba de permutación."""
    y = np.asarray(y, float)
    z = y - y.mean()
    n, s0 = len(y), W.sum()
    I = n / s0 * (z @ W @ z) / (z @ z)
    rng = np.random.default_rng(semilla)
    sim = np.empty(permutaciones)
    for i in range(permutaciones):
        zp = rng.permutation(z)
        sim[i] = n / s0 * (zp @ W @ zp) / (zp @ zp)
    p = (np.sum(sim >= I) + 1) / (permutaciones + 1) if I >= sim.mean() else \
        (np.sum(sim <= I) + 1) / (permutaciones + 1)
    return I, p, -1 / (n - 1)


def lisa(y, W, permutaciones=999, semilla=SEMILLA):
    """I de Moran local (LISA) con p-valores por permutación condicional."""
    y = np.asarray(y, float)
    z = (y - y.mean()) / y.std()
    lag = W @ z
    Ii = z * lag
    rng = np.random.default_rng(semilla)
    n = len(y)
    k = (W > 0).sum(axis=1)
    p = np.empty(n)
    for i in range(n):
        otros = np.delete(z, i)
        sims = np.array([otros[rng.choice(n - 1, k[i], replace=False)].mean()
                         for _ in range(permutaciones)]) * z[i]
        p[i] = (np.sum(np.abs(sims) >= abs(Ii[i])) + 1) / (permutaciones + 1)
    cuadrante = np.where(z > 0, np.where(lag > 0, "Alto-Alto", "Alto-Bajo"),
                         np.where(lag > 0, "Bajo-Alto", "Bajo-Bajo"))
    return Ii, p, cuadrante


def getis_ord_gi_star(y, lat, lon, k=8):
    """Estadístico Gi* de Getis-Ord (puntaje z) con pesos binarios kNN incluyendo el propio punto."""
    y = np.asarray(y, float)
    X = np.radians(np.column_stack([lat, lon]))
    _, idx = BallTree(X, metric="haversine").query(X, k=k + 1)
    n = len(y)
    xbar, s = y.mean(), np.sqrt((y ** 2).mean() - y.mean() ** 2)
    w = k + 1
    num = y[idx].sum(axis=1) - xbar * w
    den = s * np.sqrt((n * w - w ** 2) / (n - 1))
    return num / den


# ---------------------------------------------------------------- mapas
def _poligonos(geojson_path=RAW / "municipios_geom.geojson"):
    with open(geojson_path) as fh:
        gj = json.load(fh)
    polys = []
    for f in gj["features"]:
        cod = str(f["properties"]["mpcodigo"]).zfill(5)
        geom = f["geometry"]
        partes = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for parte in partes:
            polys.append((cod, np.asarray(parte[0])))
    return polys


_POLYS = None


def mapa_base(ax, color="#e9ecef", borde="#adb5bd"):
    global _POLYS
    if _POLYS is None:
        _POLYS = _poligonos()
    pc = PatchCollection([Polygon(p, closed=True) for _, p in _POLYS], facecolor=color,
                         edgecolor=borde, linewidths=0.15)
    ax.add_collection(pc)
    ax.set_xlim(-79.5, -66.5)
    ax.set_ylim(-4.5, 13.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax


def coropleta(ax, valores: pd.Series, cmap="viridis", vmin=None, vmax=None, sin_dato="#f1f3f5"):
    """Coropleta municipal. `valores` indexado por código DIVIPOLA de 5 dígitos."""
    global _POLYS
    if _POLYS is None:
        _POLYS = _poligonos()
    cm = plt.get_cmap(cmap)
    vmin = valores.min() if vmin is None else vmin
    vmax = valores.max() if vmax is None else vmax
    colores = []
    for cod, _ in _POLYS:
        v = valores.get(cod, np.nan)
        colores.append(sin_dato if pd.isna(v) else cm((v - vmin) / (vmax - vmin + 1e-12)))
    pc = PatchCollection([Polygon(p, closed=True) for _, p in _POLYS], facecolor=colores,
                         edgecolor="#ced4da", linewidths=0.1)
    ax.add_collection(pc)
    ax.set_xlim(-79.5, -66.5)
    ax.set_ylim(-4.5, 13.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin, vmax))
    return sm


# ---------------------------------------------------------------- estadística
def cramers_v(tabla: pd.DataFrame) -> float:
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(tabla, correction=False)[0]
    n = tabla.to_numpy().sum()
    r, k = tabla.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def bootstrap_bloques(y, s, semanas, metrica, B=500, semilla=SEMILLA):
    """Intervalo bootstrap por bloques de semana (remuestrea semanas completas)."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame({"y": y, "s": s, "w": semanas})
    grupos = {w: g.index.to_numpy() for w, g in df.groupby("w")}
    claves = np.array(list(grupos))
    vals = []
    for _ in range(B):
        muestra = rng.choice(claves, len(claves), replace=True)
        idx = np.concatenate([grupos[w] for w in muestra])
        yy, ss = df["y"].to_numpy()[idx], df["s"].to_numpy()[idx]
        if yy.min() == yy.max():
            continue
        vals.append(metrica(yy, ss))
    return np.percentile(vals, [2.5, 97.5])


# ---------------------------------------------------------------- modelado
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada columna en los cuantiles [inferior, superior] aprendidos en fit (solo train)."""

    def __init__(self, inferior=0.01, superior=0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lim_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.lim_sup_ = np.nanquantile(X, self.superior, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lim_inf_, self.lim_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class BloquesTemporales:
    """Validación cruzada de ventana creciente sobre semanas completas, con hueco entre
    el bloque de entrenamiento y el de validación. Requiere la columna `semana_t1` en X."""

    def __init__(self, n_splits=5, gap_semanas=8, columna="semana_t1"):
        self.n_splits = n_splits
        self.gap_semanas = gap_semanas
        self.columna = columna

    def split(self, X, y=None, groups=None):
        semanas = pd.to_datetime(X[self.columna]).to_numpy()
        unicas = np.unique(semanas)
        tss = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap_semanas)
        for tr_w, va_w in tss.split(unicas):
            yield (np.flatnonzero(np.isin(semanas, unicas[tr_w])),
                   np.flatnonzero(np.isin(semanas, unicas[va_w])))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
