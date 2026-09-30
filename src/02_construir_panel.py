"""
Paso 2. Construcción del panel semanal de corredores de abastecimiento.

Unidad de observación: corredor (mercado mayorista destino, municipio de
origen) en la semana t (lunes a domingo). Cada fila contiene variables
calculadas SOLO con información disponible al cierre de la semana t y la
variable objetivo, que describe la semana t+1.

Variable objetivo (interrupcion_t1):
    1 si el abastecimiento del corredor en la semana t+1, medido en kg por día
    de encuesta del mercado, cae por debajo del 50 % de su promedio de las 8
    semanas previas (t-7..t); 0 en otro caso.

Población en riesgo (filas elegibles):
    - el mercado fue encuestado en t+1 (si no, no hay dato, no hay interrupción);
    - el corredor tuvo abastecimiento en al menos 4 de las 8 semanas previas
      (corredor regular) y su promedio de referencia es positivo.

Salida: data/processed/panel_corredores.parquet
"""
import unicodedata
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ENVIOS = ROOT / "data" / "interim" / "sipsa_a_envios.parquet"
OUT = ROOT / "data" / "processed" / "panel_corredores.parquet"

VENTANA = 8        # semanas de referencia
MIN_ACTIVAS = 4    # semanas con abastecimiento para considerar el corredor regular
UMBRAL = 0.5       # caída relativa que define la interrupción

# Nombres de mercado escritos de dos formas en distintos años
ALIAS_MERCADO = {"Cali, Santa Helena": "Cali, Santa Elena", "Pereira, La 41-Impala": "Pereira, La 41"}

GRUPOS = {
    "VERDURAS Y HORTALIZAS": "verduras", "FRUTAS": "frutas",
    "TUBERCULOS, RAICES Y PLATANOS": "tuberculos", "GRANOS Y CEREALES": "granos",
    "LACTEOS Y HUEVOS": "lacteos", "CARNES": "carnes", "PESCADOS": "pescados",
    "PROCESADOS": "procesados",
}


def sin_tildes(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.upper().replace(",", " ").replace(".", " ").split())


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def cargar_divipola() -> pd.DataFrame:
    d = pd.read_csv(RAW / "divipola_municipios.csv", dtype=str)
    d["lat"] = d["latitud"].str.replace(",", ".").astype(float)
    d["lon"] = d["longitud"].str.replace(",", ".").astype(float)
    d["cod_mpio"] = d["cod_mpio"].str.zfill(5)
    d["nombre_norm"] = d["nom_mpio"].map(sin_tildes)
    return d[["cod_mpio", "nom_mpio", "dpto", "nombre_norm", "lat", "lon"]]


def municipio_mercado(mercados: pd.Series, divi: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada mercado el código DIVIPOLA de la ciudad donde opera."""
    especiales = {"BOGOTA": "11001", "CARTAGENA": "13001", "IPIALES": "52356",
                  "TIBASOSA": "15806", "SANTA MARTA": "47001", "FLORENCIA": "18001",
                  "CALI": "76001", "CUCUTA": "54001"}
    filas = []
    for m in mercados.unique():
        ciudad = sin_tildes(str(m).split(",")[0].split("(")[0])
        cod = especiales.get(ciudad)
        if cod is None:
            cand = divi[divi["nombre_norm"] == ciudad]
            # Si hay homónimos, se prefiere la capital (código terminado en 001)
            cand = cand.sort_values("cod_mpio", key=lambda s: ~s.str.endswith("001"))
            cod = cand["cod_mpio"].iloc[0] if len(cand) else None
        filas.append({"mercado": m, "cod_mpio_mercado": cod})
    out = pd.DataFrame(filas).merge(
        divi.rename(columns={"cod_mpio": "cod_mpio_mercado", "lat": "lat_m", "lon": "lon_m",
                             "dpto": "depto_mercado"})[["cod_mpio_mercado", "lat_m", "lon_m",
                                                        "depto_mercado"]],
        on="cod_mpio_mercado", how="left")
    return out


def pendiente(y: np.ndarray) -> float:
    """Pendiente de mínimos cuadrados de una ventana, ignorando NaN."""
    m = ~np.isnan(y)
    if m.sum() < 3:
        return np.nan
    x = np.arange(len(y))[m]
    yy = y[m]
    xc = x - x.mean()
    return float((xc * (yy - yy.mean())).sum() / (xc ** 2).sum())


def main() -> None:
    env = pd.read_parquet(ENVIOS)
    env = env[env["kg"].notna() & (env["kg"] > 0) & env["fecha"].notna()].copy()
    env["mercado"] = env["mercado"].astype(str).replace(ALIAS_MERCADO)
    divi = cargar_divipola()
    # Solo corredores nacionales: se excluyen importaciones (códigos de país o 'n.a.')
    env = env[env["cod_mpio"].astype(str).isin(set(divi["cod_mpio"]))].copy()
    env["semana"] = env["fecha"].dt.to_period("W-SUN").dt.start_time
    env["grupo_c"] = env["grupo"].astype(str).map(GRUPOS)

    pdet = pd.read_csv(RAW / "pdet_municipios.csv", dtype=str)
    pdet_set = set(pdet["cod_muni"].str.zfill(5))

    # Días de encuesta por mercado y semana (0 = mercado no encuestado)
    dias = env.groupby(["mercado", "semana"], observed=True)["fecha"].nunique().rename("dias_enc")

    # Agregado por corredor-semana
    g = env.groupby(["mercado", "cod_mpio", "semana"], observed=True)
    agg = g.agg(kg=("kg", "sum"), n_envios=("kg", "size"), n_alimentos=("alimento", "nunique"))
    kg_grupo = (env.pivot_table(index=["mercado", "cod_mpio", "semana"], columns="grupo_c",
                                values="kg", aggfunc="sum", observed=True).fillna(0.0))
    agg = agg.join(kg_grupo).reset_index()
    agg["mercado"] = agg["mercado"].astype(str)
    agg["cod_mpio"] = agg["cod_mpio"].astype(str)

    semanas = pd.date_range(env["semana"].min(), env["semana"].max(), freq="W-MON")
    dias = dias.reset_index()
    dias["mercado"] = dias["mercado"].astype(str)
    dias_w = dias.pivot(index="semana", columns="mercado", values="dias_enc").reindex(semanas).fillna(0)

    grupos_cols = [c for c in GRUPOS.values() if c in agg.columns]
    filas = []
    for (mercado, mpio), d in agg.groupby(["mercado", "cod_mpio"], sort=False):
        inicio = d["semana"].min()
        idx = semanas[semanas >= inicio]
        d = d.set_index("semana").reindex(idx)
        dd = dias_w.loc[idx, mercado].to_numpy()
        kg = d["kg"].fillna(0.0).to_numpy(copy=True)
        kg[dd == 0] = np.nan                       # semana sin encuesta: dato ausente
        tasa = kg / np.where(dd > 0, dd, np.nan)   # kg por día de encuesta
        s = pd.Series(tasa, index=idx)
        base = s.rolling(VENTANA, min_periods=MIN_ACTIVAS).mean()
        std = s.rolling(VENTANA, min_periods=MIN_ACTIVAS).std()
        activas8 = (s > 0).astype(float).where(s.notna()).rolling(VENTANA, min_periods=1).sum()
        activas26 = (s > 0).astype(float).where(s.notna()).rolling(26, min_periods=1).mean()
        pend = s.rolling(VENTANA, min_periods=MIN_ACTIVAS).apply(pendiente, raw=True)
        tasa_t1 = s.shift(-1)
        dias_t1 = pd.Series(dd, index=idx).shift(-1)
        n_env = d["n_envios"].fillna(0).where(s.notna()).rolling(VENTANA, min_periods=1).mean()
        n_ali = d["n_alimentos"].fillna(0).where(s.notna()).rolling(VENTANA, min_periods=1).mean()
        kg8 = d[grupos_cols].fillna(0).rolling(VENTANA, min_periods=1).sum()
        share = kg8.div(kg8.sum(axis=1).replace(0, np.nan), axis=0)

        f = pd.DataFrame({
            "mercado": mercado, "cod_mpio": mpio, "semana": idx,
            "tasa_t": s.values, "base8": base.values, "std8": std.values,
            "tasa_t_1": s.shift(1).values, "activas8": activas8.values,
            "activas26": activas26.values, "pend8": pend.values,
            "n_envios8": n_env.values, "n_alimentos8": n_ali.values,
            "antig_sem": np.arange(len(idx)), "dias_t1": dias_t1.values,
            "tasa_t1": tasa_t1.values,
        })
        for c in grupos_cols:
            f["sh_" + c] = share[c].values
        filas.append(f)

    p = pd.concat(filas, ignore_index=True)

    # Elegibilidad y variable objetivo
    p = p[(p["dias_t1"] > 0) & (p["activas8"] >= MIN_ACTIVAS) & (p["base8"] > 0) & p["tasa_t"].notna()]
    p["interrupcion_t1"] = (p["tasa_t1"].fillna(0) < UMBRAL * p["base8"]).astype(int)

    # Variables derivadas (todas con información hasta t)
    p["ratio_t"] = p["tasa_t"] / p["base8"]
    p["ratio_t_1"] = p["tasa_t_1"] / p["base8"]
    p["cv8"] = p["std8"] / p["base8"]
    p["pend8_rel"] = p["pend8"] / p["base8"]
    p["log_base8"] = np.log1p(p["base8"] / 1000)  # toneladas por día

    # Contexto del mercado y nacional en t (fracción de corredores del mercado en caída en t)
    p["caida_t"] = (p["ratio_t"] < UMBRAL).astype(float)
    p["frac_caida_mercado_t"] = p.groupby(["mercado", "semana"])["caida_t"].transform("mean")
    p["frac_caida_nacional_t"] = p.groupby("semana")["caida_t"].transform("mean")

    # Geografía
    divi_o = divi.rename(columns={"lat": "lat_o", "lon": "lon_o", "dpto": "depto_origen",
                                  "nom_mpio": "mpio_origen"})
    p = p.merge(divi_o[["cod_mpio", "mpio_origen", "depto_origen", "lat_o", "lon_o"]],
                on="cod_mpio", how="left")
    mm = municipio_mercado(p["mercado"], divi)
    p = p.merge(mm, on="mercado", how="left")
    p["dist_km"] = haversine(p["lat_o"], p["lon_o"], p["lat_m"], p["lon_m"])
    p["mismo_depto"] = (p["depto_origen"] == p["depto_mercado"]).astype(int)
    p["local"] = (p["cod_mpio"] == p["cod_mpio_mercado"]).astype(int)
    p["pdet_origen"] = p["cod_mpio"].isin(pdet_set).astype(int)

    # Calendario de la semana t+1 (conocido de antemano)
    sem_t1 = p["semana"] + pd.Timedelta(days=7)
    anios = range(p["semana"].dt.year.min(), p["semana"].dt.year.max() + 2)
    fest = pd.to_datetime(list(holidays.Colombia(years=anios).keys()))
    fest_sem = pd.Series(1, index=fest.to_period("W-SUN").start_time).groupby(level=0).sum()
    p["festivos_t1"] = sem_t1.map(fest_sem).fillna(0).astype(int)
    semana_anio = sem_t1.dt.isocalendar().week.astype(int)
    p["sem_sin"] = np.sin(2 * np.pi * semana_anio / 52.18)
    p["sem_cos"] = np.cos(2 * np.pi * semana_anio / 52.18)
    p["semana_t1"] = sem_t1
    p["grupo_dom"] = p[[c for c in p.columns if c.startswith("sh_")]].idxmax(axis=1).str[3:]

    p = p.drop(columns=["caida_t", "tasa_t_1", "std8", "pend8", "dias_t1"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(OUT, index=False, compression="zstd")
    print(p.shape, p["interrupcion_t1"].mean())


if __name__ == "__main__":
    main()
