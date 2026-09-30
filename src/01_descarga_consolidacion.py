"""
Paso 1. Descarga y consolidación de los microdatos SIPSA-A (DANE).

Fuente: DANE, Sistema de Información de Precios y Abastecimiento del Sector
Agropecuario, componente Abastecimiento de Alimentos (SIPSA-A), catálogo 697
de microdatos (https://microdatos.dane.gov.co/index.php/catalog/697),
publicado también en datos.gov.co (recurso ymnp-apvk).

Uso:
    python src/01_descarga_consolidacion.py --zips <carpeta_con_zips>

Si la carpeta no contiene los ZIP, el script los descarga. Solo se extraen los
CSV (los ZIP traen además .sav y .dta con el mismo contenido).

Salida: data/interim/sipsa_a_envios.parquet (un registro por envío/carga).
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "interim" / "sipsa_a_envios.parquet"

# Identificador de descarga -> periodo publicado por el DANE
DESCARGAS = {
    20281: "2018-S1", 20282: "2018-S2", 20283: "2019-S1", 20284: "2019-S2",
    20285: "2020-S1", 20286: "2020-S2", 20536: "2021-S1", 21267: "2021-S2",
    21418: "2022-S1", 22198: "2022-S2", 22886: "2023-S1", 23116: "2023-S2",
    23637: "2024-C1", 23638: "2024-C2", 23771: "2024-C3", 24171: "2025-C1",
    24294: "2025-C2", 24413: "2025-C3",
}
URL = "https://microdatos.dane.gov.co/index.php/catalog/697/download/{}"

# Los encabezados cambian entre años; se normalizan por posición semántica.
MAPA_COLUMNAS = {
    "fuente": "mercado", "cuidad, mercado mayorista": "mercado",
    "ciudad, mercado mayorista": "mercado",
    "fechaencuesta": "fecha", "fecha": "fecha",
    "cod. depto proc.": "cod_depto", "código departamento": "cod_depto",
    "codigo departamento": "cod_depto",
    "cod. municipio proc.": "cod_mpio", "código municipio": "cod_mpio",
    "codigo municipio": "cod_mpio",
    "departamento proc.": "depto_origen", "municipio proc.": "mpio_origen",
    "grupo": "grupo", "ali": "alimento", "alimento": "alimento",
    "divipola depto proc.": "cod_depto",
    "divipola municipio / iso 3166-1 país proc.": "cod_mpio",
    "departamento": "depto_origen", "municipio de colombia / país proc.": "mpio_origen",
    "cant kg": "kg", "codigo cpc": "cod_cpc", "código cpc": "cod_cpc",
}


def descargar(carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    for ident in DESCARGAS:
        destino = carpeta / f"{ident}.zip"
        if destino.exists() and destino.stat().st_size > 0:
            continue
        print("Descargando", ident)
        urllib.request.urlretrieve(URL.format(ident), destino)


def leer_zip(ruta: Path, periodo: str) -> pd.DataFrame:
    with zipfile.ZipFile(ruta) as z:
        nombre = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        bruto = z.read(nombre)
    try:  # la mayoría de archivos viene en latin-1; algunos (2024-C3) en UTF-8 con BOM
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = bruto.decode("latin-1").lstrip("\ufeffï»¿")
    df = pd.read_csv(io.StringIO(texto), sep=";", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower().lstrip("\ufeff") for c in df.columns]
    df = df[[c for c in df.columns if c in MAPA_COLUMNAS]].rename(columns=MAPA_COLUMNAS)
    df["periodo"] = periodo
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    # Algunos CSV traen filas completamente vacías al final (solo separadores)
    vacias = (df.drop(columns="periodo") == "").all(axis=1)
    print("Filas vacías eliminadas:", int(vacias.sum()))
    df = df[~vacias].copy()
    for c in ["mercado", "depto_origen", "mpio_origen", "grupo", "alimento"]:
        df[c] = df[c].str.strip()
    for c in ["cod_depto", "cod_mpio"]:
        df[c] = df[c].str.replace(r"[\s']", "", regex=True)
    df["cod_depto"] = df["cod_depto"].str.zfill(2)
    df["cod_mpio"] = df["cod_mpio"].str.zfill(5)
    # Cantidades en formato español: '.' separa miles y ',' decimales.
    kg = df["kg"].str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    df["kg"] = pd.to_numeric(kg, errors="coerce")
    df["fecha"] = pd.to_datetime(df["fecha"].str.strip(), format="%d/%m/%Y", errors="coerce")
    df = df.drop(columns=[c for c in ["cod_cpc"] if c in df.columns])
    for c in ["mercado", "cod_depto", "cod_mpio", "depto_origen", "mpio_origen",
              "grupo", "alimento", "periodo"]:
        df[c] = df[c].astype("category")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", type=Path, default=ROOT / "data" / "raw" / "zips")
    args = ap.parse_args()
    descargar(args.zips)
    partes = []
    for ident, periodo in DESCARGAS.items():
        df = leer_zip(args.zips / f"{ident}.zip", periodo)
        print(periodo, df.shape)
        partes.append(df)
    df = limpiar(pd.concat(partes, ignore_index=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print("Guardado", OUT, df.shape)


if __name__ == "__main__":
    main()
