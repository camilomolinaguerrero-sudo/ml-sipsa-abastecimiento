# Proyecto ML · Entregable 1: interrupciones en corredores de abastecimiento (SIPSA-A)

Informe en Jupyter Book 1 sobre la predicción de interrupciones semanales en corredores
municipio de origen → central mayorista, con microdatos SIPSA-A del DANE (2018-2025).

## Estructura

```
proyecto_sipsa/
├── requirements.txt                  dependencias con versiones exactas (Python 3.11)
├── .vscode/                          intérprete, extensiones recomendadas y tareas de VS Code
├── data/
│   ├── raw/                          PDET, DIVIPOLA y geometrías municipales (datos.gov.co)
│   ├── interim/sipsa_a_envios.parquet  15,5 M cargas consolidadas 2018-2025
│   └── processed/panel_corredores.parquet  744.921 corredor-semanas (panel de modelado)
├── src/
│   ├── 01_descarga_consolidacion.py  descarga los 18 ZIP del DANE y consolida
│   ├── 02_construir_panel.py         panel semanal, variable objetivo y predictoras
│   ├── utils.py                      rutas, semilla, partición, estadística espacial, mapas
│   ├── py2nb.py                      convierte cuadernos/*.py en .ipynb y los ejecuta
│   └── ver_salidas.py                imprime salidas de texto de un cuaderno
├── cuadernos/                        fuente de los cuadernos (formato # %%)
└── book/                             Jupyter Book: _config.yml, _toc.yml, intro, 4 cuadernos, conclusiones
    └── _build/html/                  sitio HTML generado
```

## Abrir en VS Code

1. `code "proyecto_sipsa"` (o *File → Open Folder*).
2. VS Code recomienda las extensiones Python, Jupyter y MyST (`.vscode/extensions.json`).
3. Abrir cualquier cuaderno de `book/` y elegir el kernel **Python (ml-sipsa)**; *Run All* lo ejecuta.
4. `Terminal → Run Task` ofrece: descargar datos, construir el panel, construir el Jupyter Book y
   abrir el libro. `Cmd+Shift+B` construye el libro.

## Reproducir desde la terminal

```bash
python3.11 -m venv ~/.venvs/ml-sipsa-jb
~/.venvs/ml-sipsa-jb/bin/pip install -r requirements.txt
~/.venvs/ml-sipsa-jb/bin/python -m ipykernel install --user --name ml-sipsa --display-name "Python (ml-sipsa)"
~/.venvs/ml-sipsa-jb/bin/python src/01_descarga_consolidacion.py --zips data/raw/zips
~/.venvs/ml-sipsa-jb/bin/python src/02_construir_panel.py
for n in 01_base_datos 02_eda_general 03_eda_temporal_espacial 04_modelo_base; do
  ~/.venvs/ml-sipsa-jb/bin/python src/py2nb.py cuadernos/$n.py book/$n.ipynb --ejecutar
done
~/.venvs/ml-sipsa-jb/bin/jupyter-book build book/
open book/_build/html/index.html
```

El libro usa **Jupyter Book 1** (`book/_config.yml`, `book/_toc.yml`), igual que el tutorial del
curso. Para publicarlo en GitHub Pages: `ghp-import -n -p -f book/_build/html`.

## Datos

* DANE, SIPSA-A, microdatos catálogo 697: <https://microdatos.dane.gov.co/index.php/catalog/697>
  (también en datos.gov.co, recurso `ymnp-apvk`). El archivo 2026-I no estaba disponible al descargar.
* Municipios PDET: <https://www.datos.gov.co/d/idrk-ba8y>
* DIVIPOLA con coordenadas: <https://www.datos.gov.co/d/gdxc-w37w>
* Geometrías municipales: <https://www.datos.gov.co/d/bhcx-bx97>
