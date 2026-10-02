# Interrupciones en corredores de abastecimiento de alimentos en Colombia

**Primer entregable del proyecto de Machine Learning** · Doctorado en Ingeniería

**Autores:** Camilo Molina Guerrero y Lina Margarita Buelvas

Selección de base de datos, análisis exploratorio e implementación de un modelo base

---

## Contexto

La propuesta doctoral *Arquitectura de gobernanza para la gestión de proyectos de logística
inteligente: optimización multicriterio basada en deep learning para la resiliencia alimentaria en
territorios PDET y ZOMAC* necesita un componente analítico que anticipe riesgos de
desabastecimiento. Este proyecto construye su primera pieza con datos oficiales: el componente de
Abastecimiento de Alimentos del SIPSA (DANE), que registra cada carga que llega a las 32 principales
centrales mayoristas del país con su municipio de procedencia.

## Pregunta

Con la información disponible al cierre de una semana, ¿se puede anticipar si un corredor de
abastecimiento (municipio de origen → central mayorista) sufrirá una **caída de más del 50 %**
frente a su nivel habitual la semana siguiente?

## Resumen de resultados

| Aspecto | Resultado |
|---|---|
| Datos | 15,5 millones de cargas (2018-2025) → 744.921 corredor-semanas, 4.512 corredores, 925 municipios de origen, 32 centrales |
| Objetivo | 28 % de las corredor-semanas presentan interrupción en la semana siguiente |
| Tipo de problema | Clasificación binaria sobre un panel espacio-temporal (ruta A + secciones 2.6-2.8) |
| Partición | Cronológica: entrenamiento hasta octubre de 2024, prueba todo 2025, hueco de 10 semanas |
| Línea base trivial | Dummy: AUC 0,50; exactitud 0,74 (clase mayoritaria) |
| Persistencia | AUC 0,584 |
| Modelo base (regresión logística) | AUC-ROC 0,793 [0,789; 0,797], AUC-PR 0,539, recall 0,76 con umbral 0,26 |
| Territorios no vistos (bloques por departamento) | AUC 0,780 ± 0,010 |

Los corredores pequeños e irregulares son los que más se interrumpen. Los que nacen en municipios
PDET se interrumpen más (33 % frente a 28 %), pero esa diferencia desaparece al controlar por
tamaño, regularidad y distancia: su fragilidad es logística.

## Estructura del informe

1. **Base de datos**: problema, justificación, fuente, licencia, diccionario, estructura, tamaño,
   calidad y ética.
2. **EDA (I)**: reserva del conjunto de prueba, variable objetivo, análisis uni, bi y multivariado,
   y auditoría de fuga.
3. **EDA (II)**: componentes temporal, espacial y espacio-temporal.
4. **Preprocesamiento y modelo base**: pipeline, validación temporal y espacial, métricas con
   intervalos bootstrap, residuos, curva de aprendizaje y coeficientes.
5. **Conclusiones y limitaciones**.

## Reproducibilidad

Entorno: Python 3.11, dependencias en `requirements.txt`, semilla aleatoria global `42`
(`src/utils.py`). Desde la carpeta del proyecto:

```bash
python3.11 -m venv ~/.venvs/ml-sipsa-jb
~/.venvs/ml-sipsa-jb/bin/pip install -r requirements.txt
~/.venvs/ml-sipsa-jb/bin/python -m ipykernel install --user --name ml-sipsa
~/.venvs/ml-sipsa-jb/bin/python src/01_descarga_consolidacion.py   # descarga los 18 ZIP del DANE
~/.venvs/ml-sipsa-jb/bin/python src/02_construir_panel.py          # panel semanal
~/.venvs/ml-sipsa-jb/bin/jupyter-book build book/                  # genera el libro HTML
```

En VS Code, los cuadernos de `book/` usan el kernel **Python (ml-sipsa)** y las tareas
(`Terminal → Run Task`) ejecutan cada paso.
