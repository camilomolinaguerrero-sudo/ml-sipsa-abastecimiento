# 5. Conclusiones y limitaciones

## Conclusiones

1. **La base de datos es pertinente y suficiente.** SIPSA-A permite observar de forma directa la
   continuidad del abastecimiento de las centrales mayoristas. El panel de 744.921 corredor-semanas
   supera con amplitud el mínimo de 20 mil observaciones, aunque su tamaño efectivo es menor por la
   dependencia temporal y espacial.
2. **La interrupción tiene estructura aprovechable.** Depende sobre todo del tamaño y la regularidad
   del corredor, de su dinámica reciente (relación en U con el nivel de la semana actual), de la
   estacionalidad (festivos, Semana Santa, fin de año) y de la región. Tiene autocorrelación
   temporal (0,28 a una semana dentro de cada corredor) y espacial (I de Moran 0,25), y los eventos
   de interrupción masiva se agrupan en espacio y tiempo (Knox, p = 0,01).
3. **El modelo base es un punto de partida sólido y honesto.** La regresión logística alcanza AUC
   0,79 en 2025 y 0,78 en departamentos no vistos, muy por encima del Dummy (0,50) y de la
   persistencia (0,58), con probabilidades bien calibradas y sin sobreajuste. Su exactitud (0,77)
   supera por poco la de la clase mayoritaria (0,74), y por eso se evalúa con AUC, AUC-PR, F1 y
   recall.
4. **El límite es la forma del modelo, no los datos.** La curva de aprendizaje se estabiliza hacia
   las 50 mil filas. Los residuos ya no tienen dependencia temporal pero conservan algo de
   dependencia espacial (I = 0,045), lo que orienta el trabajo siguiente.
5. **Hallazgo para la tesis.** Los corredores que nacen en municipios PDET se interrumpen más, pero
   el efecto PDET desaparece al controlar por escala, regularidad y distancia. Las intervenciones
   que consoliden cargas y regularicen despachos atacan la causa de la fragilidad.

## Limitaciones

* **Cobertura.** SIPSA-A solo ve lo que llega a 32 centrales urbanas. No mide mercados locales,
  plazas municipales ni venta directa, que pesan más en los territorios PDET. 210 municipios no
  tienen ningún corredor regular.
* **Procedencia declarada.** El origen lo declara el transportador y puede ser un centro de acopio.
* **Definición del objetivo.** El umbral de 50 % y la ventana de 8 semanas son decisiones del
  investigador. Otros umbrales cambiarían la prevalencia y el desempeño; conviene un análisis de
  sensibilidad en el siguiente entregable.
* **Choques no observados.** Clima, orden público, cierres viales y precios no están en el modelo.
  Explican los picos del paro de 2021 y del confinamiento de 2020.
* **Periodo.** El primer cuatrimestre de 2026 no estaba disponible en el DANE al momento de la
  descarga; la prueba cubre solo 2025.
* **ZOMAC.** No se encontró un listado oficial de municipios ZOMAC en formato abierto y
  estructurado; el análisis territorial se limita a PDET.
* **Ética.** Las predicciones no deben usarse para excluir territorios de mercados o inversiones,
  sino para priorizar apoyo logístico.

## Siguientes pasos

* Modelos no lineales del curso (k-NN, Random Forest, XGBoost, SVM) con el mismo pipeline y
  esquema de validación.
* Variables de vecindad espacial (rezago espacial de interrupciones recientes) y de clima o
  cierres viales (IDEAM, INVÍAS).
* Análisis de sensibilidad del umbral y de la ventana que definen la interrupción.
* Evaluación específica en corredores PDET y hacia la central de Bazurto (Cartagena).

## Referencias de datos

* DANE (2018-2025). *Sistema de Información de Precios y Abastecimiento del Sector Agropecuario,
  componente Abastecimiento de Alimentos (SIPSA-A)*. Microdatos, catálogo 697.
  <https://microdatos.dane.gov.co/index.php/catalog/697>
* Agencia de Renovación del Territorio. *Municipios PDET*. <https://www.datos.gov.co/d/idrk-ba8y>
* DANE. *DIVIPOLA: códigos de municipios*. <https://www.datos.gov.co/d/gdxc-w37w>
* Gestores catastrales de Colombia (geometrías municipales). <https://www.datos.gov.co/d/bhcx-bx97>
