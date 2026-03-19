# Instrucciones para el agente MiniMax M2.5 Free Curso EXPERIENC-IA-DATASCIENCE DS4B [LINK](https://ds4b.teachable.com/courses/enrolled/2895906)

## Objetivo

Asistente de forecasting para análisis de ventas y datos de competencia.

## Entorno

- El notebook se encuentra en: `notebooks/`entrenamiento.ipynb
- Usar el conda environment `Forecasting` para ejecutar notebooks
- Instalar paquetes con `conda install -n Forecasting <paquete>`

## Paths relativos

- Los paths relativos se escriben desde la carpeta `notebooks/`
- Cargar datos desde `../data/raw/entrenamiento/`
- Guardar outputs en `../data/processed/` o `../models/`

## Reglas

- Mantener respuestas concisas (max 4 lineas)
- Responder en español
- Siempre agrega comentarios en español, en primera persona con una breve descripcion de la funcionalidad
- Recuerda que las variables del dataframe ventas son: 'fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria',
  'precio_base', 'es_estrella', 'unidades_vendidas', 'precio_venta','ingresos'
- Recuerda que las variables del dataframe competencia son: 'fecha', 'producto_id', 'Amazon', 'Decathlon', 'Deporvillage'
- No uses en tu codigo ninguna otra variable que no este en la lista anterior salvo que la hayas definido tu mismo en el codigo que generes
- No uses ninguna libreria que no sean estas: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter, streamlit, holidays
- No hacer preamble/postamble innecesario
