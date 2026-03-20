# Skill: Sales Forecasting

Skill especializada en forecasting de ventas y análisis de competencia para el proyecto DS4B

**Versión**: 1.0.0

---

## Objetivo

Asistente de forecasting para análisis de ventas y datos de competencia.

---

## Entorno

| Variable  | Valor                             |
| --------- | --------------------------------- |
| Notebook  | `notebooks/entrenamiento.ipynb` |
| Conda Env | `Forecasting`                   |

### Paths

| Tipo      | Path                           |
| --------- | ------------------------------ |
| Input     | `../data/raw/entrenamiento/` |
| Processed | `../data/processed/`         |
| Models    | `../models/`                 |

---

## Dataframes

### Ventas (original)

| Variable          | Descripción                   |
| ----------------- | ------------------------------ |
| fecha             | Fecha de la venta              |
| producto_id       | Identificador del producto     |
| nombre            | Nombre del producto            |
| categoria         | Categoría del producto        |
| subcategoria      | Subcategoría del producto     |
| precio_base       | Precio base del producto       |
| es_estrella       | Indica si es producto estrella |
| unidades_vendidas | Cantidad de unidades vendidas  |
| precio_venta      | Precio de venta                |
| ingresos          | Ingresos totales               |

### Competencia (original)

| Variable     | Descripción               |
| ------------ | -------------------------- |
| fecha        | Fecha                      |
| producto_id  | Identificador del producto |
| Amazon       | Precio en Amazon           |
| Decathlon    | Precio en Decathlon        |
| Deporvillage | Precio en Deporvillage     |

### df (unificado y procesado - 440 columnas, 2880 filas)

**Origenes**: Merge de ventas y competencia por fecha y producto_id

#### Variables originales
| Variable          | Descripción                   |
| ----------------- | ------------------------------ |
| fecha             | Fecha de la venta (datetime)   |
| producto_id       | Identificador del producto     |
| nombre            | Nombre del producto            |
| categoria         | Categoría del producto        |
| subcategoria      | Subcategoría del producto     |
| precio_base       | Precio base del producto       |
| es_estrella       | Indica si es producto estrella |
| unidades_vendidas | Cantidad de unidades vendidas  |
| precio_venta      | Precio de venta                |
| ingresos          | Ingresos totales               |

#### Variables temporales
| Variable            | Descripción                              |
| ------------------- | ---------------------------------------- |
| año                | Año de la fecha                          |
| mes                | Mes (1-12)                               |
| dia_mes            | Día del mes                              |
| dia_semana         | Día de la semana (0=lunes)               |
| nombre_dia         | Nombre del día                           |
| semana_del_año     | Semana del año                           |
| trimestre          | Trimestre (1-4)                          |
| dia_del_año        | Día del año                              |
| es_fin_semana      | 1 si es viernes, sábado o domingo        |
| es_festivo         | 1 si es festivo en España                |
| es_black_friday    | 1 si es Black Friday                    |
| es_cyber_monday    | 1 si es Cyber Monday                    |
| es_comienzo_mes    | 1 si es día 1-5 del mes                 |
| es_fin_mes         | 1 si es día 25 o superior                |
| es_primer_lunes_mes| 1 si es el primer lunes del mes         |
| es_rebajas         | 1 si está en temporada de rebajas        |

#### Variables de lags
| Variable      | Descripción                              |
| ------------- | ---------------------------------------- |
| lag_1 a lag_7 | Unidades vendidas del día anterior (1-7) |
| rolling_mean_7| Media móvil de 7 días                    |

#### Variables de precios
| Variable            | Descripción                              |
| ------------------- | ---------------------------------------- |
| descuento_pct       | Porcentaje descuento (precio_venta - precio_base / precio_base) * 100 |
| precio_competencia  | Promedio de precios de competidores       |
| ratio_precio        | precio_venta / precio_competencia         |

#### Variables one-hot (dummies)
| Variable | Descripción |
| -------- | ------------|
| nombre_h_* | Dummys de cada nombre de producto |
| categoria_h_* | Dummys de cada categoría |
| subcategoria_h_* | Dummys de cada subcategoría |

**Nota**: Los datos originales tienen 3552 filas. Tras crear lags (se eliminan 672 filas con nulos), quedan 2880 filas.

---

## Librerías Permitidas

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter
- streamlit
- holidays

---

## Workflows

### 1. Análisis Exploratorio (EDA)

Descripción: Análisis exploratorio de datos de ventas

Pasos:

1. Cargar datos de ventas y competencia desde `../data/raw/entrenamiento/`
2. Verificar formas (shapes) de ambos dataframes
3. Revisar tipos de datos y valores nulos
4. Convertir `fecha` a datetime
5. Análisis descriptivo de ventas (unidades_vendidas, ingresos)
6. Distribución por categoria y subcategoria
7. Identificar productos estrellas (es_estrella)
8. Análisis temporal: ventas por día/semana/mes

---

### 2. Feature Engineering

Descripción: Ingeniería de características para forecasting

Pasos:

1. Extraer características temporales de `fecha`: año, mes, día, día_semana, semana, trimestre
2. Crear lags de ventas (lag_7, lag_14, lag_30)
3. Crear medias móviles (rolling_mean_7, rolling_mean_30)
4. Agregar holidays de España (usando library holidays)
5. Características de competencia: merge con dataframe competencia
6. Crear ratios de precio vs competidores

---

### 3. Modelo de Forecasting

Descripción: Entrenamiento de modelo de forecasting

Pasos:

1. Preparar features y target (unidades_vendidas o ingresos)
2. Split temporal (train/test manteniendo orden temporal)
3. Entrenar modelo con scikit-learn (ej: RandomForest, LinearRegression, GradientBoosting)
4. Evaluar con métricas: MAE, RMSE, MAPE
5. Análisis de feature importance
6. Guardar modelo en `../models/`

---

### 4. Análisis de Competencia

Descripción: Análisis de precios de competencia

Pasos:

1. Merge ventas con competencia por fecha y producto_id
2. Comparar precio_venta con Amazon, Decathlon, Deporvillage
3. Calcular diferencia de precios y ratios
4. Identificar oportunidades de pricing
5. Visualizar evolución de precios competitivos

---

## Visualizaciones

### Series Temporales

- Evolución temporal de ventas (matplotlib/seaborn)
- Tendencia por categoria
- Comparación con precios de competencia

### Distribucion

- Histograma de ventas por producto
- Boxplot por categoria
- Heatmap de correlación

---

## Reglas

1. Mantener respuestas concisas (máximo 4 líneas)
2. Responder en español
3. Al escribir codigo en notebooks, SIEMPRE agregar comentarios en español, en primera persona, con una breve descripcion de la funcionalidad de cada bloque de codigo (no linea por linea)
4. Solo usar variables de los dataframes definidos o variables creadas en el código
5. Solo usar librerías: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, holidays
6. No hacer preamble/postamble innecesario

---

## Ejemplos de Tareas

| Tipo          | Descripción                                                         |
| ------------- | -------------------------------------------------------------------- |
| EDA           | Realiza un análisis exploratorio de los datos de ventas             |
| Forecasting   | Entrena un modelo de forecasting para predecir unidades_vendidas     |
| Competition   | Analiza cómo se comparan nuestros precios con los de la competencia |
| Visualization | Crea una visualización de la tendencia de ventas por categoría     |
