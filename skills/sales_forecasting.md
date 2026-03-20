# Skill: Sales Forecasting

Skill especializada en forecasting de ventas y análisis de competencia para el proyecto DS4B

**Versión**: 1.1.0

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

### Archivos

| Archivo              | Descripción                                      |
| -------------------- | ------------------------------------------------ |
| df_procesado.csv     | Dataset unificado y transformado (2880 filas)    |

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
| dia                | Día del mes                              |
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
| descuento_pct       | Porcentaje descuento: ((precio_venta - precio_base) / precio_base) * 100 |
| precio_competencia  | Promedio de precios de competidores (Amazon + Decathlon + Deporvillage) / 3 |
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
8. Análisis temporal: ventas por día/semana/mes con gráficos seaborn

---

### 2. Feature Engineering

Descripción: Ingeniería de características para forecasting

Pasos:

1. Extraer características temporales de `fecha`: año, mes, día, dia_mes, día_semana, semana_del_año, trimestre, dia_del_año
2. Crear variables binarias: es_fin_semana, es_festivo, es_black_friday, es_cyber_monday, es_comienzo_mes, es_fin_mes, es_primer_lunes_mes, es_rebajas
3. Agregar holidays de España usando library `holidays`
4. Crear lags de ventas (lag_1 a lag_7) por producto y año
5. Crear media móvil de 7 días (rolling_mean_7)
6. Merge ventas con competencia por fecha y producto_id
7. Crear descuento_pct, precio_competencia y ratio_precio
8. Aplicar one-hot encoding a nombre, categoria y subcategoria
9. Eliminar registros con nulos generados por los lags
10. Guardar dataset procesado en `../data/processed/df_procesado.csv`

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

1. Comparar precio_venta con precio_competencia
2. Analizar ratio_precio: si es >1 somos más caros, si <1 somos más baratos
3. Analizar descuento_pct para identificar estrategias de pricing
4. Identificar oportunidades de pricing basadas en la diferencia con competidores

---

## Visualizaciones

### Series Temporales

- Evolución temporal de ventas (matplotlib/seaborn)
- Series por año con marcadores de Black Friday
- Tendencia por categoria
- Comparación con precios de competencia

### Distribucion

- Histograma de ventas por producto
- Barplot por dia de semana, categoria y subcategoria
- Boxplot por categoria
- KDE de densidades de precios

---

## Reglas

1. Mantener respuestas concisas (máximo 4 líneas)
2. Responder en español
3. Al escribir codigo en notebooks, SIEMPRE agregar comentarios en español, en primera persona, con una breve descripcion de la funcionalidad de cada bloque de codigo (no linea por linea)
4. Solo usar variables de los dataframes definidos o variables creadas en el código
5. Solo usar librerías: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit, holidays
6. No hacer preamble/postamble innecesario
7. Al guardar datasets o modelos usar paths desde `notebooks/`: `../data/processed/` o `../models/`

---

## Ejemplos de Tareas

| Tipo          | Descripción                                                         |
| ------------- | -------------------------------------------------------------------- |
| EDA           | Realiza un análisis exploratorio de los datos de ventas             |
| Forecasting   | Entrena un modelo de forecasting para predecir unidades_vendidas     |
| Competition   | Analiza cómo se comparan nuestros precios con los de la competencia |
| Visualization | Crea una visualización de la tendencia de ventas por categoría       |
