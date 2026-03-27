"""
Streamlit App - Simulación de Predicciones de Ventas Noviembre 2025
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Simulación Ventas Noviembre 2025",
    page_icon="📊",
    layout="wide"
)

# Estilos personalizados
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .kpi-title {
        font-size: 14px;
        color: #666;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: bold;
        color: #667eea;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #333;
        margin-bottom: 15px;
        border-bottom: 2px solid #667eea;
        padding-bottom: 8px;
    }
    .black-friday-row {
        background-color: #ffe6e6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelo y datos


@st.cache_data
def load_model():
    try:
        model = joblib.load('models/modelo_final.joblib')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/inferencia_df_transformado.csv')
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# Función para realizar predicciones recursivas


def predict_recursive(df_producto, model, descuento_pct, escenario_competencia):
    """
    Realiza predicciones recursivas día a día, actualizando los lags después de cada predicción.
    """
    df = df_producto.copy()

    # Ordenar por fecha
    df = df.sort_values('fecha').reset_index(drop=True)

    # Aplicar descuento
    df['precio_venta'] = df['precio_base'] * (1 - descuento_pct / 100)

    # Escenario de competencia
    if escenario_competencia == "Competencia -5%":
        df['Amazon'] = df['Amazon'] * 0.95
        df['Decathlon'] = df['Decathlon'] * 0.95
        df['Deporvillage'] = df['Deporvillage'] * 0.95
    elif escenario_competencia == "Competencia +5%":
        df['Amazon'] = df['Amazon'] * 1.05
        df['Decathlon'] = df['Decathlon'] * 1.05
        df['Deporvillage'] = df['Deporvillage'] * 1.05

    # Recalcular columnas dependientes del precio
    df['precio_competencia'] = df[[
        'Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)
    df['descuento_porcentaje'] = (
        (df['precio_base'] - df['precio_venta']) / df['precio_base']) * 100
    df['ratio_precio'] = df['precio_venta'] / df['precio_competencia']

    # Obtener columnas de features que espera el modelo
    feature_cols = list(model.feature_names_in_)

    # Determinar nombres de columnas de lags y media móvil
    # Buscar columnas de lag
    lag_cols = {}
    ma_col = None
    for col in df.columns:
        if 'lag' in col.lower() and 'unidades' in col.lower():
            # Obtener el número del lag
            num = col.split('_')[-1]
            lag_cols[int(num)] = col
        elif 'ma7' in col.lower() or ('rolling' in col.lower() and '7' in col):
            ma_col = col

    # También buscar sin prefijo unidades_vendidas
    if not lag_cols:
        for col in df.columns:
            if col.startswith('lag_'):
                num = col.replace('lag_', '')
                lag_cols[int(num)] = col

    if not ma_col:
        for col in df.columns:
            if 'rolling' in col.lower() or col == 'rolling_mean_7':
                ma_col = col
                break

    # Predicciones recursivas
    predicciones = []

    for i in range(len(df)):
        row = df.iloc[i:i+1][feature_cols].copy()

        # Hacer predicción
        pred = model.predict(row)[0]
        pred = max(0, pred)  # No permitir valores negativos
        predicciones.append(pred)

        # Actualizar lags para el siguiente día (si no es el último día)
        if i < len(df) - 1 and lag_cols:
            # Obtener valores actuales de lags
            lag_values = {}
            for lag_num in sorted(lag_cols.keys()):
                lag_values[lag_num] = df.iloc[i][lag_cols[lag_num]]

            # Actualizar lags en el siguiente registro (desplazar hacia atrás)
            for lag_num in sorted(lag_cols.keys(), reverse=True):
                if lag_num == 1:
                    df.iloc[i+1, df.columns.get_loc(lag_cols[lag_num])] = pred
                else:
                    df.iloc[i+1, df.columns.get_loc(lag_cols[lag_num])
                            ] = lag_values[lag_num - 1]

            # Actualizar media móvil de 7 días
            if ma_col:
                ultimas_7 = predicciones[-7:] if len(
                    predicciones) >= 7 else predicciones
                df.iloc[i+1, df.columns.get_loc(ma_col)] = np.mean(ultimas_7)

    df['unidades_predichas'] = predicciones
    df['ingresos_proyectados'] = df['unidades_predichas'] * df['precio_venta']

    return df

# Función para convertir fecha


def convertir_fecha(fecha):
    """Convierte la fecha a formato datetime"""
    if isinstance(fecha, str):
        return pd.to_datetime(fecha)
    return fecha


# ============================================================
# SIDEBAR - Controles de Simulación
# ============================================================
st.sidebar.title("🎮 Controles de Simulación")
st.sidebar.markdown("---")

# Título en sidebar
st.sidebar.header("📊 Configuración")

# Cargar datos
df_inferencia = load_data()
modelo = load_model()

if df_inferencia is None or modelo is None:
    st.error(
        "No se pudieron cargar los datos o el modelo. Por favor verifica que existen los archivos.")
    st.stop()

# Obtener lista de productos (necesitamos los nombres pero están en el CSV original)
# Primero verificamos qué columnas tenemos
columnas = df_inferencia.columns.tolist()

# Buscar columnas de nombre de producto (nombre_h_)
nombre_cols = [col for col in columnas if col.startswith('nombre_h_')]
productos = [col.replace('nombre_h_', '') for col in nombre_cols]

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "👟 Seleccionar Producto",
    options=productos,
    index=0
)

# Slider de descuento
descuento = st.sidebar.slider(
    "🏷️ Ajuste de Descuento",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%+d%%"
)

# Selector de escenario de competencia
escenario = st.sidebar.radio(
    "🏪 Escenario Competencia",
    options=["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0
)

# Botón de simulación
st.sidebar.markdown("---")
if st.sidebar.button("🚀 Simular Ventas", type="primary", use_container_width=True):
    st.session_state['simular'] = True
    st.session_state['producto'] = producto_seleccionado
    st.session_state['descuento'] = descuento
    st.session_state['escenario'] = escenario

# ============================================================
# ZONA PRINCIPAL - Dashboard
# ============================================================

# Header
st.title("📈 Dashboard de Predicciones de Ventas")
st.subheader(f"🗓️ Noviembre 2025 | Producto: **{producto_seleccionado}**")
st.markdown("---")

# Verificar si hay simulación activa
if 'simular' not in st.session_state or not st.session_state.get('simular', False):
    st.info("👈 Configura los parámetros en el sidebar y haz clic en 'Simular Ventas' para comenzar.")
    st.stop()

# Obtener datos del producto seleccionado
columna_producto = f'nombre_h_{producto_seleccionado}'

# Verificar si la columna existe
if columna_producto not in df_inferencia.columns:
    st.error(f"Columna {columna_producto} no encontrada en los datos.")
    st.write("Columnas disponibles:", df_inferencia.columns.tolist()[:20])
    st.stop()

# Filtrar por producto
df_producto = df_inferencia[df_inferencia[columna_producto] == 1].copy()

if len(df_producto) == 0:
    st.error(
        f"No se encontraron datos para el producto: {producto_seleccionado}")
    st.stop()

# Asegurar que fecha es datetime
df_producto['fecha'] = pd.to_datetime(df_producto['fecha'])

# Realizar predicción con spinner
with st.spinner('🔮 Realizando predicciones recursivas...'):
    df_resultado = predict_recursive(
        df_producto,
        modelo,
        st.session_state['descuento'],
        st.session_state['escenario']
    )

# ============================================================
# KPIs
# ============================================================
st.markdown('<p class="section-title">📊 KPIs Destacados</p>',
            unsafe_allow_html=True)

# Calcular KPIs
unidades_totales = df_resultado['unidades_predichas'].sum()
ingresos_totales = df_resultado['ingresos_proyectados'].sum()
precio_promedio = df_resultado['precio_venta'].mean()
descuento_promedio = df_resultado['descuento_porcentaje'].mean()

# Mostrar KPIs en columnas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📦 Unidades Totales",
        value=f"{unidades_totales:,.0f}",
        delta=None
    )

with col2:
    st.metric(
        label="💰 Ingresos Proyectados",
        value=f"€{ingresos_totales:,.2f}",
        delta=None
    )

with col3:
    st.metric(
        label="🏷️ Precio Promedio",
        value=f"€{precio_promedio:.2f}",
        delta=None
    )

with col4:
    st.metric(
        label="📉 Descuento Promedio",
        value=f"{descuento_promedio:.1f}%",
        delta=None
    )

st.markdown("---")

# ============================================================
# Gráfico de Predicción Diaria
# ============================================================
st.markdown('<p class="section-title">📈 Predicción Diaria de Ventas</p>',
            unsafe_allow_html=True)

# Crear gráfico con Seaborn
fig, ax = plt.subplots(figsize=(14, 6))

# Datos para el gráfico
dias = range(1, len(df_resultado) + 1)
unidades = df_resultado['unidades_predichas'].values

# Gráfico principal
sns.lineplot(x=dias, y=unidades, ax=ax, color='#667eea',
             linewidth=2.5, marker='o', markersize=6)

# Marcar Black Friday (día 28)
bf_day = 28
if bf_day <= len(df_resultado):
    bf_value = df_resultado[df_resultado['fecha'].dt.day ==
                            bf_day]['unidades_predichas'].values
    if len(bf_value) > 0:
        ax.axvline(x=bf_day, color='#e74c3c',
                   linestyle='--', linewidth=2, alpha=0.7)
        ax.scatter([bf_day], [bf_value[0]], color='#e74c3c', s=200,
                   zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(f'BLACK FRIDAY\nDía {bf_day}',
                    xy=(bf_day, bf_value[0]),
                    xytext=(bf_day + 2, bf_value[0] + 2),
                    fontsize=11,
                    fontweight='bold',
                    color='#e74c3c',
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

# Personalizar gráfico
ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
ax.set_title(
    f'Predicción Diaria de Ventas - {producto_seleccionado}', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(range(1, 31, 2))
ax.set_xlim(0.5, 30.5)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')

# Fondo alternativo para semanas
for i in range(0, 30, 7):
    ax.axvspan(i+0.5, i+7.5, alpha=0.05, color='gray')

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ============================================================
# Tabla Detallada
# ============================================================
st.markdown('<p class="section-title">📋 Detalle Diario</p>',
            unsafe_allow_html=True)

# Preparar datos para la tabla
df_tabla = df_resultado[['fecha', 'precio_venta', 'precio_competencia',
                         'descuento_porcentaje', 'unidades_predichas', 'ingresos_proyectados']].copy()
df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%d/%m/%Y')
df_tabla['día_semana'] = pd.to_datetime(df_resultado['fecha']).dt.day_name()

# Renombrar columnas
df_tabla = df_tabla.rename(columns={
    'fecha': 'Fecha',
    'día_semana': 'Día',
    'precio_venta': 'Precio Venta (€)',
    'precio_competencia': 'Precio Comp. (€)',
    'descuento_porcentaje': 'Descuento (%)',
    'unidades_predichas': 'Unidades',
    'ingresos_proyectados': 'Ingresos (€)'
})

# Formatear números
df_tabla['Precio Venta (€)'] = df_tabla['Precio Venta (€)'].round(2)
df_tabla['Precio Comp. (€)'] = df_tabla['Precio Comp. (€)'].round(2)
df_tabla['Descuento (%)'] = df_tabla['Descuento (%)'].round(1)
df_tabla['Unidades'] = df_tabla['Unidades'].round(0).astype(int)
df_tabla['Ingresos (€)'] = df_tabla['Ingresos (€)'].round(2)

# Reordenar columnas
df_tabla = df_tabla[[
    'Fecha', 'Día', 'Precio Venta (€)', 'Precio Comp. (€)', 'Descuento (%)', 'Unidades', 'Ingresos (€)']]

# Resaltar Black Friday


def highlight_black_friday(row):
    if row['Fecha'].startswith('28/11'):
        return ['background-color: #ffe6e6; font-weight: bold'] * len(row)
    return [''] * len(row)


# Mostrar tabla
st.dataframe(
    df_tabla.style.apply(highlight_black_friday, axis=1),
    use_container_width=True,
    height=400
)

st.markdown("---")

# ============================================================
# Comparativa de Escenarios
# ============================================================
st.markdown('<p class="section-title">⚖️ Comparativa de Escenarios de Competencia</p>',
            unsafe_allow_html=True)

# Calcular resultados para cada escenario
escenarios = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
resultados_escenarios = {}

for esc in escenarios:
    df_esc = predict_recursive(
        df_producto, modelo, st.session_state['descuento'], esc)
    resultados_escenarios[esc] = {
        'unidades': df_esc['unidades_predichas'].sum(),
        'ingresos': df_esc['ingresos_proyectados'].sum()
    }

# Mostrar tarjetas de comparación
col1, col2, col3 = st.columns(3)

for i, (esc, res) in enumerate(resultados_escenarios.items()):
    with [col1, col2, col3][i]:
        st.metric(
            label=f"🏪 {esc}",
            value=f"{res['unidades']:,.0f} uds",
            delta=f"€{res['ingresos']:,.2f}"
        )

# Información adicional
st.info(
    f"💡 Esta comparativa mantiene el descuento en {st.session_state['descuento']}% y varía solo los precios de competencia.")

# Pie de página
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "📊 <b>Simulador de Predicciones de Ventas</b> | Modelo: HistGradientBoostingRegressor | "
    "Datos: Noviembre 2025"
    "</div>",
    unsafe_allow_html=True
)
