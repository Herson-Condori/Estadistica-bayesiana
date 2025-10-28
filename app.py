import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

# Configuración inicial
st.set_page_config(page_title="Paradoja del Falso Positivo e Inferencia Estadística", layout="wide")

# === HEADER ===
st.markdown(
    """
    <div style="text-align: center; padding: 10px; background-color: #f0f8ff; border-radius: 10px; margin-bottom: 20px;">
        <h1>Aplicación Interactiva: Paradoja del Falso Positivo e Inferencia Paramétrica vs No Paramétrica</h1>
        <h3>CURSO: ESTADISTICA BAYESIANA</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# === CARGAR DATASET IRIS ===
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# === FUNCIONES AUXILIARES ===
def bootstrap_ci(data, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])

def download_csv_button(df, filename, label):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label=label, data=csv, file_name=filename, mime='text/csv')

# === PESTAÑAS ===
tab1, tab2 = st.tabs(["Parte 1: Paradoja del Falso Positivo", "Parte 2: Inferencia Paramétrica vs No Paramétrica"])

# ===================================================================
# PARTE 1: PARADOJA DEL FALSO POSITIVO (MEJORADA)
# ===================================================================
with tab1:
    st.header("Parte 1: Paradoja del Falso Positivo")
    
    # Selección de método de entrada
    input_method = st.radio(
        "¿Cómo deseas ingresar los datos?",
        ("Simular datos automáticamente", 
         "Simular datos manualmente (TP, FP, TN, FN)", 
         "Usar dataset Breast Cancer (local)", 
         "Cargar archivo CSV"),
        key="parte1_input"
    )
    
    df_test = None
    prevalencia_input = sensibilidad_input = especificidad_input = None
    
    # === OPCIÓN 1: SIMULACIÓN AUTOMÁTICA ===
    if input_method == "Simular datos automáticamente":
        st.subheader("Parámetros de la simulación")
        col1, col2, col3 = st.columns(3)
        with col1:
            prevalencia_input = st.number_input("Prevalencia (%)", min_value=0.01, max_value=50.0, value=0.3, step=0.1) / 100
        with col2:
            sensibilidad_input = st.number_input("Sensibilidad (%)", min_value=50.0, max_value=100.0, value=99.0, step=0.5) / 100
        with col3:
            especificidad_input = st.number_input("Especificidad (%)", min_value=50.0, max_value=100.0, value=98.0, step=0.5) / 100
        
        n_pob = st.number_input("Tamaño de la población simulada", min_value=100, max_value=100000, value=10000, step=1000)
        
        if st.button("Generar simulación"):
            rng = np.random.default_rng(2025)
            estado_real = rng.binomial(1, prevalencia_input, n_pob)
            prueba_positiva = np.zeros(n_pob, dtype=int)
            
            enfermos = estado_real == 1
            sanos = ~enfermos
            prueba_positiva[enfermos] = rng.binomial(1, sensibilidad_input, enfermos.sum())
            prueba_positiva[sanos] = rng.binomial(1, 1 - especificidad_input, sanos.sum())
            
            df_test = pd.DataFrame({
                'Estado_Real': estado_real,
                'Prueba_Positiva': prueba_positiva
            })
            st.session_state['df_test'] = df_test
            st.session_state['params'] = {
                'prevalencia': prevalencia_input,
                'sensibilidad': sensibilidad_input,
                'especificidad': especificidad_input,
                'source': 'simulacion',
                'n': n_pob
            }
    
    # === OPCIÓN 2: SIMULACIÓN MANUAL (TP, FP, TN, FN) ===
    elif input_method == "Simular datos manualmente (TP, FP, TN, FN)":
        st.subheader("Ingresa los conteos directamente")
        col1, col2 = st.columns(2)
        with col1:
            tp = st.number_input("Verdaderos Positivos (TP)", min_value=0, value=100, step=1)
            fp = st.number_input("Falsos Positivos (FP)", min_value=0, value=50, step=1)
        with col2:
            fn = st.number_input("Falsos Negativos (FN)", min_value=0, value=5, step=1)
            tn = st.number_input("Verdaderos Negativos (TN)", min_value=0, value=9845, step=1)
        
        total = tp + fp + fn + tn
        if total == 0:
            st.warning("Ingresa al menos un caso.")
        else:
            estado_real = np.concatenate([np.ones(tp), np.zeros(fp), np.ones(fn), np.zeros(tn)])
            prueba_positiva = np.concatenate([np.ones(tp), np.ones(fp), np.zeros(fn), np.zeros(tn)])
            
            df_test = pd.DataFrame({
                'Estado_Real': estado_real.astype(int),
                'Prueba_Positiva': prueba_positiva.astype(int)
            })
            st.session_state['df_test'] = df_test
            st.session_state['params'] = {
                'prevalencia': (tp + fn) / total,
                'sensibilidad': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'especificidad': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'source': 'manual',
                'n': total
            }
    
    # === OPCIÓN 3: DATASET BREAST CANCER (LOCAL) ===
    elif input_method == "Usar dataset Breast Cancer (local)":
        st.subheader("Dataset: Breast Cancer Wisconsin (Diagnóstico)")
        st.markdown("""
        Este dataset contiene mediciones de tumores mamarios.  
        - **Estado real**: `M` (maligno = enfermo) vs `B` (benigno = sano)  
        - **Prueba simulada**: se usa la variable `worst concave points` con un umbral clínico.
        """)
        
        try:
            # Cargar dataset local
            df_bc = pd.read_csv("data.csv")
            
            # Mostrar nombres de columnas para depuración
            st.write("📌 Columnas disponibles:")
            st.write(list(df_bc.columns))
            
            # Verificar si existe la columna clave
            target_col = None
            for col in df_bc.columns:
                if 'concave' in col.lower() and 'worst' in col.lower():
                    target_col = col
                    break
            
            if target_col is None:
                st.error("❌ No se encontró ninguna columna con 'worst' y 'concave'. Por favor, verifica el nombre de la columna en tu archivo CSV.")
            else:
                st.success(f"✅ Usando columna: `{target_col}`")
                
                # Estado real: 1 = maligno (enfermo), 0 = benigno (sano)
                if 'diagnosis' not in df_bc.columns:
                    st.error("❌ La columna 'diagnosis' no existe. Asegúrate de que tu CSV tenga esta columna.")
                else:
                    df_bc = df_bc.dropna(subset=['diagnosis', target_col])
                    estado_real = (df_bc['diagnosis'] == 'M').astype(int).values
                    
                    # Simular prueba diagnóstica
                    biomarker = df_bc[target_col].values
                    umbral_default = np.percentile(biomarker, 75)  # Umbral en percentil 75
                    
                    umbral = st.slider(
                        f"Umbral para la prueba diagnóstica ({target_col})",
                        min_value=float(biomarker.min()),
                        max_value=float(biomarker.max()),
                        value=float(umbral_default),
                        step=0.01
                    )
                    
                    prueba_positiva = (biomarker > umbral).astype(int)
                    
                    df_test = pd.DataFrame({
                        'Estado_Real': estado_real,
                        'Prueba_Positiva': prueba_positiva
                    })
                    
                    # Calcular métricas empíricas
                    total = len(df_test)
                    enfermos = df_test['Estado_Real'].sum()
                    sanos = total - enfermos
                    tp = ((df_test['Estado_Real'] == 1) & (df_test['Prueba_Positiva'] == 1)).sum()
                    fp = ((df_test['Estado_Real'] == 0) & (df_test['Prueba_Positiva'] == 1)).sum()
                    tn = ((df_test['Estado_Real'] == 0) & (df_test['Prueba_Positiva'] == 0)).sum()
                    
                    prevalencia_input = enfermos / total
                    sensibilidad_input = tp / enfermos if enfermos > 0 else 0
                    especificidad_input = tn / sanos if sanos > 0 else 0
                    
                    st.session_state['df_test'] = df_test
                    st.session_state['params'] = {
                        'prevalencia': prevalencia_input,
                        'sensibilidad': sensibilidad_input,
                        'especificidad': especificidad_input,
                        'source': 'breast_cancer',
                        'umbral': umbral,
                        'columna': target_col,
                        'n': total
                    }
                    
        except FileNotFoundError:
            st.error("❌ Archivo 'data.csv' no encontrado. Asegúrate de que esté en la misma carpeta que app.py.")
        except Exception as e:
            st.error(f"Error al cargar el dataset: {e}")
    
    # === OPCIÓN 4: CARGAR CSV ===
    else:  # Cargar CSV
        st.subheader("Cargar archivo CSV")
        st.markdown("El archivo debe contener dos columnas: `Estado_Real` (0=sano, 1=enfermo) y `Prueba_Positiva` (0=negativo, 1=positivo)")
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"], key="csv_parte1")
        if uploaded_file is not None:
            try:
                df_test = pd.read_csv(uploaded_file)
                if not {'Estado_Real', 'Prueba_Positiva'}.issubset(df_test.columns):
                    st.error("El CSV debe contener las columnas 'Estado_Real' y 'Prueba_Positiva'")
                    df_test = None
                else:
                    total = len(df_test)
                    enfermos = df_test['Estado_Real'].sum()
                    sanos = total - enfermos
                    tp = ((df_test['Estado_Real'] == 1) & (df_test['Prueba_Positiva'] == 1)).sum()
                    fp = ((df_test['Estado_Real'] == 0) & (df_test['Prueba_Positiva'] == 1)).sum()
                    tn = ((df_test['Estado_Real'] == 0) & (df_test['Prueba_Positiva'] == 0)).sum()
                    prevalencia_input = enfermos / total
                    sensibilidad_input = tp / enfermos if enfermos > 0 else 0
                    especificidad_input = tn / sanos if sanos > 0 else 0
                    st.session_state['df_test'] = df_test
                    st.session_state['params'] = {
                        'prevalencia': prevalencia_input,
                        'sensibilidad': sensibilidad_input,
                        'especificidad': especificidad_input,
                        'source': 'csv',
                        'n': total
                    }
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
    
    # === MOSTRAR TABLA DE DATOS (SI EXISTE) ===
    if 'df_test' in st.session_state:
        df_test = st.session_state['df_test']
        params = st.session_state['params']
        prevalencia_input = params['prevalencia']
        sensibilidad_input = params['sensibilidad']
        especificidad_input = params['especificidad']
        
        st.divider()
        st.subheader("📊 Tabla de Datos (primeras 10 filas)")
        st.dataframe(df_test.head(10))
        
        # Botón para ver más (¡con key único!)
        if len(df_test) > 10:
            if st.button("Mostrar todos los datos", key="show_all_data_part1"):
                st.dataframe(df_test)
        
        # Estadísticas básicas
        st.write("### 📈 Estadísticas Descriptivas")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de casos", len(df_test))
            st.metric("Prevalencia", f"{prevalencia_input:.2%}")
        with col2:
            st.metric("Sensibilidad", f"{sensibilidad_input:.2%}")
            st.metric("Especificidad", f"{especificidad_input:.2%}")
        
        # Cálculos VPP
        total = len(df_test)
        enfermos = df_test['Estado_Real'].sum()
        sanos = total - enfermos
        tp = ((df_test['Estado_Real'] == 1) & (df_test['Prueba_Positiva'] == 1)).sum()
        fp = ((df_test['Estado_Real'] == 0) & (df_test['Prueba_Positiva'] == 1)).sum()
        tn = ((df_test['Estado_Real'] == 0) & (df_test['Prueba_Positiva'] == 0)).sum()
        fn = ((df_test['Estado_Real'] == 1) & (df_test['Prueba_Positiva'] == 0)).sum()
        
        vpp_empirico = tp / (tp + fp) if (tp + fp) > 0 else 0
        vpp_teorico = (sensibilidad_input * prevalencia_input) / (
            sensibilidad_input * prevalencia_input + (1 - especificidad_input) * (1 - prevalencia_input)
        ) if (sensibilidad_input * prevalencia_input + (1 - especificidad_input) * (1 - prevalencia_input)) > 0 else 0
        
        # Mostrar métricas
        st.subheader("📈 Resultados Clave")
        col1, col2, col3 = st.columns(3)
        col1.metric("Valor Predictivo Positivo (Empírico)", f"{vpp_empirico:.2%}")
        col2.metric("Valor Predictivo Negativo", f"{tn / (tn + fn):.2%}" if (tn + fn) > 0 else "N/A")
        col3.metric("Precisión", f"{(tp + tn) / total:.2%}")
        
        # Diagrama de árbol
        st.subheader("🌳 Diagrama de Árbol Probabilístico")
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Coordenadas para el diagrama
        x = [0, 1, 1]
        y = [0, 0.5, -0.5]
        
        # Dibujar nodos
        ax.plot(x[0], y[0], 'ko', markersize=10)
        ax.plot(x[1], y[1], 'ro', markersize=8)
        ax.plot(x[1], y[2], 'bo', markersize=8)
        
        # Etiquetas
        ax.text(x[0]-0.05, y[0]+0.05, f"P(Enfermo) = {prevalencia_input:.2f}", fontsize=10, ha='right')
        ax.text(x[0]-0.05, y[0]-0.05, f"P(Sano) = {1-prevalencia_input:.2f}", fontsize=10, ha='right')
        
        ax.text(x[1]+0.05, y[1]+0.05, f"P(+|Enfermo) = {sensibilidad_input:.2f}", fontsize=10, ha='left')
        ax.text(x[1]+0.05, y[1]-0.05, f"P(-|Enfermo) = {1-sensibilidad_input:.2f}", fontsize=10, ha='left')
        
        ax.text(x[1]+0.05, y[2]+0.05, f"P(+|Sano) = {1-especificidad_input:.2f}", fontsize=10, ha='left')
        ax.text(x[1]+0.05, y[2]-0.05, f"P(-|Sano) = {especificidad_input:.2f}", fontsize=10, ha='left')
        
        # Líneas
        ax.plot([x[0], x[1]], [y[0], y[1]], 'r-', linewidth=2)
        ax.plot([x[0], x[1]], [y[0], y[2]], 'b-', linewidth=2)
        
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        ax.set_title("Diagrama de Árbol Probabilístico", fontsize=12)
        st.pyplot(fig)
        
        # Interpretación
        st.write("### 💡 Interpretación")
        if vpp_empirico < 0.5:
            st.warning(f"""
            **Paradoja del falso positivo**:  
            Aunque la prueba tiene sensibilidad del {sensibilidad_input*100:.1f}% y especificidad del {especificidad_input*100:.1f}%, 
            solo el **{vpp_empirico*100:.1f}%** de los resultados positivos corresponden a casos reales.  
            Esto implica que más del **{(1 - vpp_empirico)*100:.1f}%** de las intervenciones serían innecesarias.
            """)
        else:
            st.success(f"""
            El valor predictivo positivo es alto ({vpp_empirico:.2%}), lo que indica que la mayoría de los resultados positivos 
            corresponden a casos reales. Esto es deseable en pruebas de cribado.
            """)
        
        # Tabla de contingencia
        st.write("### 📊 Tabla de Contingencia")
        cont_tab = pd.DataFrame({
            "Prueba +": [tp, fp],
            "Prueba -": [fn, tn]
        }, index=["Enfermo", "Sano"])
        st.table(cont_tab)
        
        # Gráfico de barras
        st.write("### 📉 Distribución de Resultados Positivos")
        fig, ax = plt.subplots()
        ax.bar(["Verdaderos Positivos", "Falsos Positivos"], [tp, fp], color=["green", "red"])
        ax.set_ylabel("Número de casos")
        ax.set_title("Desglose de resultados positivos")
        st.pyplot(fig)
        
        # Descarga
        results_df = pd.DataFrame({
            "Métrica": ["Prevalencia", "Sensibilidad", "Especificidad", "VPP Empírico", "VPP Teórico", "Total"],
            "Valor": [prevalencia_input, sensibilidad_input, especificidad_input, vpp_empirico, vpp_teorico, total]
        })
        download_csv_button(results_df, "resultados_falso_positivo.csv", "📥 Descargar resultados (CSV)")
    else:
        st.info("Selecciona una opción y genera/carga los datos para ver los resultados.")

# ===================================================================
# PARTE 2: INFERENCIA PARAMÉTRICA VS NO PARAMÉTRICA (MEJORADA)
# ===================================================================
with tab2:
    st.header("Parte 2: Inferencia Paramétrica vs No Paramétrica")
    
    input_method2 = st.radio(
        "¿Cómo deseas ingresar los datos?",
        ("Simular datos automáticamente",
         "Ingresar datos manualmente",
         "Usar dataset Iris integrado",
         "Cargar archivo CSV"),
        key="parte2_input"
    )
    
    data = None
    variable_name = ""
    
    # === OPCIÓN 1: SIMULACIÓN AUTOMÁTICA ===
    if input_method2 == "Simular datos automáticamente":
        st.subheader("Simular datos desde una distribución")
        col1, col2 = st.columns(2)
        with col1:
            dist_type = st.selectbox("Distribución", ["Normal", "Exponencial", "Log-Normal", "Uniforme"])
        with col2:
            n_sim = st.number_input("Tamaño de la muestra", min_value=10, max_value=10000, value=100, step=10)
        
        if dist_type == "Normal":
            col1, col2 = st.columns(2)
            with col1:
                mu = st.number_input("Media (μ)", value=0.0, step=0.5)
            with col2:
                sigma = st.number_input("Desviación estándar (σ)", min_value=0.1, value=1.0, step=0.1)
            data = np.random.normal(mu, sigma, n_sim)
            variable_name = f"Normal(μ={mu}, σ={sigma})"
        
        elif dist_type == "Exponencial":
            lam = st.number_input("Tasa (λ)", min_value=0.1, value=1.0, step=0.1)
            data = np.random.exponential(1/lam, n_sim)
            variable_name = f"Exponencial(λ={lam})"
        
        elif dist_type == "Log-Normal":
            col1, col2 = st.columns(2)
            with col1:
                mu_log = st.number_input("μ (log)", value=0.0, step=0.5)
            with col2:
                sigma_log = st.number_input("σ (log)", min_value=0.1, value=1.0, step=0.1)
            data = np.random.lognormal(mu_log, sigma_log, n_sim)
            variable_name = f"Log-Normal(μ={mu_log}, σ={sigma_log})"
        
        elif dist_type == "Uniforme":
            col1, col2 = st.columns(2)
            with col1:
                a = st.number_input("Mínimo (a)", value=0.0, step=0.5)
            with col2:
                b = st.number_input("Máximo (b)", value=a + 1.0, step=0.5)
            data = np.random.uniform(a, b, n_sim)
            variable_name = f"Uniforme(a={a}, b={b})"
    
    # === OPCIÓN 2: INGRESO MANUAL ===
    elif input_method2 == "Ingresar datos manualmente":
        st.subheader("Ingresa tus datos (separados por comas)")
        data_input = st.text_input("Ejemplo: 1.2, 3.4, 5.6, 7.8", value="1.2, 2.3, 3.1, 4.5, 5.0")
        try:
            data = np.array([float(x.strip()) for x in data_input.split(",") if x.strip()])
            variable_name = "Datos manuales"
            if len(data) < 2:
                st.warning("Se necesitan al menos 2 valores.")
                data = None
        except ValueError:
            st.error("Formato inválido. Ingresa números separados por comas.")
    
    # === OPCIÓN 3: DATASET IRIS ===
    elif input_method2 == "Usar dataset Iris integrado":
        st.subheader("Selecciona especie y variable")
        col1, col2 = st.columns(2)
        with col1:
            especie = st.selectbox("Especie", ["setosa", "versicolor", "virginica"])
        with col2:
            variable = st.selectbox("Variable", iris.feature_names)
        
        data = iris_df[iris_df['species'] == especie][variable].values
        variable_name = f"{variable} ({especie})"
    
    # === OPCIÓN 4: CARGAR CSV ===
    else:  # Cargar CSV
        st.subheader("Cargar archivo CSV")
        st.markdown("El archivo debe contener al menos una columna numérica continua.")
        uploaded_file2 = st.file_uploader("Sube tu archivo CSV", type=["csv"], key="csv_parte2")
        if uploaded_file2 is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file2)
                numeric_cols = df_uploaded.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.error("No se encontraron columnas numéricas en el archivo.")
                else:
                    col_sel = st.selectbox("Selecciona la variable numérica", numeric_cols)
                    data = df_uploaded[col_sel].dropna().values
                    variable_name = col_sel
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")
    
    # === RESULTADOS PARTE 2 ===
    if data is not None and len(data) > 5:
        n = len(data)
        st.divider()
        st.subheader("📊 Tabla de Datos (primeros 10 valores)")
        st.write(data[:10])
        
        # Botón para ver más (¡con key único!)
        if len(data) > 10:
            if st.button("Mostrar todos los datos", key="show_all_data_part2"):
                st.write(data)
        
        st.write(f"**Variable analizada**: {variable_name} (n = {n})")
        
        # Prueba de normalidad (solo si n <= 5000)
        if n <= 5000:
            _, p_shapiro = stats.shapiro(data)
        else:
            p_shapiro = np.nan
        
        # Inferencia paramétrica (IC para la media)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if n > 1:
            ic_param = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))
        else:
            ic_param = (mean, mean)
        
        # Inferencia no paramétrica (bootstrap)
        ic_boot = bootstrap_ci(data)
        
        # Mostrar resultados
        st.subheader("📈 Resultados de Inferencia")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Inferencia Paramétrica** (asume normalidad)")
            st.write(f"Media: {mean:.4f}")
            st.write(f"IC 95%: [{ic_param[0]:.4f}, {ic_param[1]:.4f}]")
        with col2:
            st.write("**Inferencia No Paramétrica** (bootstrap)")
            st.write(f"Media: {mean:.4f}")
            st.write(f"IC 95%: [{ic_boot[0]:.4f}, {ic_boot[1]:.4f}]")
        
        if not np.isnan(p_shapiro):
            st.write(f"**Prueba de normalidad (Shapiro-Wilk)**: p-valor = {p_shapiro:.4f}")
        
        # Interpretación
        st.write("### 💡 Interpretación")
        if np.isnan(p_shapiro):
            normal_msg = "No se realizó prueba de normalidad (n > 5000)."
            use_param = "Se recomienda usar inferencia no paramétrica para mayor robustez."
        elif p_shapiro > 0.05:
            normal_msg = f"Los datos **parecen normales** (p = {p_shapiro:.4f} > 0.05)."
            use_param = "La inferencia paramétrica es apropiada."
        else:
            normal_msg = f"Los datos **no son normales** (p = {p_shapiro:.4f} < 0.05)."
            use_param = "Se recomienda usar inferencia no paramétrica (bootstrap)."
        
        ic_diff = abs((ic_param[1] - ic_param[0]) - (ic_boot[1] - ic_boot[0]))
        if ic_diff < 0.1 * (ic_boot[1] - ic_boot[0]):
            sim_msg = "Los intervalos son muy similares."
        else:
            sim_msg = "Los intervalos difieren notablemente."
        
        st.info(f"""
        {normal_msg} {sim_msg}  
        **Conclusión**: {use_param}
        """)
        
        # Gráfico
        st.write("### 📉 Distribución de los datos")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data, kde=True, stat="density", ax=ax, color="skyblue", alpha=0.7, label="Datos + KDE")
        
        # Curva normal ajustada
        x = np.linspace(data.min(), data.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mean, std), 'r--', label="Normal ajustada")
        
        # ICs
        ax.axvline(ic_param[0], color='red', linestyle=':', alpha=0.7)
        ax.axvline(ic_param[1], color='red', linestyle=':', alpha=0.7, label="IC Paramétrico")
        ax.axvline(ic_boot[0], color='green', linestyle='-.', alpha=0.7)
        ax.axvline(ic_boot[1], color='green', linestyle='-.', alpha=0.7, label="IC Bootstrap")
        
        ax.set_xlabel(variable_name)
        ax.legend()
        st.pyplot(fig)
        
        # Descarga
        results_df2 = pd.DataFrame({
            "Métrica": ["Media", "IC Paramétrico (inferior)", "IC Paramétrico (superior)",
                        "IC Bootstrap (inferior)", "IC Bootstrap (superior)", "p-valor Shapiro", "Tamaño muestra"],
            "Valor": [mean, ic_param[0], ic_param[1], ic_boot[0], ic_boot[1], 
                     p_shapiro if not np.isnan(p_shapiro) else "N/A", n]
        })
        download_csv_button(results_df2, "resultados_inferencia.csv", "📥 Descargar resultados (CSV)")
    else:
        if data is not None and len(data) <= 5:
            st.warning("Se necesitan al menos 6 observaciones para realizar inferencia.")
        else:
            st.info("Selecciona una opción para cargar o generar datos.")

# === PIE DE PÁGINA ===
st.markdown("---")
st.caption("Desarrollado para el curso de Estadística Bayesiana • Escuela Profesional de Ingeniería Estadística e Informática")