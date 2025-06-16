import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Análisis de Frecuencia de Palabras", layout="wide")

# Título de la aplicación
st.title("📊 Análisis de Frecuencia de Palabras en Discursos Presidenciales")

# Cargar datos (con cache para mejorar rendimiento)
@st.cache_data
def cargar_datos():
    return pd.read_csv('discursos_presidenciales_limpios.csv')

def calcular_frecuencias(df, palabras_input):
    # Parsear y validar entrada
    palabras = [p.strip().lower() for p in palabras_input.split(',') if p.strip()]
    
    if len(palabras) > 2:
        raise ValueError("Solo se permiten máximo 2 palabras separadas por coma")
    
    df = df.copy()
    
    if len(palabras) == 1:
        # Caso de una palabra (código original)
        palabra = palabras[0]
        df['freq_absoluta'] = df['texto_limpio'].apply(lambda texto: texto.lower().split().count(palabra))
        
        aparece = (df['freq_absoluta'] > 0).sum()
        
        if aparece == 0:
            return None, palabras  # No se encontraron resultados
        
        tfidf_scores = []
        for x, row in df.iterrows():
            tokens = row['texto_limpio'].lower().split()
            tf = tokens.count(palabra) / len(tokens) if len(tokens) > 0 else 0
            idf = np.log(len(df) / aparece) if aparece > 0 else 0
            tfidf = tf * idf
            tfidf_scores.append(tfidf)
        
        df['freq_relativa'] = tfidf_scores
        
    else:
        # Caso de dos palabras - cálculo ponderado (promedio)
        palabra1, palabra2 = palabras[0], palabras[1]
        
        # Calcular frecuencias absolutas para cada palabra
        df['freq_abs_1'] = df['texto_limpio'].apply(lambda texto: texto.lower().split().count(palabra1))
        df['freq_abs_2'] = df['texto_limpio'].apply(lambda texto: texto.lower().split().count(palabra2))
        
        # Promedio de frecuencias absolutas
        df['freq_absoluta'] = (df['freq_abs_1'] + df['freq_abs_2']) / 2
        
        # Verificar si alguna de las palabras aparece
        aparece1 = (df['freq_abs_1'] > 0).sum()
        aparece2 = (df['freq_abs_2'] > 0).sum()
        
        if aparece1 == 0 and aparece2 == 0:
            return None, palabras  # No se encontraron resultados
        
        # Calcular TF-IDF para cada palabra
        tfidf_scores = []
        for x, row in df.iterrows():
            tokens = row['texto_limpio'].lower().split()
            
            # TF-IDF para palabra 1
            tf1 = tokens.count(palabra1) / len(tokens) if len(tokens) > 0 else 0
            idf1 = np.log(len(df) / aparece1) if aparece1 > 0 else 0
            tfidf1 = tf1 * idf1
            
            # TF-IDF para palabra 2
            tf2 = tokens.count(palabra2) / len(tokens) if len(tokens) > 0 else 0
            idf2 = np.log(len(df) / aparece2) if aparece2 > 0 else 0
            tfidf2 = tf2 * idf2
            
            # Promedio ponderado de TF-IDF
            tfidf_promedio = (tfidf1 + tfidf2) / 2
            tfidf_scores.append(tfidf_promedio)
        
        df['freq_relativa'] = tfidf_scores
    
    return df, palabras

def graficar_frecuencias(df, palabras_lista):
    # Limpiar figuras anteriores para evitar problemas de actualización
    plt.close('all')
    
    resumen = df.groupby('anio').agg({
        'freq_absoluta': 'sum',
        'freq_relativa': 'mean'
    }).reset_index()
    
    # Encontrar el documento con mayor TF-IDF
    idx_max = df['freq_relativa'].idxmax()
    anio_max = df.loc[idx_max, 'anio']
    presidente_max = df.loc[idx_max, 'presidente']
    
    # Crear título según número de palabras
    if len(palabras_lista) == 1:
        titulo = f"Análisis de Frecuencia: '{palabras_lista[0]}' por Año"
    else:
        titulo = f"Análisis de Frecuencia: '{palabras_lista[0]}' + '{palabras_lista[1]}' por Año"
    
    # Configuración del gráfico
    fig, ax1 = plt.subplots(figsize=(15, 8))
    bars = ax1.bar(resumen['anio'], resumen['freq_absoluta'],
                   color='skyblue', label='Frecuencia Absoluta')
    
    ax1.set_ylabel('Frecuencia Absoluta', color='blue')
    ax1.set_xlabel('Año')
    plt.xticks(rotation=45)
    
    # Anotación
    anio_serie = resumen[resumen['anio'] == anio_max]
    if not anio_serie.empty:
        altura_barra = anio_serie['freq_absoluta'].iloc[0]
        ax1.annotate(f'Máximo TF-IDF\n{presidente_max}',
                     xy=(anio_max, altura_barra),
                     xytext=(-15, 15),
                     textcoords='offset points',
                     ha='right', va='bottom',
                     fontsize=11,
                     color='darkred',
                     bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="lightyellow",
                              edgecolor="orange",
                              alpha=0.9),
                     arrowprops=dict(arrowstyle='->',
                                   color='darkred',
                                   alpha=0.8,
                                   linewidth=1.5))
    
    # Eje secundario para frecuencia relativa
    ax2 = ax1.twinx()
    ax2.plot(resumen['anio'], resumen['freq_relativa'],
             color='red', marker='o', linewidth=2,
             label='TF-IDF (Relativa)')
    ax2.set_ylabel('Frecuencia Relativa (TF-IDF)', color='red')
    
    # Configuración final del gráfico
    plt.title(titulo, fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Leyendas combinadas
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    return fig

# Interfaz de usuario
try:
    df = cargar_datos()
    
    # Crear dos columnas para la interfaz
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuración")
        palabra_usuario = st.text_input(
            "Ingresa palabra(s) a analizar (1 o 2 separadas por coma; sin acentos):", 
            value="inmigrantes",
            help="Ingresa una palabra o dos palabras separadas por coma (ej: 'trabajo, empleo')"
        )
        
        if st.button("🔍 Visualizar", type="primary"):
            try:
                if palabra_usuario.strip():
                    # Validar entrada
                    palabras_test = [p.strip() for p in palabra_usuario.split(',') if p.strip()]
                    if len(palabras_test) > 2:
                        st.error("⚠️ Solo se permiten máximo 2 palabras separadas por coma")
                        st.session_state.mostrar_grafico = False
                    elif len(palabras_test) == 0:
                        st.error("⚠️ Por favor ingresa al menos una palabra válida")
                        st.session_state.mostrar_grafico = False
                    else:
                        st.session_state.palabra_actual = palabra_usuario.strip()
                        st.session_state.mostrar_grafico = True
                        st.session_state.num_palabras = len(palabras_test)
                else:
                    st.error("⚠️ Por favor ingresa una palabra válida")
                    st.session_state.mostrar_grafico = False
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.mostrar_grafico = False
    
    # Container para el gráfico que puede ser limpiado
    grafico_container = col2.container()
    
    if hasattr(st.session_state, 'mostrar_grafico') and st.session_state.mostrar_grafico:
        try:
            with grafico_container:
                with st.spinner(f"Analizando: '{st.session_state.palabra_actual}'..."):
                    df_resultado, palabras_lista = calcular_frecuencias(df, st.session_state.palabra_actual)
                    
                    if df_resultado is None:
                        # No se encontraron resultados
                        st.warning(f"🔍 No se encontraron resultados para: '{st.session_state.palabra_actual}'")
                        st.info("💡 Sugerencias:\n- Verifica la ortografía\n- Prueba con sinónimos\n- Usa palabras más generales")
                        st.session_state.mostrar_grafico = False
                    else:
                        # Generar y mostrar gráfico
                        fig = graficar_frecuencias(df_resultado, palabras_lista)
                        st.pyplot(fig)
                        
                        # Mostrar estadísticas
                        col2a, col2b, col2c = st.columns(3)
                        
                        total_apariciones = df_resultado['freq_absoluta'].sum()
                        documentos_con_palabra = (df_resultado['freq_absoluta'] > 0).sum()
                        
                        with col2a:
                            st.metric("Total de apariciones", f"{total_apariciones:.1f}")
                        with col2b:
                            st.metric("Documentos con palabra(s)", f"{documentos_con_palabra}/{len(df_resultado)}")
                        with col2c:
                            if len(palabras_lista) == 2:
                                st.metric("Modo de análisis", "2 palabras (promedio)")
                            else:
                                st.metric("Modo de análisis", "1 palabra")
                        
        except Exception as e:
            with grafico_container:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("💡 Verifica que las palabras estén escritas correctamente")
            st.session_state.mostrar_grafico = False
    
    elif hasattr(st.session_state, 'mostrar_grafico') and not st.session_state.mostrar_grafico:
        # Limpiar el área del gráfico cuando hay error
        with grafico_container:
            st.empty()

except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'discursos_presidenciales_limpios.csv'. Asegúrate de que esté en el mismo directorio que la aplicación.")
except Exception as e:
    st.error(f"❌ Error al cargar los datos: {str(e)}")

# Información adicional
with st.expander("ℹ️ Información sobre el análisis"):
    st.write("""
    **Análisis de una palabra:**
    - **Frecuencia Absoluta:** Número total de veces que aparece la palabra en cada año.
    - **TF-IDF:** Medida que refleja la importancia de la palabra en un documento.
    
    **Análisis de dos palabras (separadas por coma):**
    - **Frecuencia Absoluta:** Promedio de apariciones de ambas palabras en cada año.
    - **TF-IDF:** Promedio ponderado del TF-IDF de ambas palabras.
    
    **Ejemplos de uso:**
    - Una palabra: `democracia`
    - Dos palabras: `trabajo, empleo` o `seguridad, violencia`
    """)
    
    st.info("💡 **Tip:** El análisis de dos palabras es útil para estudiar temas relacionados o sinónimos.")
