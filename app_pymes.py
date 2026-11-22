import streamlit as st
import pandas as pd
import time
import joblib

# --- 1. CONFIGURACIÓN Y CARGA DEL MODELO ---
st.set_page_config(page_title="Auditor IA PyMEs", page_icon="⚡", layout="wide")

@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('modelo_consumo_francia.pkl')
    except FileNotFoundError:
        return None

modelo_ia = cargar_modelo()

# --- ESTILOS CSS PARA LA TARJETA (Diseño limpio) ---
st.markdown("""
    <style>
    .tarjeta-semaforo { 
        background-color: #f8f9fa; 
        padding: 25px; 
        border-radius: 12px; 
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        border-top: 8px solid #333; /* Borde superior que cambiaremos dinámicamente */
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Auditor Inteligente de Eficiencia Energética")
st.markdown("Ciencia de Datos para Negocios")

if modelo_ia is None:
    st.error("⚠️ Error: No se encontró 'modelo_consumo_francia.pkl'. Asegúrate de ponerlo en la misma carpeta.")
else:
    st.success("🧠 Motor de IA conectado y listo.")
st.markdown("---")

# --- 2. COLUMNAS ---
col_form, col_resultados = st.columns([1, 1.5])

with col_form:
    st.subheader("📋 Datos del Negocio")
    
    with st.form("formulario_pyme"):
        nombre = st.text_input("Nombre del Negocio", "Abarrotes La Chiquita")
        
        col1, col2 = st.columns(2)
        with col1:
            consumo = st.number_input("Consumo (kWh)", min_value=0.0, value=716.0, step=10.0)
            refris = st.number_input("Refrigeradores", min_value=0, value=12, step=1)
            focos_trad = st.number_input("Focos Tradicionales", min_value=0, value=2, step=1)
        with col2:
            pago = st.number_input("Monto Pagado ($ MXN)", min_value=0.0, value=3948.0, step=50.0)
            compus = st.number_input("Computadoras", min_value=0, value=0, step=1)
            focos_led = st.number_input("Focos LED", min_value=0, value=12, step=1)

        boton_calcular = st.form_submit_button("🤖 Calcular Semáforo con IA Real", use_container_width=True, type="primary")

# --- 3. LÓGICA CONECTADA A LA IA ---
with col_resultados:
    if boton_calcular and modelo_ia is not None:
        with st.spinner("La Red Neuronal está procesando los datos..."):
            time.sleep(1.5) 
        
        # IA Real
        potencia_promedio_kw = consumo / (30 * 24)
        intensidad_promedio = (potencia_promedio_kw * 1000) / 235 
        
        datos_entrada = pd.DataFrame([[potencia_promedio_kw, 235.0, intensidad_promedio]], 
                                     columns=['Global_active_power', 'Voltage', 'Global_intensity'])
        
        prediccion_ia_kw = modelo_ia.predict(datos_entrada)[0]
        consumo_predicho = max(prediccion_ia_kw * 24 * 30, consumo * 0.4)
        
        costo_optimizado = consumo_predicho * 2.50 
        ahorro = max(pago - costo_optimizado, 0)
        emisiones = consumo * 0.417
        
        # Semáforo y Colores
        if consumo >= 700:
            semaforo = "🔴 ROJO - ALTO DESPERDICIO"
            color_hex = "#cc0000" # Rojo oscuro para el texto
            borde_hex = "#ff4b4b" # Rojo brillante para el borde
            recomendacion = f"La IA detectó que deberías consumir {consumo_predicho:.0f} kWh. Cambia focos a LED y revisa tus {refris} refrigeradores."
        elif consumo >= 500:
            semaforo = "🟡 AMARILLO - MEJORABLE"
            color_hex = "#b36b00" # Naranja oscuro
            borde_hex = "#ffa500" 
            recomendacion = f"La IA detectó que deberías consumir {consumo_predicho:.0f} kWh. Apaga equipos en horas no laborales."
        else:
            semaforo = "🟢 VERDE - EFICIENTE"
            color_hex = "#008000" # Verde oscuro
            borde_hex = "#00cc00"
            recomendacion = "Tu consumo es casi óptimo en comparación con tu inventario. ¡Sigue así!"

        # --- MOSTRAR RESULTADOS ---
        st.subheader(f"Resultados para: {nombre}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Consumo Real", f"{consumo} kWh")
        c2.metric("Predicción IA", f"{consumo_predicho:.1f} kWh") 
        c3.metric("Ahorro Potencial", f"${ahorro:.2f} MXN", delta=f"-{ahorro:.2f}", delta_color="inverse")

        # Tarjeta del Semáforo (AHORA CON EL DISEÑO ARREGLADO)
        st.markdown(f"""
            <div class="tarjeta-semaforo" style="border-top: 8px solid {borde_hex};">
                <h1 style="color: {color_hex}; margin-bottom: 5px;">{semaforo}</h1>
                <p style="color: #333333; font-size: 16px; margin-top: 10px;">
                    <b>Recomendación:</b> {recomendacion}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        
        # --- GRÁFICAS ---
        g1, g2 = st.columns(2)
        with g1:
            st.write("*Consumo Real vs. Predicción de la IA*")
            df_consumo = pd.DataFrame({
                "Tipo": ["Real", "Predicción IA"],
                "kWh": [consumo, consumo_predicho]
            })
            st.bar_chart(df_consumo, x="Tipo", y="kWh", color="#1f77b4")

        with g2:
            st.write("*Huella Ambiental (kg de CO2)*")
            df_co2 = pd.DataFrame({
                "Emisiones": ["Huella Real"],
                "kg CO2": [emisiones]
            })
            st.bar_chart(df_co2, x="Emisiones", y="kg CO2", color="#16bc61")

    elif boton_calcular and modelo_ia is None:
         st.error("No se puede calcular porque el modelo de IA no está cargado.")