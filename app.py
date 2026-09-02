import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

import plotly.graph_objects as go  # <-- ASEGÚRATE DE QUE ESTA LÍNEA ESTÉ AL INICIO




# Configuración de la página
st.set_page_config(page_title="Simulador Mecánico Universal", layout="wide")

st.title("🎛️ Simulador Mecánico Universal con Animación")
st.markdown("""
Esta aplicación calcula la dinámica de cualquier sistema de mecánica clásica **hasta 4 GDL**.
Incluye la corrección exacta de signos físicos y una **animación interactiva con efecto trazo/órbita** en el espacio de configuración.
""")

# --- Entrada de datos en la barra lateral ---
st.sidebar.header("1. Definición del Sistema")

gdl = st.sidebar.selectbox("Grados de Libertad (GDL)", [1, 2, 3, 4], index=1)

st.sidebar.subheader("Variables y Parámetros")
parámetros_str = st.sidebar.text_input("Parámetros constantes (separados por coma)", "m, k, l, g")
    
st.sidebar.markdown("**Expresión del Lagrangiano $L$:**")

# Formulario para construir el Lagrangiano
with st.sidebar.form("lagrangian_form"):
    st.markdown("### Definir Lagrangiano")
    st.caption("Usa `q1, q2, q3, q4` para posiciones y `dq1, dq2, dq3, dq4` para velocidades.")
    
    if gdl == 1:
        default_L = "0.5 * m * dq1**2 - 0.5 * k * q1**2"
    elif gdl == 2:
        default_L = "0.5 * m * (dq1**2 + dq2**2) - 0.5 * k * (q1**2 + q2**2 + (q2 - q1)**2)"
    elif gdl == 3:
        default_L = "0.5 * m * (dq1**2 + dq2**2 + dq3**2) - 0.5 * k * (q1**2 + (q2-q1)**2 + (q3-q2)**2)"
    else:
        default_L = "0.5 * m * (dq1**2 + dq2**2 + dq3**2 + dq4**2) - 0.5 * k * (q1**2 + q2**2 + q3**2 + q4**2)"
        
    lagr_input = st.text_input("L =", default_L)
        
    st.markdown("### Condiciones Iniciales")
    c_iniciales = {}
    col_q, col_dq = st.columns(2)
    for i in range(gdl):
        with col_q:
            c_iniciales[f'q{i+1}'] = st.number_input(f"q{i+1} inicial", value=1.0 if i==0 else 0.0)
        with col_dq:
            c_iniciales[f'dq{i+1}'] = st.number_input(f"dq{i+1} inicial (vel)", value=0.0)
    
    st.markdown("### Parámetros de Simulación")
    t_max = st.number_input("Tiempo total (s)", value=10.0, min_value=1.0)
    dt = st.number_input("Paso de tiempo (dt)", value=0.01, min_value=0.0001, max_value=0.5, format="%.4f")
    
    st.markdown("### Valores de los Parámetros")
    param_vals = {}
    if parámetros_str:
        for p in parámetros_str.split(","):
            p = p.strip()
            if p:
                param_vals[p] = st.number_input(f"Valor de {p}", value=1.0 if p != 'g' else 9.81)

    calcular = st.form_submit_button("Calcular y Resolver")

if calcular:
    t = sp.Symbol('t')
    
    local_dict = {p: sp.Symbol(p) for p in param_vals.keys()}
    local_dict.update({'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'pi': sp.pi, 'exp': sp.exp})
    
    q = [sp.Function(f'q{i+1}')(t) for i in range(gdl)]
    dq = [q[i].diff(t) for i in range(gdl)]
    ddq = [dq[i].diff(t) for i in range(gdl)]
    
    for i in range(gdl):
        local_dict[f'q{i+1}'] = q[i]
        local_dict[f'dq{i+1}'] = dq[i]
        
    try:
        L = sp.parse_expr(lagr_input, local_dict=local_dict)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🔬 Análisis Simbólico")
            st.latex(f"L = {sp.latex(L)}")
            
        eqs = []
        for i in range(gdl):
            dL_dq = L.diff(q[i])
            dL_ddq = L.diff(dq[i])
            d_dt_dL_ddq = dL_ddq.diff(t)
            
            eq = d_dt_dL_ddq - dL_dq
            eqs.append(eq)
            
        sym_ddq_temporal = [sp.Symbol(f'ddq_{i+1}') for i in range(gdl)]
        
        eqs_algebraicas = []
        for eq in eqs:
            eq_temp = eq
            for i in range(gdl):
                eq_temp = eq_temp.subs(ddq[i], sym_ddq_temporal[i])
            eqs_algebraicas.append(eq_temp)
            
        A_sym, B_sym = sp.linear_eq_to_matrix(eqs_algebraicas, sym_ddq_temporal)
        
        A_num_expr = A_sym.subs(param_vals)
        B_num_expr = B_sym.subs(param_vals)
        
        sym_q = [sp.Symbol(f'q{i+1}') for i in range(gdl)]
        sym_dq = [sp.Symbol(f'dq{i+1}') for i in range(gdl)]
        
        for i in range(gdl):
            for j in range(gdl):
                A_num_expr = A_num_expr.subs(dq[j], sym_dq[j]).subs(q[j], sym_q[j])
                B_num_expr = B_num_expr.subs(dq[j], sym_dq[j]).subs(q[j], sym_q[j])
                
        func_A = sp.lambdify((*sym_q, *sym_dq), A_num_expr, 'numpy')
        func_B = sp.lambdify((*sym_q, *sym_dq), B_num_expr, 'numpy')

        # --- BUCLE DE INTEGRACIÓN (SIGNO CORREGIDO) ---
        tiempos = np.arange(0, t_max, dt)
        n_pasos = len(tiempos)
        
        historial_q = np.zeros((gdl, n_pasos))
        historial_dq = np.zeros((gdl, n_pasos))
        
        for i in range(gdl):
            historial_q[i, 0] = c_iniciales[f'q{i+1}']
            historial_dq[i, 0] = c_iniciales[f'dq{i+1}']
            
        for k in range(n_pasos - 1):
            q_actual = historial_q[:, k]
            dq_actual = historial_dq[:, k]
            
            M_eval = np.array(func_A(*q_actual, *dq_actual), dtype=float)
            if gdl == 1: M_eval = np.array([[M_eval]])
                
            F_eval = np.array(func_B(*q_actual, *dq_actual), dtype=float).flatten()
            
            # Corrección de signo físico efectuada aquí (F_eval en lugar de -F_eval)
            try:
                aceleraciones = np.linalg.solve(M_eval, F_eval)
            except np.linalg.LinAlgError:
                aceleraciones = np.linalg.pinv(M_eval).dot(F_eval)
            
            historial_dq[:, k+1] = dq_actual + aceleraciones * dt
            historial_q[:, k+1] = q_actual + historial_dq[:, k+1] * dt

        # --- GRÁFICOS ESTÁTICOS ---
        with col2:
            st.header(f"📊 Curvas de Evolución Temporal")
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            for i in range(gdl):
                ax[0].plot(tiempos, historial_q[i, :], label=f"q{i+1} (Posición)", lw=2)
                ax[1].plot(tiempos, historial_dq[i, :], label=f"dq{i+1} (Velocidad)", linestyle="--", lw=1.5)
            
            ax[0].set_ylabel("Posición")
            ax[0].grid(True)
            ax[0].legend()
            ax[0].set_title("Coordenadas Generalizadas")
            
            ax[1].set_ylabel("Velocidad")
            ax[1].set_xlabel("Tiempo (s)")
            ax[1].grid(True)
            ax[1].legend()
            st.pyplot(fig)
            
        # --- SECCIÓN DE ANIMACIÓN (ÓRBITA / TRAZO) ---
        st.markdown("---")
        st.header("🌌 Órbita en el Espacio de Configuración Animada")
        
        # Preparación de datos para Plotly
        data_dict = {'Tiempo': tiempos}
        for i in range(gdl):
            data_dict[f'q{i+1}'] = historial_q[i, :]
            data_dict[f'dq{i+1}'] = historial_dq[i, :]
            
        df = pd.DataFrame(data_dict)
        
        # Para que la animación corra fluida y no congelada, diezmamos los frames si hay demasiados pasos
        max_frames = 150
        step = max(1, n_pasos // max_frames)
        df_anim = df.iloc[::step].copy()
        
        # Definición de ejes interactivos
        if gdl == 1:
            st.info("Mostrando Espacio de Fase (q1 vs dq1) debido a que el sistema tiene solo 1 GDL.")
            x_col, y_col = 'q1', 'dq1'
        else:
            col_sel1, col_sel2 = st.columns(2)
            opciones_q = [f"q{i+1}" for i in range(gdl)]
            with col_sel1:
                x_col = st.selectbox("Eje X (Órbita)", opciones_q, index=0)
            with col_sel2:
                y_col = st.selectbox("Eje Y (Órbita)", opciones_q, index=1)
                
        # Construimos las columnas acumulativas para el efecto "trazo/estela"
        # Esto hace que se dibuje la línea completa detrás de la esfera viajera
        df_frames = []
        for i, t_actual in enumerate(df_anim['Tiempo']):
            sub_df = df_anim[df_anim['Tiempo'] <= t_actual].copy()
            sub_df['Frame'] = t_actual  # Identificador único de tiempo para el cuadro animado
            # Marcamos cuál es la última fila del cuadro actual para dibujarla como la "esfera" de la órbita
            sub_df['Tipo'] = 'Estela'
            if len(sub_df) > 0:
                sub_df.iloc[-1, sub_df.columns.get_loc('Tipo')] = 'Partícula'
            df_frames.append(sub_df)
            
        df_final_anim = pd.concat(df_frames, ignore_index=True)
        
        # Generar gráfico animador de Plotly
        # === REEMPLAZA DESDE AQUÍ HASTA EL FINAL DE TU CÓDIGO ===
        
        # 1. Extraemos los vectores numéricos limpios del DataFrame
        x_vals = df_anim[x_col].values
        y_vals = df_anim[y_col].values
        
        # 2. Calculamos los límites de la pantalla para que no se mueva el gráfico
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        pad_x = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
        pad_y = (y_max - y_min) * 0.1 if y_max != y_min else 1.0

        # 3. Creamos el nuevo gráfico configurado como línea fina continuas
        fig_plotly = go.Figure(
            data=[
                # La estela ahora es una línea fina continua gracias a mode="lines"
                go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(0, 128, 128, 0.7)", width=1.5), name="Estela"),
                # La masa que se mueve es un punto rojo
                go.Scatter(x=[], y=[], mode="markers", marker=dict(color="red", size=10), name="Partícula")
            ],
            layout=go.Layout(
                xaxis=dict(range=[x_min - pad_x, x_max + pad_x], title=f"Posición {x_col}"),
                yaxis=dict(range=[y_min - pad_y, y_max + pad_y], title=f"Posición {y_col}"),
                hovermode="closest",
                height=550,
                showlegend=False,
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 30, "redraw": False}, "fromcurrent": True}]
                        },
                        {
                            "label": "⏸ Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                        }
                    ]
                }]
            ),
            frames=[
                go.Frame(
                    data=[
                        # Aquí le ordenamos a la línea crecer uniendo todos los puntos anteriores
                        go.Scatter(x=x_vals[:idx+1], y=y_vals[:idx+1]), 
                        # Aquí posicionamos el punto rojo justo en la punta actual
                        go.Scatter(x=[x_vals[idx]], y=[y_vals[idx]])    
                    ],
                    name=str(df_anim['Tiempo'].iloc[idx])
                ) for idx in range(len(x_vals))
            ]
        )
        
        # Desactivar configuraciones pesadas para que la animación vuele en el navegador
        for frame in fig_plotly.frames:
            frame.layout = go.Layout(sliders=[])

        # Renderizar en la pantalla de Streamlit
        st.plotly_chart(fig_plotly, use_container_width=True)


    except Exception as e:
        st.error(f"Error en la simulación: {e}")
