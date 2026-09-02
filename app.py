import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# Configuración de la página
st.set_page_config(page_title="Simulador Mecánico Universal", layout="wide")

st.title("🎛️ Simulador Mecánico Universal con Animación")
st.markdown("""
Esta versión incluye un motor de animación optimizado con **trazos de línea fina** y un **controlador de velocidad** en la barra lateral.
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

# Control de velocidad de animación fuera del formulario para que sea dinámico
st.sidebar.markdown("---")
st.sidebar.subheader("🏃 Controles de la Animación")
velocidad_sim = st.sidebar.slider("Velocidad de reproducción", min_value=1, max_value=10, value=3, help="Aumenta este valor para saltar pasos y acelerar la animación.")
duracion_frame = st.sidebar.slider("Duración del Frame (ms)", min_value=10, max_value=200, value=30, help="Tiempo de espera en milisegundos entre cada cuadro.")

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
            eqs.append(d_dt_dL_ddq - dL_dq)
            
        sym_ddq_temporal = [sp.Symbol(f'ddq_{i+1}') for i in range(gdl)]
        eqs_algebraicas = []
        for eq in eqs:
            eq_temp = eq
            for i in range(gdl):
                eq_temp = eq_temp.subs(ddq[i], sym_ddq_temporal[i])
            eqs_algebraicas.append(eq_temp)
            
        A_sym, B_sym = sp.linear_eq_to_matrix(eqs_algebraicas, sym_ddq_temporal)



        # --- CÓDIGO CORREGIDO (Alrededor de la línea 105) ---
        # Cambiamos M_num_expr por A_num_expr para que use la matriz lineal de SymPy
        for i in range(gdl):
            for j in range(gdl):
                A_num_expr = A_num_expr.subs(dq[j], sym_dq[j]).subs(q[j], sym_q[j])
                B_num_expr = B_num_expr.subs(dq[j], sym_dq[j]).subs(q[j], sym_q[j])
                
        func_A = sp.lambdify((*sym_q, *sym_dq), A_num_expr, 'numpy')
        func_M = sp.lambdify((*sym_q, *sym_dq), A_num_expr, 'numpy') # <-- Cambiado M_num_expr por A_num_expr
        func_B = sp.lambdify((*sym_q, *sym_dq), B_num_expr, 'numpy')


        # --- BUCLE DE INTEGRACIÓN ---
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
            
            M_eval = np.array(func_M(*q_actual, *dq_actual), dtype=float)
            if gdl == 1: M_eval = np.array([[M_eval]])
            F_eval = np.array(func_B(*q_actual, *dq_actual), dtype=float).flatten()
            
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
                ax.plot(tiempos, historial_q[i, :], label=f"q{i+1} (Posición)", lw=2)
                ax.plot(tiempos, historial_dq[i, :], label=f"dq{i+1} (Velocidad)", linestyle="--", lw=1.5)
            ax.set_ylabel("Posición")
            ax.grid(True)
            ax.legend()
            ax.set_title("Coordenadas Generalizadas")
            ax.set_xlabel("Tiempo (s)")
            st.pyplot(fig)
            
        # --- SECCIÓN DE ANIMACIÓN OPTIMIZADA (go.Figure) ---
        st.markdown("---")
        st.header("🌌 Órbita en el Espacio de Configuración Animada")
        
        if gdl == 1:
            st.info("Mostrando Espacio de Fase (q1 vs dq1) debido a que el sistema tiene solo 1 GDL.")
            x_vals = historial_q[0, :]
            y_vals = historial_dq[0, :]
            x_label, y_label = "Posición q1", "Velocidad dq1"
        else:
            col_sel1, col_sel2 = st.columns(2)
            opciones_q = [f"q{i+1}" for i in range(gdl)]
            with col_sel1:
                x_col = st.selectbox("Eje X (Órbita)", opciones_q, index=0)
            with col_sel2:
                y_col = st.selectbox("Eje Y (Órbita)", opciones_q, index=1)
            
            x_vals = historial_q[int(x_col[-1]) - 1, :]
            y_vals = historial_q[int(y_col[-1]) - 1, :]
            x_label, y_label = f"Posición {x_col}", f"Posición {y_col}"
                
        # Aplicamos el multiplicador de velocidad (Salto de pasos)
        indices_anim = np.arange(0, n_pasos, velocidad_sim)
        if indices_anim[-1] != n_pasos - 1:
            indices_anim = np.append(indices_anim, n_pasos - 1)
            
        # Límites fijos de los ejes
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()
        pad_x = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
        pad_y = (y_max - y_min) * 0.1 if y_max != y_min else 1.0

        # Crear base de la animación utilizando la API de objetos de Plotly (mucho más rápida)
        fig_anim = go.Figure(
            data=[
                go.Scatter(x=[], y=[], mode="lines", line=dict(color="teal", width=1.5), name="Estela"), # Trazo fino
                go.Scatter(x=[], y=[], mode="markers", marker=dict(color="red", size=10), name="Partícula") # Esfera principal
            ],
            layout=go.Layout(
                xaxis=dict(range=[x_min - pad_x, x_max + pad_x], title=x_label),
                yaxis=dict(range=[y_min - pad_y, y_max + pad_y], title=y_label),
                hovermode="closest",
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "▶ Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": duracion_frame, "redraw": False}, "fromcurrent": True}]
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
                        go.Scatter(x=x_vals[:idx+1], y=y_vals[:idx+1]), # La línea crece hasta el índice actual
                        go.Scatter(x=[x_vals[idx]], y=[y_vals[idx]])    # El punto rojo se sitúa al extremo
                    ],
                    name=str(tiempos[idx])
                ) for idx in indices_anim
            ]
        )
        
        # # Desactivar redibujado costoso en los frames
        for frame in fig_anim.frames:
            frame.layout = go.Layout(sliders=[])

        st.plotly_chart(fig_anim, use_container_width=True)

    except Exception as e:
        st.error(f"Error en la simulación: {e}")
