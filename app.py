import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Simulador Mecánico - Lagrangiano", layout="wide")

st.title("🎛️ Simulador Mecánico a partir del Lagrangiano")
st.markdown("""
Esta aplicación calcula la dinámica de un sistema a partir de su **Lagrangiano $L(q, \\dot{q})$** 
utilizando un solucionador numérico basado en diferencias finitas y el algoritmo de Euler-Cromer.
""")

# --- Entrada de datos en la barra lateral ---
st.sidebar.header("1. Definición del Sistema")

# Selección de Grados de Libertad (GDL)
gdl = st.sidebar.selectbox("Grados de Libertad (GDL)", [1, 2], index=1)

st.sidebar.subheader("Variables y Parámetros")
parámetros_str = st.sidebar.text_input("Parámetros constantes (separados por coma)", "m, k")
    
st.sidebar.markdown("**Expresión del Lagrangiano $L$:**")

# Formulario para construir el Lagrangiano limpiamente
with st.sidebar.form("lagrangian_form"):
    st.markdown("### Definir Lagrangiano")
    if gdl == 1:
        st.caption("Usa `q1` para la posición y `dq1` para la velocidad.")
        lagr_input = st.text_input("L =", "0.5 * m * dq1**2 - 0.5 * k * q1**2")
    else:
        st.caption("Usa `q1, q2` para posiciones y `dq1, dq2` para velocidades.")
        lagr_input = st.text_input("L =", "0.5 * m * (dq1**2 + dq2**2) - 0.5 * k * (q1**2 + q2**2 + (q2 - q1)**2)")
        
    st.markdown("### Condiciones Iniciales")
    q1_0 = st.number_input("q1 inicial", value=1.0)
    dq1_0 = st.number_input("dq1 inicial (velocidad)", value=0.0)
    if gdl == 2:
        q2_0 = st.number_input("q2 inicial", value=0.0)
        dq2_0 = st.number_input("dq2 inicial (velocidad)", value=0.0)
    
    st.markdown("### Parámetros de Simulación")
    t_max = st.number_input("Tiempo total (s)", value=10.0, min_value=1.0)
    dt = st.number_input("Paso de tiempo (dt)", value=0.01, min_value=0.0001, max_value=0.5, format="%.4f")
    
    # Valores de las constantes ingresadas por el usuario
    st.markdown("### Valores de los Parámetros")
    param_vals = {}
    if parámetros_str:
        for p in parámetros_str.split(","):
            p = p.strip()
            if p:
                param_vals[p] = st.number_input(f"Valor de {p}", value=1.0)

    calcular = st.form_submit_button("Calcular y Resolver")

if calcular:
    # --- PROCESAMIENTO SIMBÓLICO (SymPy) ---
    t = sp.Symbol('t')
    
    # Definir parámetros dinámicamente
    local_dict = {p: sp.Symbol(p) for p in param_vals.keys()}
    local_dict.update({'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'pi': sp.pi, 'exp': sp.exp})
    
    # Definir posiciones y velocidades dinámicamente como funciones del tiempo
    q = [sp.Function(f'q{i+1}')(t) for i in range(gdl)]
    dq = [q[i].diff(t) for i in range(gdl)]
    
    # Mapear los strings que escribe el usuario a variables de SymPy
    for i in range(gdl):
        local_dict[f'q{i+1}'] = q[i]
        local_dict[f'dq{i+1}'] = dq[i]
        
    try:
        # Parsear la ecuación escrita por el usuario
        L = sp.parse_expr(lagr_input, local_dict=local_dict)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🔬 Análisis Simbólico")
            st.latex(f"L = {sp.latex(L)}")
            
        # Calcular las fuerzas generalizadas (dL/dq)
        dL_dq = [L.diff(q[i]) for i in range(gdl)]
        
        with col1:
            st.subheader("Fuerzas Generalizadas ($Q_i = \\partial L / \\partial q_i$)")
            for i in range(gdl):
                st.latex(f"Q_{i+1} = {sp.latex(dL_dq[i])}")
                
        # Calcular la matriz de masa generalizada M_ij = d^2L / (d(dq_i) * d(dq_j))
        M_sym = sp.Matrix([[L.diff(dq[i]).diff(dq[j]) for j in range(gdl)] for i in range(gdl)])

        # --- PREPARACIÓN NUMÉRICA PARA EL BUCLE ---
        # Sustituir primero las variables constantes (m, k, etc.) por sus valores numéricos reales
        dL_dq_num = [expr.subs(param_vals) for expr in dL_dq]
        M_num_expr = M_sym.subs(param_vals)
        
        # Símbolos planos para evaluar con NumPy rápidamente mediante lambdify
        sym_q = [sp.Symbol(f'q{i+1}') for i in range(gdl)]
        sym_dq = [sp.Symbol(f'dq{i+1}') for i in range(gdl)]
        
        # Desvincular de las funciones del tiempo para evitar fallos de evaluación en arrays
        for i in range(gdl):
            for j in range(gdl):
                dL_dq_num[i] = dL_dq_num[i].subs(dq[j], sym_dq[j]).subs(q[j], sym_q[j])
                M_num_expr = M_num_expr.subs(dq[j], sym_dq[j]).subs(q[j], sym_q[j])
        
        # Convertir las expresiones de SymPy a funciones eficientes de NumPy
        func_dL_dq = [sp.lambdify((*sym_q, *sym_dq), dL_dq_num[i], 'numpy') for i in range(gdl)]
        func_M = sp.lambdify((*sym_q, *sym_dq), M_num_expr, 'numpy')

        # --- BUCLE DE INTEGRACIÓN NUMÉRICA (EULER-CROMER) ---
        tiempos = np.arange(0, t_max, dt)
        n_pasos = len(tiempos)
        
        # Inicializar matrices para almacenar los resultados del recorrido
        historial_q = np.zeros((gdl, n_pasos))
        historial_dq = np.zeros((gdl, n_pasos))
        
        # Asignar condiciones iniciales
        historial_q[0, 0] = q1_0
        historial_dq[0, 0] = dq1_0
        if gdl == 2:
            historial_q[1, 0] = q2_0
            historial_dq[1, 0] = dq2_0
            
        # Ejecución del paso temporal dinámico
        for k in range(n_pasos - 1):
            q_actual = historial_q[:, k]
            dq_actual = historial_dq[:, k]
            
            # Evaluar numéricamente la matriz de masa y el vector de fuerzas
            M_eval = np.array(func_M(*q_actual, *dq_actual), dtype=float)
            if gdl == 1: 
                M_eval = np.array([[M_eval]]) # Asegurar estructura matricial para 1 GDL
                
            Q_eval = np.array([f(*q_actual, *dq_actual) for f in func_dL_dq], dtype=float)
            
            # Resolver numéricamente las aceleraciones: M * ddq = Q (Evita el despeje analítico difícil)
            try:
                aceleraciones = np.linalg.solve(M_eval, Q_eval)
            except np.linalg.LinAlgError:
                aceleraciones = np.linalg.pinv(M_eval).dot(Q_eval)
            
            # Integración de Euler-Cromer (conserva de manera excelente la energía mecánica en osciladores)
            historial_dq[:, k+1] = dq_actual + aceleraciones * dt
            historial_q[:, k+1] = q_actual + historial_dq[:, k+1] * dt

        # --- GRÁFICOS ---
        with col2:
            st.header(f"📊 Solución Numérica ($\Delta t = {dt}$s)")
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Subplot 1: Evolución de las Posiciones
            for i in range(gdl):
                ax[0].plot(tiempos, historial_q[i, :], label=f"q{i+1} (Posición)", lw=2)
            ax[0].set_ylabel("Posición")
            ax[0].grid(True)
            ax[0].legend()
            ax[0].set_title("Evolución Temporal del Sistema")
            
            # Subplot 2: Evolución de las Velocidades
            for i in range(gdl):
                ax[1].plot(tiempos, historial_dq[i, :], label=f"dq{i+1} (Velocidad)", linestyle="--", lw=2)
            ax[1].set_ylabel("Velocidad")
            ax[1].set_xlabel("Tiempo (s)")
            ax[1].grid(True)
            ax[1].legend()
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error en la simulación numérica: {e}")
        st.info("Asegúrate de escribir la ecuación correctamente respetando las variables e indicando explícitamente los signos de multiplicación (*).")
