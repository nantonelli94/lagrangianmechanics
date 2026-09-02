import streamlit as st
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Generador de Ecuaciones de Euler-Lagrange", layout="wide")

st.title("🎛️ Simulador Mecánico a partir del Lagrangiano")
st.markdown("""
Esta aplicación calcula las **ecuaciones diferenciales del movimiento** a partir del Lagrangiano $L(q, \dot{q})$ 
mediante las ecuaciones de Euler-Lagrange, las resuelve numéricamente y gráfica sus trayectorias.
""")

# --- Entrada de datos ---
st.sidebar.header("1. Definición del Sistema")

# Coordenadas (por simplicidad, asumimos un sistema de hasta 2 grados de libertad)
gdl = st.sidebar.selectbox("Grados de Libertad (GDL)", [1, 2], index=1)

st.sidebar.subheader("Variables y Parámetros")
parámetros_str = st.sidebar.text_input("Parámetros constantes (separados por coma)", "m, g, l")

# Nombres de variables por defecto
q_names = ["theta", "x"] if gdl == 2 else ["theta"]
    
st.sidebar.markdown("**Expresión del Lagrangiano $L$:**")
st.sidebar.caption("Usa `diff(q, t)` para velocidades o define alias simplificados.")

# Formulario para construir el Lagrangiano limpiamente
with st.sidebar.form("lagrangian_form"):
    st.markdown("### Definir Lagrangiano")
    if gdl == 1:
        st.caption("Usa `q1` para la posición y `dq1` para la velocidad.")
        lagr_input = st.text_input("L =", "0.5 * m * l**2 * dq1**2 - m * g * l * (1 - cos(q1))")
    else:
        st.caption("Usa `q1, q2` para posiciones y `dq1, dq2` para velocidades.")
        lagr_input = st.text_input("L =", "0.5 * m * (dq1**2 + dq2**2) - m * g * q2")
        
    # Condiciones iniciales y tiempo
    st.markdown("### Condiciones Iniciales")
    q1_0 = st.number_input("q1 inicial", value=1.0)
    dq1_0 = st.number_input("dq1 inicial (velocidad)", value=0.0)
    if gdl == 2:
        q2_0 = st.number_input("q2 inicial", value=0.0)
        dq2_0 = st.number_input("dq2 inicial (velocidad)", value=0.0)
    
    t_max = st.number_input("Tiempo total de simulación (s)", value=10.0, min_value=1.0)
    
    # Valores de las constantes
    st.markdown("### Valores de los Parámetros")
    param_vals = {}
    if parámetros_str:
        for p in parámetros_str.split(","):
            p = p.strip()
            if p:
                param_vals[p] = st.number_input(f"Valor de {p}", value=1.0 if p != 'g' else 9.81)

    calcular = st.form_submit_button("Calcular y Resolver")

if calcular:
    # --- PROCESAMIENTO SIMBÓLICO (SymPy) ---
    t = sp.Symbol('t')
    
    # Definir parámetros dinámicamente
    local_dict = {p: sp.Symbol(p) for p in param_vals.keys()}
    # Funciones trigonométricas estándar y comunes
    local_dict.update({'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'pi': sp.pi, 'exp': sp.exp})
    
    # Definir posiciones y velocidades dinámicamente
    q = [sp.Function(f'q{i+1}')(t) for i in range(gdl)]
    dq = [q[i].diff(t) for i in range(gdl)]
    ddq = [dq[i].diff(t) for i in range(gdl)]
    
    # Mapeo de texto ingresado a variables de SymPy
    for i in range(gdl):
        local_dict[f'q{i+1}'] = q[i]
        local_dict[f'dq{i+1}'] = dq[i]
        
    try:
        L = sp.parse_expr(lagr_input, local_dict=local_dict)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🔬 Análisis Simbólico")
            st.latex(f"L = {sp.latex(L)}")
            
        # Calcular Ecuaciones de Euler-Lagrange
        eqs = []
        for i in range(gdl):
            dL_dq = L.diff(q[i])
            dL_ddq = L.diff(dq[i])
            d_dt_dL_ddq = dL_ddq.diff(t)
            
            eq = d_dt_dL_ddq - dL_dq
            eqs.append(eq)
            
        # Despejar las aceleraciones (ddq)
        aceleraciones = sp.solve(eqs, ddq)
        
        with col1:
            st.subheader("Ecuaciones del Movimiento (Euler-Lagrange)")
            for i in range(gdl):
                st.latex(f"\\frac{{d}}{{dt}}\\left(\\frac{{\\partial L}}{{\\partial \\dot{{q}}_{i+1}}}\\right) - \\frac{{\\partial L}}{{\\partial q_{i+1}}} = 0")
            
            st.subheader("Aceleraciones Despejadas:")
            for i in range(gdl):
                st.latex(f"\\ddot{{q}}_{i+1} = {sp.latex(aceleraciones[ddq[i]])}")

        # --- RESOLUCIÓN NUMÉRICA (SciPy) ---
        # Reemplazar parámetros por sus valores numéricos en las expresiones de aceleración
        acel_num_expr = [aceleraciones[ddq[i]].subs(param_vals) for i in range(gdl)]
        
        # Convertir variables simbólicas a argumentos numéricos planos para lambdify
        sym_q = [sp.Symbol(f'q{i+1}') for i in range(gdl)]
        sym_dq = [sp.Symbol(f'dq{i+1}') for i in range(gdl)]
        
        # Sustituir funciones del tiempo por símbolos planos para evaluar numéricamente de forma segura
        for i in range(gdl):
            acel_num_expr[i] = acel_num_expr[i].subs({dq[i]: sym_dq[i], q[i]: sym_q[i]})
            
        # Crear funciones numéricas rápidas (lambdify)
        # El orden de los argumentos para la función evaluadora será: (q1, ..., qN, dq1, ..., dqN)
        acel_funcs = [sp.lambdify((*sym_q, *sym_dq), acel_num_expr[i], 'numpy') for i in range(gdl)]
        
        # Definir el sistema de EDOs de primer orden para solve_ivp
        def sistema_dinamico(t_val, y):
            # y contiene = [q1, ..., qN, dq1, ..., dqN]
            q_vals = y[:gdl]
            dq_vals = y[gdl:]
            
            # Las derivadas de las posiciones son las velocidades
            dy_dt = list(dq_vals)
            
            # Las derivadas de las velocidades son las aceleraciones evaluadas numéricamente
            for i in range(gdl):
                acel = acel_funcs[i](*q_vals, *dq_vals)
                dy_dt.append(acel)
                
            return dy_dt

        # Vector de condiciones iniciales
        if gdl == 1:
            y0 = [q1_0, dq1_0]
        else:
            y0 = [q1_0, q2_0, dq1_0, dq0_0 if 'dq0_0' in locals() else dq2_0]
            
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 1000)
        
        # Resolver EDO
        sol = solve_ivp(sistema_dinamico, t_span, y0, t_eval=t_eval, method='RK45')
        
        # --- GRÁFICOS (Matplotlib / Streamlit) ---
        with col2:
            st.header("📊 Solución Numérica")
            
            # Gráfico de Posiciones
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            for i in range(gdl):
                ax[0].plot(sol.t, sol.y[i], label=f"q{i+1} (Posición)")
            ax[0].set_ylabel("Posición")
            ax[0].grid(True)
            ax[0].legend()
            ax[0].set_title("Evolución de las Coordenadas Generalizadas")
            
            # Gráfico de Velocidades
            for i in range(gdl):
                ax[1].plot(sol.t, sol.y[gdl + i], label=f"dq{i+1} (Velocidad)", linestyle="--")
            ax[1].set_ylabel("Velocidad")
            ax[1].set_xlabel("Tiempo (s)")
            ax[1].grid(True)
            ax[1].legend()
            
            st.pyplot(fig)
            
            # Espacio de fase (si tiene 1 GDL)
            if gdl == 1:
                fig_fase, ax_fase = plt.subplots(figsize=(6, 4))
                ax_fase.plot(sol.y[0], sol.y[1], color='purple')
                ax_fase.set_title("Espacio de Fase")
                ax_fase.set_xlabel("Posición (q1)")
                ax_fase.set_ylabel("Velocidad (dq1)")
                ax_fase.grid(True)
                st.pyplot(fig_fase)

    except Exception as e:
        st.error(f"Error al procesar el Lagrangiano o resolver las ecuaciones: {e}")
        st.info("Asegúrate de escribir la ecuación en formato Python matemático correcto. Ej: `0.5 * m * dq1**2` y usar las variables indicadas.")
