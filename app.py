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


    try:
        # 1. Parsear la expresión de forma segura
        L = sp.parse_expr(lagr_input, local_dict=local_dict)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🔬 Análisis Simbólico")
            st.latex(f"L = {sp.latex(L)}")
            
        # 2. Calcular Ecuaciones de Euler-Lagrange
        eqs = []
        for i in range(gdl):
            dL_dq = L.diff(q[i])
            dL_ddq = L.diff(dq[i])
            d_dt_dL_ddq = dL_ddq.diff(t)
            
            eq = d_dt_dL_ddq - dL_dq
            eqs.append(eq)
            
        # 3. Despejar las aceleraciones (ddq)
        aceleraciones = sp.solve(eqs, ddq)
        
        # Validación en caso de sistemas altamente acoplados
        if not aceleraciones:
            raise ValueError("No se pudieron despejar analíticamente las aceleraciones. Revisa el acoplamiento de las velocidades.")

        with col1:
            st.subheader("Ecuaciones del Movimiento")
            for i in range(gdl):
                st.latex(f"\\frac{{d}}{{dt}}\\left(\\frac{{\\partial L}}{{\\partial \\dot{{q}}_{i+1}}}\\right) - \\frac{{\\partial L}}{{\\partial q_{i+1}}} = 0")
            
            st.subheader("Aceleraciones Despejadas:")
            for i in range(gdl):
                st.latex(f"\\ddot{{q}}_{i+1} = {sp.latex(aceleraciones[ddq[i]])}")

        # --- RESOLUCIÓN NUMÉRICA (SciPy) ---
        # Sustituir primero los parámetros constantes por sus valores numéricos reales
        acel_num_expr = [aceleraciones[ddq[i]].subs(param_vals) for i in range(gdl)]
        
        # Variables ficticias planas para desvincular las funciones t de SymPy antes de lambdify
        sym_q = [sp.Symbol(f'q{i+1}') for i in range(gdl)]
        sym_dq = [sp.Symbol(f'dq{i+1}') for i in range(gdl)]
        
        # Sustitución explícita en orden inverso (velocidades primero, luego posiciones) para evitar colisiones
        for i in range(gdl):
            for j in range(gdl):
                acel_num_expr[i] = acel_num_expr[i].subs(dq[j], sym_dq[j])
                acel_num_expr[i] = acel_num_expr[i].subs(q[j], sym_q[j])
            
        # Generar funciones ejecutables por NumPy de forma segura
        acel_funcs = [sp.lambdify((*sym_q, *sym_dq), acel_num_expr[i], 'numpy') for i in range(gdl)]
        
        # Definir el sistema de EDOs
        def sistema_dinamico(t_val, y):
            q_vals = y[:gdl]
            dq_vals = y[gdl:]
            
            dy_dt = list(dq_vals)
            
            for i in range(gdl):
                # Forzar a que los argumentos se expandan correctamente en la tupla
                acel = acel_funcs[i](*q_vals, *dq_vals)
                # Si el resultado es una constante pura (ej: 0.0), convertir a float de Python para evitar errores con SciPy
                if isinstance(acel, (np.ndarray, list)) and len(acel) == 1:
                    acel = acel[0]
                dy_dt.append(float(acel))
                
            return dy_dt

        # Vector de condiciones iniciales (CORREGIDO SIN VARIABLE ERRONEA)
        if gdl == 1:
            y0 = [q1_0, dq1_0]
        else:
            y0 = [q1_0, q2_0, dq1_0, dq2_0]
            
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 1000)
        
        # Resolver EDO
        sol = solve_ivp(sistema_dinamico, t_span, y0, t_eval=t_eval, method='RK45')
        
        # --- GRÁFICOS ---
        with col2:
            st.header("📊 Solución Numérica")
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Posiciones
            for i in range(gdl):
                ax[0].plot(sol.t, sol.y[i], label=f"q{i+1} (Posición)", lw=2)
            ax[0].set_ylabel("Posición")
            ax[0].grid(True)
            ax[0].legend()
            ax[0].set_title("Evolución de las Coordenadas Generalizadas")
            
            # Velocidades
            for i in range(gdl):
                ax[1].plot(sol.t, sol.y[gdl + i], label=f"dq{i+1} (Velocidad)", linestyle="--", lw=2)
            ax[1].set_ylabel("Velocidad")
            ax[1].set_xlabel("Tiempo (s)")
            ax[1].grid(True)
            ax[1].legend()
            
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al procesar el Lagrangiano o resolver las ecuaciones: {e}")

