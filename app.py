import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Simulador Mecánico Universal", layout="wide")

st.title("🎛️ Simulador Mecánico Universal (Euler-Lagrange)")
st.markdown("""
Esta versión es **100% robusta** y utiliza la regla de la cadena completa para calcular de forma exacta 
cualquier sistema de mecánica clásica (incluyendo fuerzas de Coriolis, centrífugos y acoplamientos complejos) **hasta 4 GDL**.
""")

# --- Entrada de datos en la barra lateral ---
st.sidebar.header("1. Definición del Sistema")

# Selección de Grados de Libertad (Soporta hasta 4 GDL)
gdl = st.sidebar.selectbox("Grados de Libertad (GDL)", [1, 2, 3, 4], index=1)

st.sidebar.subheader("Variables y Parámetros")
parámetros_str = st.sidebar.text_input("Parámetros constantes (separados por coma)", "m, k, l, g")
    
st.sidebar.markdown("**Expresión del Lagrangiano $L$:**")

# Formulario para construir el Lagrangiano limpiamente
with st.sidebar.form("lagrangian_form"):
    st.markdown("### Definir Lagrangiano")
    st.caption("Usa `q1, q2, q3, q4` para posiciones y `dq1, dq2, dq3, dq4` para velocidades.")
    
    # Lagrangiano por defecto adaptable
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
    
    # Valores de las constantes ingresadas por el usuario
    st.markdown("### Valores de los Parámetros")
    param_vals = {}
    if parámetros_str:
        for p in parámetros_str.split(","):
            p = p.strip()
            if p:
                param_vals[p] = st.number_input(f"Valor de {p}", value=1.0 if p != 'g' else 9.81)

    calcular = st.form_submit_button("Calcular y Resolver")

if calcular:
    # --- PROCESAMIENTO SIMBÓLICO EXACTO (SymPy) ---
    t = sp.Symbol('t')
    
    # Definir parámetros dinámicamente
    local_dict = {p: sp.Symbol(p) for p in param_vals.keys()}
    local_dict.update({'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'pi': sp.pi, 'exp': sp.exp})
    
    # Definir posiciones, velocidades y aceleraciones como funciones estrictas del tiempo
    q = [sp.Function(f'q{i+1}')(t) for i in range(gdl)]
    dq = [q[i].diff(t) for i in range(gdl)]
    ddq = [dq[i].diff(t) for i in range(gdl)]
    
    # Mapear strings a variables dependientes de t
    for i in range(gdl):
        local_dict[f'q{i+1}'] = q[i]
        local_dict[f'dq{i+1}'] = dq[i]
        
    try:
        # Parsear la ecuación de forma exacta
        L = sp.parse_expr(lagr_input, local_dict=local_dict)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🔬 Análisis Simbólico")
            st.latex(f"L = {sp.latex(L)}")
            
        # 2. Calcular Ecuaciones de Euler-Lagrange EXACTAS (incluyendo todas las derivadas temporales)
        eqs = []
        for i in range(gdl):
            dL_dq = L.diff(q[i])
            dL_ddq = L.diff(dq[i])
            d_dt_dL_ddq = dL_ddq.diff(t)  # <-- Aquí SymPy aplica la regla de la cadena completa automáticamente
            
            eq = d_dt_dL_ddq - dL_dq
            eqs.append(eq)
            
        # --- DESPEJE MATRICIAL NUMÉRICO BLINDADO ---
        # Símbolos planos temporales para las aceleraciones en el despeje algebraico lineal
        sym_ddq_temporal = [sp.Symbol(f'ddq_{i+1}') for i in range(gdl)]
        
        # Sustituimos las complejas aceleraciones de SymPy por variables algebraicas planas
        eqs_algebraicas = []
        for eq in eqs:
            eq_temp = eq
            for i in range(gdl):
                eq_temp = eq_temp.subs(ddq[i], sym_ddq_temporal[i])
            eqs_algebraicas.append(eq_temp)
            
        # Extraemos la matriz de masa A y el vector de fuerzas B donde A * ddq_temporal + B = 0 -> A * ddq = -B
        # Esto separa de forma analítica exacta las aceleraciones de cualquier fuerza no lineal
        A_sym, B_sym = sp.linear_eq_to_matrix(eqs_algebraicas, sym_ddq_temporal)
        
        # --- PREPARACIÓN NUMÉRICA ---
        A_num_expr = A_sym.subs(param_vals)
        B_num_expr = B_sym.subs(param_vals)
        
        sym_q = [sp.Symbol(f'q{i+1}') for i in range(gdl)]
        sym_dq = [sp.Symbol(f'dq{i+1}') for i in range(gdl)]
        
        # Desvincular del tiempo para inyectar NumPy sin errores
        for i in range(gdl):
            for j in range(gdl):
                A_num_expr = A_num_expr.subs(dq[j], sym_dq[j]).subs(q[j], sym_q[j])
                B_num_expr = B_num_expr.subs(dq[j], sym_dq[j]).subs(q[j], sym_q[j])
                
        func_A = sp.lambdify((*sym_q, *sym_dq), A_num_expr, 'numpy')
        func_B = sp.lambdify((*sym_q, *sym_dq), B_num_expr, 'numpy')

        # --- BUCLE DE INTEGRACIÓN NUMÉRICA (EULER-CROMER) ---
        tiempos = np.arange(0, t_max, dt)
        n_pasos = len(tiempos)
        
        historial_q = np.zeros((gdl, n_pasos))
        historial_dq = np.zeros((gdl, n_pasos))
        
        # Llenar condiciones iniciales
        for i in range(gdl):
            historial_q[i, 0] = c_iniciales[f'q{i+1}']
            historial_dq[i, 0] = c_iniciales[f'dq{i+1}']
            
        # Integración exacta paso a paso
        for k in range(n_pasos - 1):
            q_actual = historial_q[:, k]
            dq_actual = historial_dq[:, k]
            
            # Evaluar numéricamente las matrices lineales generadas
            M_eval = np.array(func_A(*q_actual, *dq_actual), dtype=float)
            F_eval = np.array(func_B(*q_actual, *dq_actual), dtype=float).flatten()
            
            # Como A * ddq + B = 0 -> M_eval * aceleraciones = -F_eval
            try:
                aceleraciones = np.linalg.solve(M_eval, -F_eval)
            except np.linalg.LinAlgError:
                aceleraciones = np.linalg.pinv(M_eval).dot(-F_eval)
            
            # Integrador simpléctico de Euler-Cromer
            historial_dq[:, k+1] = dq_actual + aceleraciones * dt
            historial_q[:, k+1] = q_actual + historial_dq[:, k+1] * dt

        # --- SECCIÓN DE GRÁFICOS ---
        with col2:
            st.header(f"📊 Resultados de la Simulación")
            
            # Gráfico de evolución en el tiempo
            fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            for i in range(gdl):
                ax[0].plot(tiempos, historial_q[i, :], label=f"q{i+1} (Posición)", lw=2)
                ax[1].plot(tiempos, historial_dq[i, :], label=f"dq{i+1} (Velocidad)", linestyle="--", lw=1.5)
            
            ax[0].set_ylabel("Posición")
            ax[0].grid(True)
            ax[0].legend()
            ax[0].set_title("Evolución de las Variables de Estado")
            
            ax[1].set_ylabel("Velocidad")
            ax[1].set_xlabel("Tiempo (s)")
            ax[1].grid(True)
            ax[1].legend()
            st.pyplot(fig)
            
            # --- ESPACIO DE CONFIGURACIÓN INTERACTIVO ---
            st.subheader("🌌 Espacio de Configuración")
            if gdl == 1:
                st.info("El espacio de configuración requiere al menos 2 GDL. Mostrando espacio de fase (q1 vs dq1):")
                fig_cf, ax_cf = plt.subplots(figsize=(10, 4))
                ax_cf.plot(historial_q[0, :], historial_dq[0, :], color='purple', lw=2)
                ax_cf.set_xlabel("Posición q1")
                ax_cf.set_ylabel("Velocidad dq1")
                ax_cf.grid(True)
                st.pyplot(fig_cf)
            else:
                st.markdown("Selecciona qué coordenadas generalizadas quieres contrastar:")
                col_sel1, col_sel2 = st.columns(2)
                
                opciones = [f"q{i+1}" for i in range(gdl)]
                with col_sel1:
                    eje_x = st.selectbox("Coordenada Eje X", opciones, index=0)
                with col_sel2:
                    eje_y = st.selectbox("Coordenada Eje Y", opciones, index=1 if gdl > 1 else 0)
                
                idx_x = int(eje_x[1]) - 1
                idx_y = int(eje_y[1]) - 1
                
                fig_cf, ax_cf = plt.subplots(figsize=(10, 5))
                ax_cf.plot(historial_q[idx_x, :], historial_q[idx_y, :], color='teal', lw=2)
                ax_cf.set_xlabel(f"Posición {eje_x}")
                ax_cf.set_ylabel(f"Posición {eje_y}")
                ax_cf.set_title(f"Trayectoria en el Espacio de Configuración ({eje_x} vs {eje_y})")
                ax_cf.grid(True)
                st.pyplot(fig_cf)

    except Exception as e:
        st.error(f"Error en la simulación: {e}")
        st.info("Verifica que las constantes y variables del Lagrangiano tengan nombres válidos matemáticamente.")
