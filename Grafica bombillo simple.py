import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import fresnel
import os

# ==========================================
# 1. DATOS EXPERIMENTALES (Incrustados)
# ==========================================
# Posición del sensor en mm
x_mm = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 
    3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 
    4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 
    5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 
    6.0, 6.1, 6.2, 6.3
])

# Promedio de conteos/s
y_data = np.array([
    43.0, 48.67, 57.0, 61.33, 64.0, 72.33, 77.0, 80.0, 87.0, 98.67, 
    103.0, 108.67, 119.67, 123.0, 135.67, 133.33, 143.67, 149.67, 146.33, 150.0, 
    157.33, 161.67, 172.33, 167.67, 174.67, 166.67, 176.67, 178.67, 179.67, 182.0, 
    176.0, 171.67, 154.33, 152.67, 154.67, 158.33, 158.33, 144.0, 134.33, 138.67, 
    128.67, 120.0, 120.67, 120.0, 117.33, 106.67, 102.0, 102.67, 98.33, 84.33, 
    81.67, 76.67, 75.67, 75.33, 65.67, 59.0, 58.33, 60.0, 51.33, 47.67, 
    47.0, 41.67, 43.33, 37.0
])

# Incertidumbre
y_err = np.array([
    3.79, 4.03, 4.36, 4.52, 4.62, 4.91, 5.07, 5.16, 5.39, 5.73, 
    5.86, 6.02, 6.32, 6.40, 6.72, 6.67, 6.92, 7.06, 6.98, 7.07, 
    7.24, 7.34, 7.58, 7.48, 7.63, 7.45, 7.67, 7.72, 7.74, 7.79, 
    7.66, 7.56, 7.17, 7.13, 7.18, 7.26, 7.26, 6.93, 6.69, 6.80, 
    6.55, 6.32, 6.34, 6.32, 6.25, 5.96, 5.83, 5.85, 5.73, 5.30, 
    5.22, 5.06, 5.02, 5.01, 4.68, 4.43, 4.41, 4.47, 4.14, 3.99, 
    3.96, 3.73, 3.80, 3.51
])

# CONSTANTES DEL EXPERIMENTO
L = 0.5           # Distancia Pantalla-Rendija (m)
a_fixed = 0.076e-3 # Ancho de rendija (m) - Valor que encontramos antes

# ==========================================
# 2. PROCESAMIENTO INICIAL (Centrado)
# ==========================================
# Estimamos el centro encontrando el índice del valor máximo
idx_max = np.argmax(y_data)
x_center_estimado = x_mm[idx_max]

# Convertimos a radianes preliminares: theta = (x - x_centro) / L
# Dejamos que el ajuste refine el 'theta0' exacto
theta_raw = (x_mm - x_center_estimado) * 1e-3 / L 

# ==========================================
# 3. DEFINICIÓN DE MODELOS (Rendija Simple)
# ==========================================

# --- FRAUNHOFER ---
def fraunhofer_single(theta, I0, wavelength, theta0, background):
    # theta0 ajusta el pequeño error de centrado
    t = theta - theta0
    beta = (np.pi * a_fixed * np.sin(t)) / wavelength
    # Sinc(x) = sin(x)/x. Numpy lo maneja, pero definimos manualmente para control
    # sinc_sq = (np.sin(beta)/beta)**2 con cuidado en beta=0
    val = np.where(beta == 0, 1.0, np.sin(beta)/beta)
    return I0 * (val**2) + background

# --- FRESNEL ---
def fresnel_single(theta, I0, wavelength, theta0, background):
    t = theta - theta0
    # Coordenada en pantalla x correspondiente al ángulo t
    x_pos = t * L 
    
    # Factor de escala de Fresnel
    # u = sqrt(2 / lambda L) * (xi - x)
    k_fac = np.sqrt(2 / (wavelength * L))
    
    # Límites de integración: desde el borde izquierdo (-a/2) al derecho (+a/2) de la rendija
    # u1 = k_fac * (-a/2 - x_pos)
    # u2 = k_fac * ( a/2 - x_pos)
    u1 = k_fac * (-a_fixed/2 - x_pos)
    u2 = k_fac * ( a_fixed/2 - x_pos)
    
    S1, C1 = fresnel(u1)
    S2, C2 = fresnel(u2)
    
    # Intensidad ~ | (C2-C1) + i(S2-S1) |^2
    intensity_raw = (C2 - C1)**2 + (S2 - S1)**2
    
    return I0 * intensity_raw + background

# ==========================================
# 4. AJUSTE DE CURVAS
# ==========================================
# Parámetros iniciales: [I0, lambda, theta0, fondo]
# Usamos 550 nm (verde) como semilla inicial
p0 = [180, 550e-9, 0.0, 40]

# Ajuste Fraunhofer
popt_fra, pcov_fra = curve_fit(fraunhofer_single, theta_raw, y_data, p0=p0, sigma=y_err, absolute_sigma=True)
perr_fra = np.sqrt(np.diag(pcov_fra))

# Ajuste Fresnel
popt_fre, pcov_fre = curve_fit(fresnel_single, theta_raw, y_data, p0=p0, sigma=y_err, absolute_sigma=True)
perr_fre = np.sqrt(np.diag(pcov_fre))

# ==========================================
# 5. GRAFICAR Y RESULTADOS
# ==========================================

# Eje X para graficar (Centrado perfecto en 0 usando el theta0 ajustado por Fraunhofer)
theta_plot = theta_raw - popt_fra[2]

# Generar línea suave para el modelo
theta_smooth = np.linspace(min(theta_plot), max(theta_plot), 1000)

# Evaluar modelos en la línea suave
# Nota: Sumamos popt_fra[2] para "des-centrar" al evaluar la función, ya que la función espera theta_raw
y_fra_smooth = fraunhofer_single(theta_smooth + popt_fra[2], *popt_fra)
y_fre_smooth = fresnel_single(theta_smooth + popt_fra[2], *popt_fre)

# Evaluar modelos en los puntos experimentales (para residuos)
y_model_fra = fraunhofer_single(theta_raw, *popt_fra)
y_model_fre = fresnel_single(theta_raw, *popt_fre)

# Residuos
res_fra = y_data - y_model_fra
res_fre = y_data - y_model_fre

# --- FIGURA ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Subplot 1: Datos y Ajustes
ax1.errorbar(theta_plot, y_data, yerr=y_err, fmt='.', color='black', label='Datos Experimentales', alpha=0.5)
ax1.plot(theta_smooth, y_fra_smooth, 'r-', label=f'Fraunhofer ($\lambda$={popt_fra[1]*1e9:.1f} nm)', linewidth=1.5)
ax1.plot(theta_smooth, y_fre_smooth, 'b--', label=f'Fresnel ($\lambda$={popt_fre[1]*1e9:.1f} nm)', linewidth=1.5)
ax1.set_ylabel('Intensidad (conteos/s)')
ax1.set_title(f'Bombillo - Rendija Simple ($a={a_fixed*1e3:.3f}$ mm)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)

# Subplot 2: Residuos
ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax2.scatter(theta_plot, res_fra, color='red', s=15, label='Res. Fraunhofer', alpha=0.7)
ax2.scatter(theta_plot, res_fre, color='blue', marker='x', s=15, label='Res. Fresnel', alpha=0.7)
ax2.set_ylabel('Residuos (O-C)')
ax2.set_xlabel('Ángulo $\\theta$ (rad)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.2)

plt.tight_layout()

# Guardar
output_path = 'Laboratorio 1/Figuras/Bombillo rendija simple.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

# --- IMPRIMIR RESULTADOS ---
print("="*50)
print(f"RESULTADOS RENDIJA SIMPLE (a = {a_fixed*1e3:.3f} mm)")
print("="*50)
print("MODELO FRAUNHOFER:")
print(f"  Lambda: {popt_fra[1]*1e9:.2f} +/- {perr_fra[1]*1e9:.2f} nm")
print(f"  Error relativo: {(perr_fra[1]/popt_fra[1])*100:.2f}%")
print("-" * 50)
print("MODELO FRESNEL:")
print(f"  Lambda: {popt_fre[1]*1e9:.2f} +/- {perr_fre[1]*1e9:.2f} nm")
print(f"  Error relativo: {(perr_fre[1]/popt_fre[1])*100:.2f}%")
print("="*50)