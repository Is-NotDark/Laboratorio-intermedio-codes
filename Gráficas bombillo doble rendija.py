import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import fresnel
import os

# 1. TUS DATOS EXPERIMENTALES
theta = np.array([-0.0058, -0.0056, -0.0054, -0.0052, -0.0050, -0.0048, -0.0046, -0.0044, -0.0042, -0.0040, -0.0038, -0.0036, -0.0034, -0.0032, -0.0030, -0.0028, -0.0026, -0.0024, -0.0022, -0.0020, -0.0018, -0.0016, -0.0014, -0.0012, -0.0010, -0.0008, -0.0006, -0.0004, -0.0002,  0.0000, 0.0002,  0.0004,  0.0006,  0.0008,  0.0010,  0.0012,  0.0014,  0.0016,  0.0018,  0.0020, 0.0022,  0.0024,  0.0026,  0.0028,  0.0030,  0.0032,  0.0034,  0.0036,  0.0038,  0.0040, 0.0042,  0.0044,  0.0046,  0.0048,  0.0050,  0.0052,  0.0054,  0.0056,  0.0058,  0.0060, 0.0062,  0.0064,  0.0066,  0.0068,  0.0070,  0.0072,  0.0074,  0.0076,  0.0078,  0.0080, 0.0082,  0.0084,  0.0086])

y_data = np.array([38.67, 46.00, 50.00, 57.33, 66.33, 70.67, 84.00, 81.67, 69.00, 57.67, 57.67, 81.00, 100.00, 130.67, 139.33, 116.33, 97.33, 91.00, 92.67, 135.00, 177.00, 196.00, 181.00, 152.33, 130.00, 113.33, 146.67, 192.00, 225.33, 244.67, 209.00, 170.00, 136.33, 125.33, 170.33, 206.33, 226.67, 235.67, 192.67, 150.00, 131.67, 141.67, 170.33, 185.67, 188.33, 186.00, 153.00, 116.00, 114.00, 110.00, 129.00, 142.33, 142.33, 117.00, 108.33, 84.67, 84.33, 82.67, 95.33, 88.67, 83.33, 81.00, 72.00, 63.33, 61.00, 55.67, 54.67, 52.33, 47.67, 45.67, 42.33, 40.67, 37.67])

y_err = np.sqrt(y_data) # Incertidumbre estadística

# CONSTANTES FIJAS
L = 0.5           # Distancia (m)
d = 0.356e-3     # Separación de rendijas (m)

# MODELO FRAUNHOFER OPTIMIZADO
def fraunhofer_fit(theta, I0, wavelength, theta0, fondo, a_fit):
    t = theta - theta0
    beta = (np.pi * d * np.sin(t)) / wavelength
    alpha = (np.pi * a_fit * np.sin(t)) / wavelength
    diffraction = np.where(alpha == 0, 1.0, (np.sin(alpha)/alpha)**2)
    return I0 * (np.cos(beta)**2) * diffraction + fondo

# MODELO FRESNEL OPTIMIZADO
def fresnel_fit(theta, I0, wavelength, theta0, fondo, a_fit):
    t = theta - theta0
    x = t * L
    k_fac = np.sqrt(2 / (wavelength * L))
    def get_amplitude(low, high, x_pos):
        v1, v2 = k_fac * (low - x_pos), k_fac * (high - x_pos)
        S_v2, C_v2 = fresnel(v2)
        S_v1, C_v1 = fresnel(v1)
        return (C_v2 - C_v1) + 1j * (S_v2 - S_v1)
    
    s1_l, s1_h = -d/2 - a_fit/2, -d/2 + a_fit/2
    s2_l, s2_h = d/2 - a_fit/2, d/2 + a_fit/2
    amp = get_amplitude(s1_l, s1_h, x) + get_amplitude(s2_l, s2_h, x)
    return I0 * (np.abs(amp)**2) + fondo

# AJUSTE DE PARÁMETROS (Guess inicial: I0, lambda, shift, fondo, a)
p0 = [190, 546e-9, 0.0, 65, 0.08e-3]

popt_fraun, _ = curve_fit(fraunhofer_fit, theta, y_data, p0=p0, sigma=y_err)
popt_fresn, _ = curve_fit(fresnel_fit, theta, y_data, p0=[1000, 546e-9, 0.0, 65, 0.08e-3], sigma=y_err)

# CÁLCULO DE RESIDUOS
res_fraun = y_data - fraunhofer_fit(theta, *popt_fraun)
res_fresn = y_data - fresnel_fit(theta, *popt_fresn)

# GRÁFICA COMBINADA (AJUSTE + RESIDUOS)
theta_line = np.linspace(min(theta), max(theta), 1000) # Línea suave para el modelo

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# --- Subplot 1: Ajustes ---
ax1.errorbar(theta, y_data, yerr=y_err, fmt='.', color='black', label='Datos Experimentales', alpha=0.6)
ax1.plot(theta_line, fraunhofer_fit(theta_line, *popt_fraun), 'r-', label=f'Fraunhofer ($\lambda$={popt_fraun[1]*1e9:.1f} nm)', linewidth=1.5)
ax1.plot(theta_line, fresnel_fit(theta_line, *popt_fresn), 'b--', label=f'Fresnel ($\lambda$={popt_fresn[1]*1e9:.1f} nm)', linewidth=1.5)
ax1.set_ylabel('Conteos por segundo')
ax1.set_title('Bombillo - Doble Rendija')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.2)

# --- Subplot 2: Residuos ---
ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax2.scatter(theta, res_fraun, color='red', s=10, label='Res. Fraunhofer')
ax2.scatter(theta, res_fresn, color='blue', marker='x', s=15, label='Res. Fresnel')
ax2.set_ylabel('Residuos (O-C)')
ax2.set_xlabel('Ángulo $\\theta$ (rad)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.2)

plt.tight_layout()

# Verificar si la carpeta existe, si no, crearla (opcional para evitar errores)
output_path = 'Laboratorio 1/Figuras/Bombillo doble rendija.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Resultados Fraunhofer: Lambda = {popt_fraun[1]*1e9:.2f} nm, Ancho a = {popt_fraun[4]*1e3:.3f} mm")
print(f"Resultados Fresnel: Lambda = {popt_fresn[1]*1e9:.2f} nm, Ancho a = {popt_fresn[4]*1e3:.3f} mm")