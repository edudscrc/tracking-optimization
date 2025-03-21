import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from my_functions import m_imshow, m_plot, load_data

data, spatial_position_start, spatial_position_end, time_start, time_end = load_data('aquisicao_40km_50ns_21_10_2024_ds1_19000m_22500m_tds1_40730s_40745s_fs899Hz_bandpass_43Hz_85Hz_medfilt_47_4')
fs = 900
apex = 7323
c = 1500.

real_delays = np.load('./delays_px.npy')
real_delays = real_delays / fs

x_sensors = np.linspace(spatial_position_start, spatial_position_end, data.shape[0])

real_delays_diff = np.gradient(real_delays, x_sensors)

m_plot(x_arr=x_sensors, data=real_delays_diff, title='Real Delays - First Derivative', xlabel='Spatial Position [m]', ylabel='seconds per meter [s/m]', xticks=np.linspace(x_sensors.min(), x_sensors.max(), 10), filename='./real_delays_diff.png')

# Fixa a profundidade da baleia (referências usam 20 m a 30 m)
z_whale = 30


def generate_delays(x_w, r_w, z_c):
	global c, x_sensors, z_whale
	return np.sqrt((x_w - x_sensors)**2 + r_w**2 + (z_whale - z_c)**2) / c


def costfun(params, delays):
	x_w, r_w, z_c = params
	return np.sum((delays - generate_delays(x_w, r_w, z_c))**2)


# Estimativas iniciais
x_whale = x_sensors[apex]
r_whale = 500
z_c = 150


bounds = [(x_sensors.min(), x_sensors.max()), (50, 5000), (100, 300)]
res = differential_evolution(costfun, bounds, args=(real_delays,), tol=1e-10)

# res = minimize(costfun, x0=(x_whale, r_whale, z_c), args=real_delays, tol=1e-10)

print(f'x_whale: {res.x[0]}')
print(f'r_whale: {res.x[1]}')
print(f'z_cable: {res.x[2]}')

synthetic_delays = generate_delays(res.x[0], res.x[1], res.x[2])

# Inverte a direção dos delays sintéticos (não sei é correto fazer isso)
synthetic_delays *= -1.

plt.rcParams.update({'font.size': 6})
plt.figure()
plt.plot(x_sensors, real_delays, label='True delays')
plt.plot(x_sensors, synthetic_delays, label='Synthetic delays')
plt.ylabel('Time [s]')
plt.xlabel('Spatial Position [m]')
plt.legend()
plt.grid(True)
plt.show()
