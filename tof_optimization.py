import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from my_functions import m_imshow, m_plot, load_data

data, spatial_position_start, spatial_position_end, time_start, time_end = load_data('aquisicao_40km_50ns_21_10_2024_ds1_19000m_22500m_tds1_40730s_40745s_fs899Hz_bandpass_43Hz_85Hz_medfilt_47_4')

fs = 900
c = 1500.

apex = 7323

real_delays = np.load('./delays_19000_px.npy')
real_delays = real_delays / fs
real_delays *= -1.

x_sensors = np.linspace(spatial_position_start, spatial_position_end, data.shape[0])

z_w = 30

def generate_delays(x_w, r_w, z_c):
	global c, x_sensors, z_w
	return np.sqrt((x_w - x_sensors)**2 + r_w**2 + (z_w - z_c)**2) / c


def costfun(params, delays):
	x_w, r_w, z_c = params
	return np.sum((delays - generate_delays(x_w, r_w, z_c))**2)


# real_delays0 = generate_delays(x_sensors[apex], 0)

x_whale = x_sensors[apex]

print(f'{x_whale = } m')

r_whale = 10000
z_cable = 1000

res = minimize(costfun, x0=(x_whale, r_whale, z_cable), args=real_delays, tol=1e-10)

print(f'x_whale: {res.x[0]}')
print(f'r_whale: {res.x[1]}')
print(f'z_cable: {res.x[2]}')

synthetic_delays = generate_delays(res.x[0], res.x[1], res.x[2])

plt.rcParams.update({'font.size': 6})
plt.figure()
plt.plot(x_sensors, real_delays, label='True delays')
# plt.plot(x_sensors, real_delays0, label='True generated delays')
plt.plot(x_sensors, synthetic_delays, label='Synthetic delays')
plt.ylabel('Time [s]')
plt.xlabel('Spatial Position [m]')
plt.legend()
plt.grid(True)
plt.show()
