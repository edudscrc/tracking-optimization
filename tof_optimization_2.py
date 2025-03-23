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


def generate_delays(x_w, r_w, t0_w):
	global c, x_sensors
	return t0_w + np.sqrt((x_w - x_sensors)**2 + r_w**2) / c


def costfun(params, delays):
	x_w, r_w, t0_w = params
	return np.sum((delays - generate_delays(x_w, r_w, t0_w))**2)


# real_delays0 = generate_delays(1856, 0)

x_whale = x_sensors[apex]

print(f'{x_whale = } m')

r_whale = 500

t0_whale = 2

res = minimize(costfun, x0=(x_whale, r_whale, t0_whale), args=real_delays, tol=1e-10)

print(f'x_whale: {res.x[0]}')
print(f'r_whale: {res.x[1]}')
print(f't0_whale: {res.x[2]}')

synthetic_delays = generate_delays(res.x[0], res.x[1], res.x[2])

# np.save('synthetic_delays_19000_px.npy', synthetic_delays * fs)

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
