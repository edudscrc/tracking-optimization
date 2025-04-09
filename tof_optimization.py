import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_delays(x_w, r_w, t0_w, x_sensors):
	c = 1500.
	return t0_w + np.sqrt((x_w - x_sensors)**2 + r_w**2) / c


def costfun(params, delays):
	global x_sensors
	x_w, r_w, t0_w = params
	return np.sum((delays - generate_delays(x_w, r_w, t0_w, x_sensors))**2)


# Consider r_sensors = 0

# 7000 metros em 27376 pixels
x_sensors = np.linspace(0, 7000, 27376, dtype=float)

# Parâmetros da baleia real SIMULADA
real_simulated_whale = {
	'x': 4070,
	'r': 400,
	't0': -0.38,
}
real_simulated_delays = generate_delays(real_simulated_whale['x'], real_simulated_whale['r'], real_simulated_whale['t0'], x_sensors)

corr_delays = np.load('./delays_px.npy')
corr_delays = corr_delays / 900.
corr_delays *= -1.

plt.figure()
plt.plot(x_sensors, real_simulated_delays, label=f'Real Simulated Delays (x={real_simulated_whale['x']} m, r={real_simulated_whale['r']} m, t0={real_simulated_whale['t0']})')
plt.plot(x_sensors, corr_delays, label=f'Cross-Correlation Delays')
plt.xlabel('x (sensor position)')
plt.ylabel('t (time)')
plt.legend()
plt.grid(True)
plt.show()

initial_guess_whale = {
	'x': 100,
	'r': 10,
	't0_w': 1.,
}

# optimize_results = minimize(costfun, x0=(initial_guess_whale['x'], initial_guess_whale['r'], initial_guess_whale['t0_w']), args=corr_delays)
optimize_results = minimize(costfun, x0=(initial_guess_whale['x'], initial_guess_whale['r'], initial_guess_whale['t0_w']), args=real_simulated_delays)

optimized_delays = generate_delays(optimize_results.x[0], optimize_results.x[1], optimize_results.x[2], x_sensors)

print(f'Synthetic parameters calculated (minimize):')
print(f'x_whale: {optimize_results.x[0]:.2f} m')
print(f'r_whale: {optimize_results.x[1]:.2f} m')
print(f't0_whale: {optimize_results.x[2]:.2f} s\n')

plt.figure()
plt.plot(x_sensors, optimized_delays, label=f'Synthetic Delays')
# plt.plot(x_sensors, corr_delays, label=f'Cross-Correlation Delays')
plt.plot(x_sensors, real_simulated_delays, label=f'Real Simulated Delays')
plt.xlabel('x (sensor position)')
plt.ylabel('t (time)')
plt.legend()
plt.grid(True)
plt.show()
