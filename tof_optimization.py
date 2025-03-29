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

# x_sensors = np.linspace(0, 9000, int(9000*4), dtype=float)
x_sensors = np.linspace(0, 7000, 27376, dtype=float)

whales = [
	{
		'x': 3950,
		'r': 500,
		't0': -0.3,
	},
]

real_simulated_delays = [generate_delays(whale['x'], whale['r'], whale['t0'], x_sensors) for whale in whales]

np.save('./real_simulated_delays_seconds.npy', real_simulated_delays[0])

corr_delays = np.load('./delays_px.npy')
corr_delays = corr_delays / 900.
corr_delays *= -1.

real_delays = [corr_delays]

plt.figure()
for idx, delay in enumerate(real_delays):
	plt.plot(x_sensors, real_simulated_delays[idx], label=f'Whale (x={whales[idx]['x']} m, r={whales[idx]['r']} m, t0={whales[idx]['t0']})')
	plt.plot(x_sensors, delay, label=f'Cross-Correlation Delays')
plt.xlabel('x (sensor position)')
plt.ylabel('t (time)')
plt.legend()
plt.grid(True)
plt.show()

simulated_whales = [
	{
		'x': 4000,
		'r': 1000,
		't0_w': 3,
	},
]

optimize_results = [minimize(costfun, x0=(simulated_whale['x'], simulated_whale['r'], simulated_whale['t0_w']), args=real_delays[idx]) for idx, simulated_whale in enumerate(simulated_whales)]

simulated_delays = [generate_delays(optimize_result.x[0], optimize_result.x[1], optimize_result.x[2], x_sensors) for optimize_result in optimize_results]

for idx, res in enumerate(optimize_results):
	print(f'Synthetic parameters for whale {idx}:')
	print(f'x_whale: {res.x[0]:.2f} m')
	print(f'r_whale: {res.x[1]:.2f} m')
	print(f't0_whale: {res.x[2]:.2f} s\n')

# plt.figure()
# for idx, delay in enumerate(simulated_delays):
# 	plt.plot(x_sensors, delay, label=f'Whale {idx} (synthetic)')
# 	plt.plot(x_sensors, real_delays[idx], label=f'Cross-Correlation Delays')
# plt.xlabel('x (sensor position)')
# plt.ylabel('t (time)')
# plt.legend()
# plt.grid(True)
# plt.show()
