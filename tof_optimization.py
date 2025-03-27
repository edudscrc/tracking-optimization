import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_delays(x_w, r_w, x_sensors):
	c = 1440.
	return np.sqrt((x_w - x_sensors)**2 + r_w**2) / c


def costfun(params, delays):
	global x_sensors
	x_w, r_w = params
	return np.sum((delays - generate_delays(x_w, r_w, x_sensors))**2)


# Consider r_sensors = 0

x_sensors = np.linspace(0, 9000, int(9000*4), dtype=float)

whales = [
	{
		'x': 4500,
		'r': 10,
	},
	{
		'x': 4500,
		'r': 50,
	},
	{
		'x': 4500,
		'r': 100,
	},
	{
		'x': 4500,
		'r': 300,
	},
	{
		'x': 4500,
		'r': 500,
	},
	{
		'x': 4500,
		'r': 800,
	},
	{
		'x': 4500,
		'r': 5000,
	}
]

real_delays = [generate_delays(whale['x'], whale['r'], x_sensors) for whale in whales]

plt.figure()
for idx, delay in enumerate(real_delays):
	plt.plot(x_sensors, delay, label=f'Whale {idx} (x={whales[idx]['x']} m, r={whales[idx]['r']} m)')
plt.xlabel('x (sensor position)')
plt.ylabel('t (time)')
plt.legend()
plt.grid(True)
plt.show()

simulated_whales = [
	{
		'x': 5,
		'r': 1,
	},
	{
		'x': 5,
		'r': 1,
	},
	{
		'x': 5,
		'r': 1,
	},
	{
		'x': 5,
		'r': 1,
	},
	{
		'x': 5,
		'r': 1,
	},
	{
		'x': 5,
		'r': 1,
	},
]

optimize_results = [minimize(costfun, x0=(simulated_whale['x'], simulated_whale['r']), args=real_delays[idx]) for idx, simulated_whale in enumerate(simulated_whales)]

simulated_delays = [generate_delays(optimize_result.x[0], optimize_result.x[1], x_sensors) for optimize_result in optimize_results]

for idx, res in enumerate(optimize_results):
	print(f'Synthetic parameters for whale {idx}:')
	print(f'x_whale: {res.x[0]:.2f} m')
	print(f'r_whale: {res.x[1]:.2f} m\n')

plt.figure()
for idx, delay in enumerate(simulated_delays):
	
	plt.plot(x_sensors, delay, label=f'Whale {idx} (synthetic)')
plt.xlabel('x (sensor position)')
plt.ylabel('t (time)')
plt.legend()
plt.grid(True)
plt.show()
