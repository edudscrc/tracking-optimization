import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Velocidade do som na água (m/s)
c = 1500.0

# Posicionamento dos sensores ao longo do cabo
x_sensors = np.linspace(0, 9000, int(9000*4), dtype=float)

# Posição da baleia verdadeira
whale_true = {'x': 4500, 'r': 300}

def generate_delays(x_w, r_w, x_sensors, ref_sensor=0):
    """
    Calcula os atrasos dos sinais considerando um sensor de referência.
    """
    dists = np.sqrt((x_w - x_sensors)**2 + r_w**2)
    delays = dists / c
    return delays - delays[ref_sensor]  # TDOA relativo ao sensor de referência

def costfun(params, delays, ref_sensor=0):
    """
    Função de custo: minimiza a diferença entre os atrasos simulados e reais.
    """
    x_w, r_w = params
    simulated_delays = generate_delays(x_w, r_w, x_sensors, ref_sensor)
    return np.sum((delays - simulated_delays) ** 2)

# Gerar atrasos reais com base na posição verdadeira da baleia
real_delays = generate_delays(whale_true['x'], whale_true['r'], x_sensors)

# Grade de busca inicial (resolução de 50m em x e 20m em r)
x_search = np.arange(4000, 5000, 50)
r_search = np.arange(100, 500, 20)

# Busca bruta para inicialização
best_x, best_r = None, None
best_cost = np.inf
for x in x_search:
    for r in r_search:
        cost = costfun((x, r), real_delays)
        if cost < best_cost:
            best_cost = cost
            best_x, best_r = x, r

# Otimização refinada com a melhor estimativa inicial
opt_result = minimize(costfun, x0=(best_x, best_r), args=(real_delays))

# Parâmetros estimados
x_est, r_est = opt_result.x

# Gerar atrasos simulados para comparar
simulated_delays = generate_delays(x_est, r_est, x_sensors)

# Visualização dos atrasos reais e estimados
plt.figure()
plt.plot(x_sensors, real_delays, label='Atrasos Reais')
plt.plot(x_sensors, simulated_delays, linestyle='dashed', label='Atrasos Estimados')
plt.xlabel('x (posição do sensor)')
plt.ylabel('TDOA (s)')
plt.legend()
plt.grid(True)
plt.show()

# Exibir resultados
print(f'Posição estimada da baleia: x = {x_est:.2f} m, r = {r_est:.2f} m')
