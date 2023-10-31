import matplotlib.pyplot as plt
import numpy as np

# Problemas lineares - Regressão Linear

# Equações Normais - Descida de Gradiente


def compute_cost(a, b, c, d, points):
    total_cost = 0
    N = float(len(points))

    # Calcular soma dos erros ao quadrado
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        y_p = a * x**3 + b * x**2 + c * x + d
        total_cost += (y - y_p) ** 2

    return total_cost / 2 * N


def step_gradient(a_atual, b_atual, c_atual, d_atual, points, learning_rate):
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0
    d_gradient = 0
    N = float(len(points))

    # Calcular o gradiente

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        y_p = a_atual * x**3 + b_atual * x**2 + c_atual * x + d_atual
        a_gradient += -(1 / N) * x**3 * (y - y_p)
        b_gradient += -(1 / N) * x**2 * (y - y_p)
        c_gradient += -(1 / N) * x * (y - y_p)
        d_gradient += -(1 / N) * (y - y_p)

    # Atualizar a, b, c, d
    a_update = a_atual - learning_rate * a_gradient
    b_update = b_atual - learning_rate * b_gradient
    c_update = c_atual - learning_rate * c_gradient
    d_update = d_atual - learning_rate * d_gradient

    return a_update, b_update, c_update, d_update


def gradient_descent_runner(
    points,
    starting_a,
    starting_b,
    starting_c,
    starting_d,
    learning_rate,
    num_interations,
):
    cost_graph = []
    # Para cada interação, otimizar a, b, c, d, e computar custo

    for i in range(num_interations):
        cost_graph.append(
            compute_cost(
                starting_a, starting_b, starting_c, starting_d, points
            )
        )

        a, b, c, d = step_gradient(
            starting_a,
            starting_b,
            starting_c,
            starting_d,
            np.array(points),
            learning_rate,
        )

    return [a, b, c, d, cost_graph]


# Hiperparametros

learning_rate = 0.00001
initial_a = 0
initial_b = 0
initial_c = 0
initial_d = 0
num_interations = 50


x = np.linspace(-10, 10, 550)
y = 5 * x**3 + 2 * x**2 + 3 * x


noise = 1e2 * np.random.normal(size=len(x))
y2 = y + noise

points = np.array([x, y2]).T
# print(points.shape)

a, b, c, d, cost_graph = gradient_descent_runner(
    points,
    initial_a,
    initial_b,
    initial_c,
    initial_d,
    learning_rate,
    num_interations,
)

# print(a, b, c, d)

# plt.scatter(x, y2, color="hotpink")
# plt.plot(x, y, "k")
# plt.savefig("imgs/functions.png")
# plt.close()

plt.plot(cost_graph)
plt.savefig("imgs/cost_graph.png")
plt.close()
