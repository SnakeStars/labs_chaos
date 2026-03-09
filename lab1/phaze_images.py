import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec


# Система
def system(t, X, k):
    x, y = X
    dx = y
    dy = -x - k * (x ** 2 + y ** 2 - 1) * y
    return [dx, dy]


# Начальные условия для фазового портрета
x0_vals = np.linspace(-1.5, 1.5, 5)
y0_vals = np.linspace(-1.5, 1.5, 5)

# Начальная точка для временного ряда
x0_traj, y0_traj = 0.5, 0.0

k_values = [-1, 1]  # два k
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# GridSpec: 4 строки, 2 колонки
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(4, 2, width_ratios=[1, 1.2], height_ratios=[1, 1, 1, 1], hspace=0.4)

for i, k in enumerate(k_values):
    # ---- Фазовый портрет слева ----
    # Объединяем две строки для каждого k
    ax_phase = fig.add_subplot(gs[2 * i:2 * i + 2, 0])
    for x0 in x0_vals:
        for y0 in y0_vals:
            sol = solve_ivp(system, t_span, [x0, y0], args=(k,), t_eval=t_eval)
            ax_phase.plot(sol.y[0], sol.y[1], color='purple', lw=1)
            ax_phase.plot(x0, y0, 'ro', markersize=3)
    ax_phase.set_title(f'Фазовый портрет, k={k}')
    ax_phase.set_aspect('equal')
    ax_phase.set_xlim((-2,2))
    ax_phase.set_ylim((-2, 2))
    ax_phase.grid(True)

    # ---- Временные ряды справа ----
    sol_traj = solve_ivp(system, t_span, [x0_traj, y0_traj], args=(k,), t_eval=t_eval)

    # x(t)
    ax_xt = fig.add_subplot(gs[2 * i, 1])
    ax_xt.plot(sol_traj.t, sol_traj.y[0], color='purple')
    ax_xt.set_title(f'x(t), k={k}')
    ax_xt.set_ylabel('x(t)')
    ax_xt.grid(True)

    # y(t)
    ax_yt = fig.add_subplot(gs[2 * i + 1, 1])
    ax_yt.plot(sol_traj.t, sol_traj.y[1], color='purple')
    ax_yt.set_title(f'y(t), k={k}')
    ax_yt.set_ylabel('y(t)')
    ax_yt.grid(True)

# fig.subplots_adjust(
#     top=0.95,    # верхний край фигуры
#     bottom=0.05, # нижний край
#     left=0.07,   # левый край
#     right=0.93,  # правый край
#     hspace=0.5,  # вертикальные отступы между строками
#     wspace=0.4   # горизонтальные отступы между колонками
# )
plt.show()