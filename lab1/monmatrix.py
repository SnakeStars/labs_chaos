import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import eig

# --- Система ---
def system(t, X, k):
    x, y = X
    dx = y
    dy = -x - k*(x**2 + y**2 - 1)*y
    return np.array([dx, dy])

# --- Якобиан системы ---
def jacobian(X, k):
    x, y = X
    J = np.array([
        [0, 1],
        [-1 - 2*k*x*y, -k*(x**2 + 3*y**2 - 1)]
    ])
    return J

# --- Параметры ---
k = 1.5
x0_traj, y0_traj = 1.0, 0.0  # начальная точка внутри предполагаемого цикла
t_span = (0, 200)            # интегрируем достаточно долго
dt = 0.01
t_eval = np.arange(t_span[0], t_span[1], dt)

# --- Шаг 1: интегрируем систему, чтобы найти пересечения x=0, y>0 ---
sol = solve_ivp(lambda t, X: system(t, X, k), t_span, [x0_traj, y0_traj], t_eval=t_eval)

x_vals = sol.y[0]
y_vals = sol.y[1]
t_vals = sol.t

# Ищем пересечения с x=0 (с положительным y)
crossings = []
for i in range(1, len(x_vals)):
    if x_vals[i-1] < 0 and x_vals[i] >= 0 and y_vals[i] > 0:
        # линейная интерполяция времени пересечения
        t_cross = t_vals[i-1] + (0 - x_vals[i-1])*(t_vals[i]-t_vals[i-1])/(x_vals[i]-x_vals[i-1])
        crossings.append(t_cross)

if len(crossings) < 2:
    raise RuntimeError("Не удалось найти период цикла. Попробуйте увеличить t_span или изменить начальную точку.")

# Примерная оценка периода
T_cycle = crossings[1] - crossings[0]
print("Приблизительный период предельного цикла:", T_cycle)

# --- Шаг 2: интегрируем вариационное уравнение на найденный период ---
Phi0 = np.eye(2).flatten()  # начальная вариационная матрица

def combined_system(t, Z):
    X = Z[0:2]
    Phi = Z[2:].reshape(2,2)
    dX = system(t, X, k)
    dPhi = jacobian(X, k) @ Phi
    return np.concatenate([dX, dPhi.flatten()])

Z0 = np.concatenate([[x0_traj, y0_traj], Phi0])

sol_var = solve_ivp(combined_system, [0, T_cycle], Z0, t_eval=np.linspace(0, T_cycle, 1000))

# --- Шаг 3: матрица монодромии и мультипликаторы Флоке ---
Phi_T = sol_var.y[2:6,-1].reshape(2,2)
multipliers = eig(Phi_T)[0]

print("Мультипликаторы Флоке:", multipliers)