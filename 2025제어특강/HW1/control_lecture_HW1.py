
# import sympy as sp

# X1, X2, X3, X4, U = sp.symbols('X1 X2 X3 X4 U')
# fx = sp.Matrix([
#     X2,
#     (sp.sin(X3)*X4**2 - 9.8*sp.sin(X3)*sp.cos(X3)) / (2 - sp.cos(X3)**2) + 1/(2 - sp.cos(X3)**2) * U,
#     X4,
#     (-sp.sin(X3)*sp.cos(X3)*X4**2 + 2*9.8*sp.sin(X3)) / (2 - sp.cos(X3)**2) + (-sp.cos(X3)) / (2 - sp.cos(X3)**2) * U
# ])

# dfx = fx.jacobian([X1, X2, X3, X4, U])
# subs_values = {X1: 0, X2: 0, X3: 0, X4: 0, U: 0}
# Result = dfx.subs(subs_values)

# sp.pprint(Result)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import control


A = np.array([
    [0, 1, 0, 0],
    [0, 0, -9.8, 0],
    [0, 0, 0, 1],
    [0, 0, 19.6, 0]
])
B = np.array([[0], [1], [0], [-1]])
C = np.array([[1, 0, 0, 0]])
poles = np.array([-1 + 1j, -1 - 1j, -2, -5])
K = control.place(A, B, poles)
Q = 10 * np.eye(4)
R = np.array([[1]])
K_lqr, S, E = control.lqr(A, B, Q, R)


def plant1(X, U):
    DX = np.zeros((4, 1))
    DX[0, 0] = X[1, 0]
    DX[1, 0] = (np.sin(X[2, 0]) * X[3, 0]**2 - 9.8 * np.sin(X[2, 0]) * np.cos(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) + 1 / (2 - (np.cos(X[2, 0]))**2) * U
    DX[2, 0] = X[3, 0]
    DX[3, 0] = (-np.sin(X[2, 0]) * np.cos(X[2, 0]) * X[3, 0]**2 + 2 * 9.8 * np.sin(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) + (-np.cos(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) * U
    return DX


def rk6(X, U, T):
    k1 = plant1(X, U) * T
    k2 = plant1(X + k1 * 0.5, U) * T
    k3 = plant1(X + k2 * 0.5, U) * T
    k4 = plant1(X + k3, U) * T
    return X + ((k1 + k4) / 6.0 + (k2 + k3) / 3.0)


X = np.zeros((4, 1))
X[2, 0] = 0.5 
Tf = 10
Ti = 0.01
t = np.arange(0, Tf, Ti)
X_log = np.zeros((4, len(t)))
X_log[:, 0] = X.flatten()


for i in range(len(t) - 1):
    U = -K_lqr @ X 
    X = rk6(X, U, Ti) 
    X_log[:, i + 1] = X.flatten()


plt.figure(figsize=(10, 6))
plt.plot(t, X_log.T)
plt.xlabel('Time [s]')
plt.ylabel('State Variables')
plt.legend(['X1', 'X2', 'X3', 'X4'])
plt.grid()
plt.show()
