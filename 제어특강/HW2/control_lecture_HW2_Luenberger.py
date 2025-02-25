import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import control


# System Matrices
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -9.8, 0],
    [0, 0, 0, 1],
    [0, 0, 19.6, 0]
])
B = np.array([[0], [1], [0], [-1]])
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # Measure only x and θ

# LQR Controller
Q = 10 * np.eye(4)
R = np.array([[1]])
K_lqr, S, E = control.lqr(A, B, Q, R)

# Observer Gain (Luenberger Observer)
poles = np.array([-2, -3, -4, -5])  # Observer poles (should be stable)
L = control.place(A.T, C.T, poles).T  # Compute observer gain


# System Simulation
def plant1(X, U):
    DX = np.zeros((4, 1))
    DX[0, 0] = X[1, 0]
    DX[1, 0] = (np.sin(X[2, 0]) * X[3, 0]**2 - 9.8 * np.sin(X[2, 0]) * np.cos(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) + 1 / (2 - (np.cos(X[2, 0]))**2) * U
    DX[2, 0] = X[3, 0]
    DX[3, 0] = (-np.sin(X[2, 0]) * np.cos(X[2, 0]) * X[3, 0]**2 + 2 * 9.8 * np.sin(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) + (-np.cos(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) * U
    return DX


# Runge-Kutta Integration
def rk6(X, U, T):
    k1 = plant1(X, U) * T
    k2 = plant1(X + k1 * 0.5, U) * T
    k3 = plant1(X + k2 * 0.5, U) * T
    k4 = plant1(X + k3, U) * T
    return X + ((k1 + k4) / 6.0 + (k2 + k3) / 3.0)

# Animation Setup
fig, ax = plt.subplots(figsize=(8, 5))
cart_width = 0.3
cart_height = 0.1

ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("LQR-Controlled Inverted Pendulum with Observer")

cart = plt.Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, color="black")
cart_hat = plt.Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, color="gray", alpha=0.5)  # Estimated cart

line, = ax.plot([], [], lw=3, color="blue")  # True pendulum
line_hat, = ax.plot([], [], lw=3, linestyle='dashed', color="red")  # Estimated pendulum

mass, = ax.plot([], [], 'ro', markersize=6)
mass_hat, = ax.plot([], [], 'go', markersize=6, alpha=0.5)  # Estimated mass

text_template = "t = {:.2f}s\nx = {:.3f} m\nθ = {:.3f} rad\nF = {:.3f} N"
text_display = ax.text(-1.8, 0.3, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.6))


# Initial Conditions
X = np.zeros((4, 1))
X[2, 0] = 0.5  # Initial pendulum angle
X_hat = np.zeros((4, 1))  # Initial estimated state

Tf = 10
Ti = 0.01
t = np.arange(0, Tf, Ti)
X_log = np.zeros((4, len(t)))
X_hat_log = np.zeros((4, len(t)))
F_log = np.zeros(len(t))  # Control input force logging

X_log[:, 0] = X.flatten()
X_hat_log[:, 0] = X_hat.flatten()


U = np.zeros((1, 1))  # Initialize control input

for i in range(len(t) - 1):
    Y = C @ X  # Measured output (only x and θ)
    X_hat = X_hat + (A @ X_hat + B @ U + L @ (Y - C @ X_hat)) * Ti  # Observer update

    U = -K_lqr @ X_hat  # Compute control input using estimated state
    X = rk6(X, U, Ti)  # Update system state using Runge-Kutta integration

    X_log[:, i + 1] = X.flatten()  # Store actual state
    X_hat_log[:, i + 1] = X_hat.flatten()  # Store estimated state
    F_log[i] = U.item()  # Store control input force


# Animation Initialization
def init():
    ax.add_patch(cart)
    ax.add_patch(cart_hat)
    line.set_data([], [])
    line_hat.set_data([], [])
    mass.set_data([], [])
    mass_hat.set_data([], [])
    text_display.set_text("")
    return cart, cart_hat, line, line_hat, mass, mass_hat, text_display

# Animation Update Function
def update(frame):
    x = X_log[0, frame]
    theta = np.pi - X_log[2, frame]
    x_hat = X_hat_log[0, frame]
    theta_hat = np.pi - X_hat_log[2, frame]
    F = F_log[frame]

    # True system
    cart.set_xy((x - cart_width / 2, -cart_height / 2))
    pendulum_x = x + np.sin(theta)
    pendulum_y = -np.cos(theta)
    line.set_data([x, pendulum_x], [0, pendulum_y])
    mass.set_data([pendulum_x], [pendulum_y])

    # Estimated system
    cart_hat.set_xy((x_hat - cart_width / 2, -cart_height / 2))
    pendulum_x_hat = x_hat + np.sin(theta_hat)
    pendulum_y_hat = -np.cos(theta_hat)
    line_hat.set_data([x_hat, pendulum_x_hat], [0, pendulum_y_hat])
    mass_hat.set_data([pendulum_x_hat], [pendulum_y_hat])

    text_display.set_text(text_template.format(frame * Ti, x, theta, F))

    return cart, cart_hat, line, line_hat, mass, mass_hat, text_display

# Run Animation
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, interval=Ti * 1000, blit=True)


# Plot Results
plt.figure(figsize=(12, 10))

# (1) Cart Position x(t)
plt.subplot(5, 1, 1)
plt.plot(t, X_log[0, :], label="Cart Position (x) - True", color='blue')
plt.plot(t, X_hat_log[0, :], label="Cart Position (x) - Estimated", linestyle='dashed', color='blue')
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Cart Position Over Time")
plt.legend()
plt.grid()

# (2) Cart Velocity dx/dt
plt.subplot(5, 1, 2)
plt.plot(t, X_log[1, :], label="Cart Velocity (dx/dt) - True", color='red')
plt.plot(t, X_hat_log[1, :], label="Cart Velocity (dx/dt) - Estimated", linestyle='dashed', color='red')
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Cart Velocity Over Time")
plt.legend()
plt.grid()

# (3) Pendulum Angle θ(t)
plt.subplot(5, 1, 3)
plt.plot(t, X_log[2, :], label="Pendulum Angle (θ) - True", color='green')
plt.plot(t, X_hat_log[2, :], label="Pendulum Angle (θ) - Estimated", linestyle='dashed', color='green')
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title("Pendulum Angle Over Time")
plt.legend()
plt.grid()

# (4) Pendulum Angular Velocity dθ/dt
plt.subplot(5, 1, 4)
plt.plot(t, X_log[3, :], label="Pendulum Angular Velocity (dθ/dt) - True", color='purple')
plt.plot(t, X_hat_log[3, :], label="Pendulum Angular Velocity (dθ/dt) - Estimated", linestyle='dashed', color='purple')
plt.xlabel("Time [s]")
plt.ylabel("Angular Velocity [rad/s]")
plt.title("Pendulum Angular Velocity Over Time")
plt.legend()
plt.grid()

# (5) Control Input Force F(t)
plt.subplot(5, 1, 5)
plt.plot(t, F_log, label="Control Input Force (F)", color='orange')
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Control Input Force Over Time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
