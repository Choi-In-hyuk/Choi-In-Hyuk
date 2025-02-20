import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import control

# 시스템 파라미터
M = 0.5  # 카트 질량 (kg)
m = 0.5  # 막대 끝 질량 (kg)
L = 0.3  # 막대 길이 (m)
g = 9.81  # 중력 가속도 (m/s^2)

# 상태 공간 행렬
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -9.8, 0],
    [0, 0, 0, 1],
    [0, 0, 19.6, 0]
])

B = np.array([[0], [1], [0], [-1]])
C = np.array([[1, 0, 0, 0]])

# LQR 게인 계산
Q = 10 * np.eye(4)
R = np.array([[1]])
K_lqr, _, _ = control.lqr(A, B, Q, R)

# 시스템 동역학 함수
def plant1(X, U):
    DX = np.zeros((4, 1))
    DX[0, 0] = X[1, 0]
    DX[1, 0] = (np.sin(X[2, 0]) * X[3, 0]**2 - 9.8 * np.sin(X[2, 0]) * np.cos(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) + 1 / (2 - (np.cos(X[2, 0]))**2) * U
    DX[2, 0] = X[3, 0]
    DX[3, 0] = (-np.sin(X[2, 0]) * np.cos(X[2, 0]) * X[3, 0]**2 + 2 * 9.8 * np.sin(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) + (-np.cos(X[2, 0])) / (2 - (np.cos(X[2, 0]))**2) * U
    return DX

# Runge-Kutta 6차 적분
def rk6(X, U, T):
    k1 = plant1(X, U) * T
    k2 = plant1(X + k1 * 0.5, U) * T
    k3 = plant1(X + k2 * 0.5, U) * T
    k4 = plant1(X + k3, U) * T
    return X + ((k1 + k4) / 6.0 + (k2 + k3) / 3.0)

# 초기 상태 설정 (위 방향을 향하는 역진자)
X = np.zeros((4, 1))
X[2, 0] = 0.5  # 원래는 위 방향이 기준이었음

# 시뮬레이션 설정
Tf = 10
Ti = 0.01
t = np.arange(0, Tf, Ti)

# 상태 로그 저장
X_log = np.zeros((4, len(t)))
X_log[:, 0] = X.flatten()

# 시뮬레이션 루프
for i in range(len(t) - 1):
    U = -K_lqr @ X  # LQR 제어 입력 계산
    X = rk6(X, U, Ti)  # Runge-Kutta 적분 적용
    X_log[:, i + 1] = X.flatten()

# 애니메이션 설정
fig, ax = plt.subplots(figsize=(8, 5))

# 카트의 최대 이동 범위 계산 (자동으로 x축 크기 조정)
x_min = np.min(X_log[0, :]) - 0.5  # 최소 x값에서 여유 공간 추가
x_max = np.max(X_log[0, :]) + 0.5  # 최대 x값에서 여유 공간 추가

ax.set_xlim(x_min, x_max)  # x축 범위를 자동 설정
ax.set_ylim(-0.5, 0.5)  
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("LQR-Controlled Inverted Pendulum with Mass at Tip")

# 카트 (사각형)
cart_width = 0.3
cart_height = 0.1
cart = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, color="black")

# 막대 (선)
line, = ax.plot([], [], lw=3, color="blue")

# 끝에 질량 추가 (빨간 원)
mass, = ax.plot([], [], 'ro', markersize=6)


text_template = "t = {:.2f}s\nx = {:.3f} m\nx_dot = {:.3f} m/s\nθ = {:.3f} rad\nθ_dot = {:.3f} rad/s\nF = {:.3f} N"
text_display = ax.text(x_min + 0.1, 0.3, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

                       
                       
# 애니메이션 초기화
def init():
    ax.add_patch(cart)
    line.set_data([], [])
    mass.set_data([], [])  # 질량 초기 위치
    text_display.set_text("")
    return cart, line, mass, text_display

# 애니메이션 업데이트 함수 (`theta` 변환 적용)
def update(frame):
    x = X_log[0, frame]
    x_dot = X_log[1, frame]
    theta = np.pi - X_log[2, frame]  
    theta_dot = X_log[3, frame]
    F = F_log[frame]

    cart.set_xy((x - cart_width / 2, -cart_height / 2))

    pendulum_x = x + L * np.sin(theta)
    pendulum_y = -L * np.cos(theta)
    line.set_data([x, pendulum_x], [0, pendulum_y])
    mass.set_data([pendulum_x], [pendulum_y])  


    text_display.set_text(text_template.format(frame * Ti, x, x_dot, theta, theta_dot, F))
    
    return cart, line, mass, text_display

F_log = np.zeros(len(t))

# 시뮬레이션 루프 (입력 F 기록)
X = np.zeros((4, 1))
X[2, 0] = 0.5  # 초기 각도 (rad)

for i in range(len(t) - 1):
    U = -K_lqr @ X  # LQR 제어 입력 계산
    X = rk6(X, U, Ti)  # Runge-Kutta 적분 적용
    X_log[:, i + 1] = X.flatten()
    F_log[i] = U  # 입력 힘 저장

frames_to_save = int((10 / Ti)/3)
# 애니메이션 실행
ani = animation.FuncAnimation(fig, update, frames_to_save, init_func=init, interval=Ti * 2000, blit=True)
ani.save("cart_pole_simulation.mp4", writer="ffmpeg", fps=30)
plt.show()

plt.figure(figsize=(12, 8))

# ① 카트 위치 x(t)
plt.subplot(3, 2, 1)
plt.plot(t, X_log[0, :], label="Cart Position (x)", color='blue')
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Cart Position Over Time")
plt.legend()
plt.grid()

# ② 카트 속도 dx/dt
plt.subplot(3, 2, 2)
plt.plot(t, X_log[1, :], label="Cart Velocity (dx/dt)", color='red')
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Cart Velocity Over Time")
plt.legend()
plt.grid()

# ③ 역진자 각도 θ(t)
plt.subplot(3, 2, 3)
plt.plot(t, X_log[2, :], label="Pendulum Angle (θ)", color='green')
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title("Pendulum Angle Over Time")
plt.legend()
plt.grid()

# ④ 역진자 각속도 dθ/dt
plt.subplot(3, 2, 4)
plt.plot(t, X_log[3, :], label="Pendulum Angular Velocity (dθ/dt)", color='purple')
plt.xlabel("Time [s]")
plt.ylabel("Angular Velocity [rad/s]")
plt.title("Pendulum Angular Velocity Over Time")
plt.legend()
plt.grid()

# ⑤ 제어 입력 F(t) - LQR이 카트를 어떻게 움직이는지
plt.subplot(3, 1, 3)
plt.plot(t, F_log, label="Control Input Force (F)", color='orange')
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("Control Input Force Over Time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



