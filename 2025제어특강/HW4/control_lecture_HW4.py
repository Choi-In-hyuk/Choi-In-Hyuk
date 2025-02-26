% Model Predictive Control Example

% Define the system matrices.
A=[2.4936 -1.2130 0.4176;
   2 0 0;
   0 1 0];
B=[0.25,0, 0]';
C=[0.0684 0.1288 0.0313];
D=0;

Np=10;           % Prediction and control horizon
lambda=1;       % Trade-off parameter
x0=[1 1 1]';    % Initial state

% Construct the prediction matrices.
At=[];
for i=1:Np
    At=[At;A^i];
end;

temp = []; % 임시 행렬 (입력 행렬을 오른쪽으로 확장)
Bt = [];   % 최종 Bt 행렬

for i = 1:Np
    new_col = A^(i-1) * B;
    if ~isempty(Bt)
        Bt = [Bt, zeros(size(Bt,1), size(new_col,2))]; % Bt 크기 확장
    end

    Bt = [Bt; new_col, temp];
    temp = [new_col, temp];
end


temp = []; % 임시 행렬 (이전 C 행렬들을 저장)
Ct = [];   % 최종 Ct 행렬

for i = 1:Np
    % 현재 반복에서 새롭게 추가할 C 행렬 계산
    new_col = C; % Ct에서는 C가 그대로 사용됨

    % 기존 Ct 행렬이 비어있지 않다면, zero padding 추가 (열 크기 맞추기)
    if ~isempty(Ct)
        Ct = [Ct, zeros(size(Ct,1), size(new_col,2))];
    end

    % 새로운 행을 추가
    Ct = [Ct; new_col, temp];

    % temp 업데이트 (입력 블록을 오른쪽으로 확장)
    temp = [new_col, temp];
end



% Define the lower and upper bounds, and the initial value.
lb=-0.25*ones(Np,1);
ub=0.25*ones(Np,1);
u0=zeros(Np,1);

% Simulation of the closed-loop system
u=[];
x=x0;           % Initial state
y=C*x0;         % Initial output
xk=x0;          % Start of the horizon
H=2*((Ct*Bt)'*(Ct*Bt) + lambda*eye(Np,Np)); % Note that H is a constant
                                             % matrix that does not depend
                                             % on k.

options=optimoptions('quadprog',...
                     'Algorithm','interior-point-convex');

% Select the active-set algorithm
for k=1:50
    ut=quadprog(H,2*xk'*At'*Ct'*Ct*Bt,[],[],[],[],lb,ub,u0,options);
    u(k)=ut(1); % Implement only the 1st control sample.
    
    % Compute the state.
    x(:,k+1)=A*x(:,k)+B*u(k);
    
    % Compute the output.
    y(:,k+1)=C*x(:,k+1);
    
    % Shift the horizon.
    xk=x(:,k+1);
    u0=[ut(2:Np);ut(Np)];
    
    % Use the shifted solution as the
    % initial solution for the next iteration.
end;

% ===========================
% MPC Optimization Results
% ===========================

% Display the final optimization results
disp('========== Optimization Results ==========');
disp(['Final state x(T): ', num2str(x(:, end)')]); % Last state value
disp(['Final output y(T): ', num2str(y(:, end)')]); % Last output value
disp(['Final control input u(T-1): ', num2str(u(end))]); % Last applied control input

% ===========================
% Plot control input u(k)
% ===========================

figure;
plot(1:length(u), u, 'bo-', 'LineWidth', 2);
xlabel('Time step'); % X-axis label
ylabel('Control Input u'); % Y-axis label
title('MPC Control Input'); % Graph title
grid on; % Enable grid for better visualization

% ===========================
% Plot state evolution x(k)
% ===========================

figure;
plot(1:size(x,2), x(1,:), 'r-', 'LineWidth', 2); hold on;
plot(1:size(x,2), x(2,:), 'g-', 'LineWidth', 2);
plot(1:size(x,2), x(3,:), 'b-', 'LineWidth', 2);
xlabel('Time step'); % X-axis label
ylabel('State x'); % Y-axis label
title('MPC State Evolution'); % Graph title
legend('x_1', 'x_2', 'x_3'); % Legend for each state
grid on; % Enable grid

% ===========================
% Plot output y(k)
% ===========================

figure;
plot(1:length(y), y, 'ko-', 'LineWidth', 2);
xlabel('Time step'); % X-axis label
ylabel('Output y'); % Y-axis label
title('MPC Output Response'); % Graph title
grid on; % Enable grid
