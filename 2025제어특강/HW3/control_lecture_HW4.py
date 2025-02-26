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


Ct=[];
for i=1:Np
    Ct=[Ct
        zeros(size(Ct,1),size(C,2))
        zeros(size(C,1),size(C,2)) C];
end;

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
