%% Discovering model from data
% Instead of using DMD, this file uses approximates dynamical system 
% using a library of nonlinear functions and then applying least square
% to find x_dot=Ac, where c consists of the coefficients in front of the 
% nonlinear functions. 

clc; clear all; close all; 

% Construct data matrix X
PH = [20,20,52,83,64,68,83,12,36,150,110,60,7,...   % Hare data
    10,70,100,92, 70,10,11,137,137,18,22,52,83,18,10,9,65];
PL = [32,50,12,10,13,36,15,12,6,6,65,70,40,9,...    % Lynx data
    20,34,45,40,15, 15,60,80,26,18,37,50,35,12,12,25];
P = [PH; PL];

% Spline interpolation for smoother data and more data points
% The performance is better than without it 
year = 0:2:58; % Year since 1854
dt=0.1; t=dt:dt:58; 
X = spline(year,P,t); 

%% Part 3: Fitting Lotka-Volterra Coefficients 
% Tried cross-validation: divided data and averaged, but worsened 
% the performance b/c the data size is so small 

% Try finite difference method to get the derivative
n=length(t); 
for j=2:n-1
  x1dot(j-1)=(X(1,j+1)-X(1,j-1))/(2*dt);
  x2dot(j-1)=(X(2,j+1)-X(2,j-1))/(2*dt);
end
x1dot=x1dot.'; x2dot=x2dot.'; 
% % Total variational derivative didn't improve substantially 
% x1dot = TVRegDiff(X(1,:), 50, 0.85); x1dot=x1dot(2:end-2);
% x2dot = TVRegDiff(X(2,:), 50, 0.85); x2dot=x2dot(2:end-2);

% Construct and solve for Ax=b
xs=X(1,2:n-1)'; ys=X(2,2:n-1)';
Ax=[xs -xs.*ys]; Ay=[xs.*ys -ys];
xi=Ax\x1dot;
yi=Ay\x2dot;
b=xi(1); p=xi(2); r=yi(1); d=yi(2); 

% Compare the data with the Lotka-Volterra model
[t,y_lv]=ode45('rhs_lv',t,X(:,1),[],b,p,r,d);
figure;
subplot(2,1,1), plot(t,X(1,:),t,y_lv(:,1),'Linewidth',[2]); 
legend('Data', 'Fitted Model'); 
xlabel(' Time (year since 1854)'); ylabel('Hare Population'); 
subplot(2,1,2), plot(t,X(2,:),t,y_lv(:,2),'Linewidth',[2]); 
legend('Data', 'Fitted Model');  
xlabel(' Time (year since 1854)'); ylabel('Lynx Population'); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');

%% Part 4: Data Fit Using a Richer Library

% Construct a library of polynomials and solve Ax=b
% Some polynomial terms were removed b/c they cause instability
A1=[xs ys xs.^2 xs.*ys ys.^2 xs.^3 (ys.^2).*xs (xs.^2).*ys ys.^3 ...
    xs.^4 (xs.^3).*ys (xs.^2).*(ys.^2) (ys.^3).*xs ys.^4];
indx=2; indy=1;  % terms to remove
A1x=A1; A1x(:,indx)=[]; 
A1y=A1; A1y(:,indy)=[]; 
xi1=A1x\x1dot;
yi1=A1y\x2dot;

% Visualize the Sparsity, already sparse
figure; subplot(2,1,1), bar(xi1)
subplot(2,1,2), bar(yi1)

% % Lasso for sparsity, removes the small high order coeff
% % hurts performance a lot! 
% % Maybe because high order functions with small coefficents do matter 
% xi1=lasso(A1x,x1dot,'Lambda',0.02,'Alpha', 0.5);
% yi1=lasso(A1y,x2dot,'Lambda',0.02,'Alpha', 0.5);
% figure; subplot(2,1,1), bar(xi1)
% subplot(2,1,2), bar(yi1)

% Compare the data with the polynomial nolinear model
[t,y_poly]=ode45('rhs_poly',t,X(:,1),[],xi1,yi1,indx,indy);
figure;
subplot(2,1,1), plot(t,X(1,:),t,y_poly(:,1),'Linewidth',[2]); 
legend('Data', 'Fitted Model'); 
xlabel(' Time (year since 1854)'); ylabel('Hare Population'); 
subplot(2,1,2), plot(t,X(2,:),t,y_poly(:,2),'Linewidth',[2]); 
legend('Data', 'Fitted Model');  
xlabel(' Time (year since 1854)'); ylabel('Lynx Population'); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');

% Construct a library of sinusoids 
% Note, sinusoid is not necessary for getting limit cycles 
% Use sin(t) instead of sin(X), which is hard to control
A2=[xs ys]; 
for om=0.2:0.5:5
    A2=[A2 sin(om*t(2:n-1)) cos(om*t(2:n-1))];
end 
indx=2; indy=1;  % terms to remove
A2x=A2; A2x(:,indx)=[]; 
A2y=A2; A2y(:,indy)=[]; 

xi2=A2x\x1dot;
yi2=A2y\x2dot;

% Visualize the Sparsity
figure; subplot(2,1,1), bar(xi2)
subplot(2,1,2), bar(yi2);

% Compare the data with the sinusoidal nolinear model
[t,y_sin]=ode45('rhs_sin',t,X(:,1),[],xi2,yi2,indx,indy);
figure;
subplot(2,1,1), plot(t,X(1,:),t,y_sin(:,1),'Linewidth',[2]); 
legend('Data', 'Fitted Model'); 
xlabel(' Time (year since 1854)'); ylabel('Hare Population'); 
subplot(2,1,2), plot(t,X(2,:),t,y_sin(:,2),'Linewidth',[2]); 
legend('Data', 'Fitted Model');
xlabel(' Time (year since 1854)'); ylabel('Lynx Population'); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');

% Some notes
% - Remove the "problem" terms really helped

%% KL Divergence
% Important caveat: no time matching! 
% But gives a general sense if the range and distribution of data 
% is kept in the model 

% First, compute the KL for the Hare Population:
edges=0:15:150; 
KL_H=zeros(3,1); 
KL_H(1) = get_KL(edges, X(1,:), y_lv(:,1)); 
KL_H(2) = get_KL(edges, X(1,:), y_poly(:,1)); 
KL_H(3) = get_KL(edges, X(1,:), y_sin(:,1)); 

% Repeat for the Lynx Population: 
edges=0:8:80;
KL_L=zeros(3,1); 
KL_L(1) = get_KL(edges, X(2,:), y_lv(:,2)); 
KL_L(2) = get_KL(edges, X(2,:), y_poly(:,2)); 
KL_L(3) = get_KL(edges, X(2,:), y_sin(:,2)); 

%% AIC BIC

% First, compute AIC BIC for the Hare population 

% Get the underlying normal distribution
[mu1H,sig1H] = normfit(y_lv(:,1));
[mu2H,sig2H] = normfit(y_poly(:,1));
[mu3H,sig3H] = normfit(y_sin(:,1));

logL = zeros(3,1); % Preallocate loglikelihood vector
logL(1)=sum(log(normpdf(X(1,:),mu1H,sig1H)));
logL(2)=sum(log(normpdf(X(1,:),mu2H,sig2H)));
logL(3)=sum(log(normpdf(X(1,:),mu3H,sig3H)));
numParam = [4; nnz(xi1)+nnz(yi1); nnz(xi2)+nnz(yi2)]; % Number of param
[aicH, bicH] = aicbic(logL, numParam, n*ones(3,1)); 

% Repeat for the Hare population 

% Get the underlying normal distribution
[mu1L,sig1L] = normfit(y_lv(:,2));
[mu2L,sig2L] = normfit(y_poly(:,2));
[mu3L,sig3L] = normfit(y_sin(:,2));

logL = zeros(3,1); % Preallocate loglikelihood vector
logL(1)=sum(log(normpdf(X(2,:),mu1L,sig1L)));
logL(2)=sum(log(normpdf(X(2,:),mu2L,sig2L)));
logL(3)=sum(log(normpdf(X(2,:),mu3L,sig3L)));
numParam = [4; nnz(xi1)+nnz(yi1); nnz(xi2)+nnz(yi2)]; 
[aicL, bicL] = aicbic(logL, numParam, n*ones(3,1)); 


