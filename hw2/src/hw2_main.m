%% Setup up data

% X is a 2 by T matrix
%   X(1,:) contains the Hare population over time
%   X(2,:) contains the Lynx population over time

clear all; close all; clc; 
format compact; 

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

% Visualize data
figure; plot(t,X(1,:), 'r'); hold on; plot(t,X(2,:),'b'); hold off;
xlabel('Year since 1854'); ylabel('population'); legend('Hare','Lynx'); 

%% DMD sliding window
% Note, I am using the sliding window approach where within each window, 
% parts of the data is allocated for training and the rest is used for 
% prediction. This gives me better result, perhaps due to the short 
% prediction window. 

% Need overlapping window, because without it the prediction is 
% very discontinuous 

wstep= round(2/dt); % small test window
ws=wstep+round(20/dt); % window size
wstart = 0:wstep:(round(58/dt)-ws); % window starting indices

r=2;    % number of modes, at most 2 b/c U is 2 by 2
u_dmd = zeros(r,round(58/dt)); 

for ii = 1:length(wstart)
    % data length used for training vs data length used for prediction
    if ii==length(wstart) % last window
        ws2 = round(58/dt)-wstart(ii); 
        wtrain = ws - wstep; 
    else
        ws2 = ws; 
        wtrain = ws2 - wstep; 
    end 
    wtest = ws2-wtrain; 
    X1 = X(:,(wstart(ii)+1):(wstart(ii)+wtrain)); 
    X2 = X(:,(wstart(ii)+2):(wstart(ii)+wtrain+1));
    [U2,Sigma2,V2] = svd(X1,'econ'); U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);

    % DMD code from the lecture 
    Atilde = U'*X2*V/Sigma;    
    [W,D] = eig(Atilde);    
    Phi = X2*V/Sigma*W;

    mu = diag(D);
    omega = log(mu)/dt;
    omega(real(omega)>0.5)=0.5+...
        1i*imag(omega(real(omega)>0.5)); % clip e-val with large real part
%     disp('The eigen values of the modes are:');
%     disp(omega);

    u0=X1(:,1);     % Now the new y0 has larger dimension
    y0 = Phi\u0;  % pseudo-inverse initial conditions
    u_modes = zeros(r,wtrain);
    for iter = 1:wtrain
         u_modes(:,iter) =(y0.*exp(omega*t(iter)));
    end
    u_dmd_train = Phi*u_modes;
    
    % forecast
    u_modes2 = zeros(r,wtest);
    for iter = 1:wtest
         u_modes2(:,iter) =(y0.*exp(omega*t(iter+wtrain)));
    end
    u_dmd_test = Phi*u_modes2;
    
    % append the window result to the final result 
    if ii ==1 % first window
        u_dmd(:,(wstart(ii)+1):(wstart(ii)+ws2)) = [u_dmd_train u_dmd_test];   
    else
        u_dmd(:,(wstart(ii)+wtrain+1):(wstart(ii)+ws2)) = u_dmd_test; 
    end 
end 

% Comparing the actual data with fitted & predicted modes
figure;
subplot(2,1,1), plot(t,X(1,:),t,u_dmd(1,:),'Linewidth',[2]); 
legend('Actual', 'Predicted'); 
xlabel(' Time (year since 1854)'); ylabel('Hare Population'); 
subplot(2,1,2), plot(t,X(2,:),t,u_dmd(2,:),'Linewidth',[2]);
legend('Actual', 'Predicted'); 
xlabel(' Time (year since 1854)'); ylabel('Lynx Population'); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');

%% Repeat for Time delay embedding
% first, specify number of delay points, which are set so that modes look 
% somewhat sinusoidal,

dly=8; % number of years to be delayed
wstep= round((dly+2)/dt); % test window slightly higher than dly 
ws=wstep+round(20/dt); % window size

% Construct the Hankel matrix
H=[]; 
for j=1:round(dly/dt)         
  H=[H; X(:,j:round((58-dly)/dt)+j)]; 
end  

wstart = 0:wstep:(round((58-dly)/dt)-ws+wstep); 
r=2*round(dly/dt);    % two variables per delayed points
u_dmd_td = zeros(r,round(58/dt)); 

for ii = 1:length(wstart)
    % data length used for training vs data length used for prediction
    if ii==length(wstart) % last window
        ws2 = round(58/dt)-wstart(ii); 
        wtrain = ws - wstep; 
        wtest = ws2 - wtrain;   % test on wstep + dly points 
    else
        ws2 = ws; 
        wtrain = ws2 - wstep; 
        wtest = wstep;
    end 
    X1 = H(:,(wstart(ii)+1):(wstart(ii)+wtrain)); 
    X2 = H(:,(wstart(ii)+2):(wstart(ii)+wtrain+1));
    [U2,Sigma2,V2] = svd(X1,'econ'); U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);

    % DMD code from the lecture 
    Atilde = U'*X2*V/Sigma;    
    [W,D] = eig(Atilde);    
    Phi = X2*V/Sigma*W;

    mu = diag(D);
    omega = log(mu)/dt;
    omega(real(omega)>0.5)=0.5+...
        1i*imag(omega(real(omega)>0.5)); % clip e-val with large real part
%     disp('The eigen values of the modes are:');
%     disp(omega);

    u0=X1(:,1);     % Now the new y0 has larger dimension
    y0 = Phi\u0;  % pseudo-inverse initial conditions
    u_modes = zeros(r,wtrain);
    for iter = 1:wtrain
         u_modes(:,iter) =(y0.*exp(omega*t(iter)));
    end
    u_dmd_train = Phi*u_modes;
    
    % forecast
    u_modes2 = zeros(r,wtest);
    for iter = 1:wtest
         u_modes2(:,iter) =(y0.*exp(omega*t(iter+wtrain)));
    end
    u_dmd_test = Phi*u_modes2;
    
    if ii ==1 % first window
        u_dmd_td(:,(wstart(ii)+1):(wstart(ii)+ws2)) = [u_dmd_train u_dmd_test];   
    else
        u_dmd_td(:,(wstart(ii)+wtrain+1):(wstart(ii)+ws2)) = u_dmd_test; 
    end   
end 

% Compare the actual data with fitted and predicted modes 
figure;
subplot(2,1,1), plot(t,X(1,:),t,u_dmd_td(1,:),'Linewidth',[2]); 
legend('Actual', 'Predicted'); 
xlabel(' Time (year since 1854)'); ylabel('Hare Population'); 
subplot(2,1,2), plot(t,X(2,:),t,u_dmd_td(2,:),'Linewidth',[2]);
legend('Actual', 'Predicted'); 
xlabel(' Time (year since 1854)'); ylabel('Lynx Population'); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');

% SVD on matrix H to see dimensionality 
[u,s,v]=svd(H,'econ');
figure; subplot(2,1,1), plot(diag(s)/(sum(diag(s))),'ro','Linewidth',[3])
xlabel('Singular Value Index'); ylabel('Proportion'); 
subplot(2,1,2), plot(v(:,1:3),'Linewidth',[2]); 
xlabel(' Time (year since 1854)'); ylabel('PCs of H'); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');
% Turns out that five modes have relative energy above 0.05
% this implies there may be hidden factors influencing the 
% population dynamics, making low dim model hard to attain 

% Bottom line: with enough time delayed points (so the modes look 
% somewhat linear), the performance is much better than without 
% time delay embedding 

%% KL, AIC BIC
% For hare, KL does not capture the striking improvement from TD, 
% probably b/c KL has the huge caveat of not taken timing into account 
% For lynx, KL does, b/c lots of data outside of the bin 

% First, for the Hare population 
edges=0:15:150;     % bin width 10% of the data range 
KL_H = get_KL(edges, X(1,:), real(u_dmd(:,1))); 
KL_H_td = get_KL(edges, X(1,:), real(u_dmd_td(:,1))); 

% Then Lynx
edges=0:8:80;
KL_L = get_KL(edges, X(2,:), real(u_dmd(:,2))); 
KL_L_td = get_KL(edges, X(2,:), real(u_dmd_td(:,2))); 

% AIC BIC for without time-delay embedding

% First, compute AIC BIC for the Hare population 
[muH,sigH] = normfit(real(u_dmd(:,1)));
logL=sum(log(normpdf(X(1,:),muH,sigH))); 
[aicH, bicH] = aicbic(logL, length(t), length(t)); 

% Then, compute AIC BIC for the Lynx population 
[muL,sigL] = normfit(real(u_dmd(:,2)));
logL=sum(log(normpdf(X(2,:),muL,sigL))); 
[aicL, bicL] = aicbic(logL, length(t), length(t)); 

% AIC BIC for time delayed embedding 

% First, compute AIC BIC for the Hare population 
[muH,sigH] = normfit(real(u_dmd_td(:,1)));
logL=sum(log(normpdf(X(1,:),muH,sigH))); 
[aicHtd, bicHtd] = aicbic(logL, length(t), length(t)); 

% Then, compute AIC BIC for the Lynx population 
[muL,sigL] = normfit(real(u_dmd_td(:,2)));
logL=sum(log(normpdf(X(2,:),muL,sigL))); 
[aicLtd, bicLtd] = aicbic(logL, length(t), length(t)); 

