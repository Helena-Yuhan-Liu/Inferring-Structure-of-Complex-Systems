%% DMD for BZ Reaction data
% Follows similar DMD analysis procedure as for the predator-prey data

clc; clear all; close all;
load('BZ.mat'); 

[m,n,k]=size(BZ_tensor); % x vs y vs time data
% for j=1:k         % visualize data
%     A=BZ_tensor(:,:,j);
%     pcolor(A), shading interp, pause(0.2)
% end

%% Look at a sample frame

% For simplicity, only sample snapshots are analyzed 

% Moreover, only diagonal pixels are analyzed. This is not 
% only for simplcity, but the video suggests that there is a source 
% at the top right corner that's diffusing diagonally 

% Example frame at 700
A=BZ_tensor(:,:,700); X=diag(A)'; p=1:length(X); l=length(X); 
figure; pcolor(A); shading interp; hold on;
plot(p,p,'r','Linewidth',[2]); hold off;
xlabel('row pixels'); ylabel('col pixels'); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');
% subplot(1,2,2); plot(p,X);

% Again, repeat the sliding window stuff
% Settings to show in the report
wstep=10; ws=wstep+80; 
wstart = 0:wstep:(l-ws); % window starting indices

r=1;    % number of modes, at most 1 b/c U is 1 by 1
u_dmd = zeros(r,l); 

for ii = 1:length(wstart)
    % data length used for training vs data length used for prediction
    if ii==length(wstart) % last window
        ws2 = l-wstart(ii); 
        wtrain = ws - wstep; 
    else
        ws2 = ws; 
        wtrain = ws2 - wstep; 
    end 
    wtest = ws2-wtrain;     % save some years for prediction
    X1 = X((wstart(ii)+1):(wstart(ii)+wtrain)); 
    X2 = X((wstart(ii)+2):(wstart(ii)+wtrain+1));
    [U2,Sigma2,V2] = svd(X1,'econ'); U=U2(:,1:r); Sigma=Sigma2(1:r,1:r); V=V2(:,1:r);

    % DMD code from the lecture 
    Atilde = U'*X2*V/Sigma;    
    [W,D] = eig(Atilde);    
    Phi = X2*V/Sigma*W;

    mu = diag(D);
    omega = log(mu);
%     omega(real(omega)>0.5)=0.5+...
%         1i*imag(omega(real(omega)>0.5)); % clip e-val with large real part
% %     disp('The eigen values of the modes are:');
% %     disp(omega);

    u0=X1(1);
    y0 = Phi\u0;  % pseudo-inverse initial conditions
    u_modes = zeros(r,wtrain);
    for iter = 1:wtrain
         u_modes(:,iter) =(y0.*exp(omega*p(iter)));
    end
    u_dmd_train = Phi*u_modes;
    
    % forecast
    u_modes2 = zeros(r,wtest);
    for iter = 1:wtest
         u_modes2(:,iter) =(y0.*exp(omega*p(iter+wtrain)));
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
figure; plot(p,X,p,u_dmd,'Linewidth',[2]); 
legend('Actual', 'Predicted'); 
xlabel('Diagonal Pixel Index'); ylabel('Reaction Data'); 
set(gcf, 'position', [100 100 450 200]); set(gcf,'color','w');

%% Time delay embedding

% Set using similar approach as for the predator-prey data
wstep=70; dly=60; ws=wstep+80; 

% Construct the Hankel matrix
H=[]; 
for j=1:dly         
  H=[H; X(:,j:(l-dly)+j)]; 
end  

% Starting point for each sliding window
wstart = 0:wstep:(l-dly-ws+wstep); 
r=1*dly;    % one variable per delayed points
u_dmd_td = zeros(r,l); 

for ii = 1:length(wstart)
    % data length used for training vs data length used for prediction
    if ii==length(wstart) % last window
        ws2 = l-wstart(ii); 
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
    omega = log(mu);
    omega(real(omega)>0.01)=0.01+...
        1i*imag(omega(real(omega)>0.01)); % clip e-val with large real part
%     disp('The eigen values of the modes are:');
%     disp(omega);

    u0=X1(:,1);     % Now the new y0 has larger dimension
    y0 = Phi\u0;  % pseudo-inverse initial conditions
    u_modes = zeros(r,wtrain);
    for iter = 1:wtrain
         u_modes(:,iter) =(y0.*exp(omega*p(iter)));
    end
    u_dmd_train = Phi*u_modes;
    
    % forecast
    u_modes2 = zeros(r,wtest);
    for iter = 1:wtest
         u_modes2(:,iter) =(y0.*exp(omega*p(iter+wtrain)));
    end
    u_dmd_test = Phi*u_modes2;
    
    if ii ==1 % first window
        u_dmd_td(:,(wstart(ii)+1):(wstart(ii)+ws2)) = [u_dmd_train u_dmd_test];   
    else
        u_dmd_td(:,(wstart(ii)+wtrain+1):(wstart(ii)+ws2)) = u_dmd_test; 
    end   
end 

% Compare the actual data with fitted and predicted modes 
figure; plot(p,X,p,u_dmd_td(1,:),'Linewidth',[2]); 
legend('Actual', 'Predicted'); 
xlabel('Diagonal Pixel Index'); ylabel('Reaction Data'); 
set(gcf, 'position', [100 100 450 200]); set(gcf,'color','w');

% SVD on matrix H to see dimensionality 
[u,s,v]=svd(H,'econ');
figure; subplot(2,1,1), plot(diag(s)/(sum(diag(s))),'ro','Linewidth',[3])
subplot(2,1,2), plot(v(:,1:3),'Linewidth',[2])
% only one sing val with relative energy above 0.05, 
% latent variables probably have very little effect 


