%% SVD for Cropped Faces 

% Load data
% xdata, # pixels by # images, contains all the cropped faces image data 
clear; close all; clc; 
load('cropped_data.mat'); 
xdata = double(xdata); 

N = size(xdata,2); % number of images
x_0mean = xdata - mean(xdata,2); % Remove baseline (mean) for each pixel

%% Run SVD
% U: columns contain the PCs, i.e. the basis for the data
% V: projection of data onto U, i.e. the component of X on the basis
% S: singular value denoting the strength of each modes 

% SVD for cropped images
[Uc,Sc,Vc] = svd(xdata,0);
[Uc_0,Sc_0,Vc_0] = svd(x_0mean,0);

% Singular values

figure; semilogy(1:length(Sc),Sc,'.');
xlabel('PC #'); ylabel('Singular Value');

% Cumulative energy plot 
Ek=diag(Sc).^2; 
Hk_c=Ek./sum(Ek); 
figure; plot(1:length(Hk_c), cumsum(Hk_c), 'r'); hold on;
plot(1:length(Hk_c),0.95*ones(size(1:length(Hk_c))),'b'); hold off; 
ylabel('Cumulative Energy'); xlabel('PC #'); xlim([0 100]);
set(gcf, 'position', [100 100 250 200]); set(gcf,'color','w');

% %Repeat for images with baseline removed
% figure;
% subplot(1,2,1); semilogy(1:length(Sc_0),Sc_0,'.');
% xlabel('PC #'); ylabel('Singular Value')
% 
% % Cumulative energy plot 
% Ek_0=diag(Sc_0).^2; 
% Hk_0=Ek_0./sum(Ek_0); 
% subplot(1,2,2); plot(1:length(Hk_0), cumsum(Hk_0), 'r'); hold on;
% plot(1:length(Hk_0),0.95*ones(size(1:length(Hk_0))),'b'); hold off; 
% ylabel('Cumulative Energy (%)'); xlabel('PC #');

% turns out w/o de-mean, 20 modes needed for 95% of the energy
% with mean removed, 75 modes needed for 95% of the energy 

%% Rank approximation to one example face
imnum=1; % example image idx

% original xdata
rnk = [1 5 20 50 100];
figure;
subplot(2,3,1)
imshow(reshape(uint8(xdata(:,imnum)),[nrow,ncol]))
title('original')
for k = 1:length(rnk)
    Xc_approx = Uc(:,1:rnk(k))*Sc(1:rnk(k),1:rnk(k))*Vc(:,1:rnk(k))';
    C = reshape(uint8(Xc_approx(:,imnum)),[nrow,ncol]);
    subplot(2,3,k+1); imshow(C);
    title(['Rank = ', num2str(rnk(k))])
end
set(gcf, 'position', [100 100 450 300]); set(gcf,'color','w');

% % mean removed 
% figure;
% subplot(2,3,1)
% imshow(reshape(uint8(x_0mean(:,imnum)),[nrow,ncol]))
% title('original')
% for k = 1:length(rnk)
%     Xc_approx = Uc_0(:,1:rnk(k))*Sc_0(1:rnk(k),1:rnk(k))*Vc_0(:,1:rnk(k))';
%     C = reshape(uint8(Xc_approx(:,imnum)),[nrow,ncol]);
%     subplot(2,3,k+1); imshow(C);
%     title(['Rank = ', num2str(rnk(k))])
% end

%% Plot the top four dominant modes 
m = 4;
% figure; 
% for k = 1:m
%     subplot(1,m,k); 
%     imshow(reshape(Uc_0(:,k),[nrow,ncol]),[min(Uc_0(:,k)); max(Uc_0(:,k))])
%     title(['Mode ', num2str(k)])
% end

figure; 
for k = 1:m    
    subplot(1,m,k); 
    imshow(reshape(Uc(:,k),[nrow,ncol]),[min(Uc(:,k)) max(Uc(:,k))])
    title(['Eigenface ', num2str(k)])
end
set(gcf, 'position', [100 100 600 150]); set(gcf,'color','w');

%% SVD for Cropped Faces 
% Load data
% xdata, # pixels by # images, contains all the cropped faces image data 
load('uncropped_data.mat'); 
xdata = double(xdata); 

N = size(xdata,2); % number of images
x_0mean = xdata - mean(xdata,2); % Remove baseline (mean) for each pixel

%% Run SVD for uncropped images

% SVD for cropped images
[Uu,Su,Vu] = svd(xdata,0);
[Uu_0,Su_0,Vu_0] = svd(x_0mean,0);

% Singular values

figure;semilogy(1:length(Su),Su,'.');
xlabel('PC #'); ylabel('Singular Value');

% Cumulative energy plot 
Ek=diag(Su).^2; 
Hk_u=Ek./sum(Ek); 
figure; plot(1:length(Hk_u), cumsum(Hk_u), 'r'); hold on;
plot(1:length(Hk_u),0.95*ones(size(1:length(Hk_u))),'b'); hold off; 
ylabel('Cumulative Energy'); xlabel('PC #'); xlim([0 100]);
set(gcf, 'position', [100 100 250 200]); set(gcf,'color','w');

% %Repeat for images with baseline removed
% figure;
% subplot(1,2,1); semilogy(1:length(Su_0),Su_0,'.');
% xlabel('PC #'); ylabel('Singular Value')
% 
% % Cumulative energy plot 
% Ek_0=diag(Su_0).^2; 
% Hk_0=Ek_0./sum(Ek_0); 
% subplot(1,2,2); plot(1:length(Hk_0), cumsum(Hk_0), 'r'); hold on;
% plot(1:length(Hk_0),0.95*ones(size(1:length(Hk_0))),'b'); hold off; 
% ylabel('Cumulative Energy (%)'); xlabel('PC #');

% turns out w/o de-mean, 1 modes needed for 95% of the energy
% with mean removed, 20 modes needed for 95% of the energy 

%% Rank approximation to the faces.
imnum=1; % example image idx

% original xdata
rnk = [1 5 20 50 100];
figure;
subplot(2,3,1)
imshow(reshape(uint8(xdata(:,imnum)),[nrow,ncol]))
title('original')
for k = 1:length(rnk)
    Xc_approx = Uu(:,1:rnk(k))*Su(1:rnk(k),1:rnk(k))*Vu(:,1:rnk(k))';
    C = reshape(uint8(Xc_approx(:,imnum)),[nrow,ncol]);
    subplot(2,3,k+1); imshow(C);
    title(['Rank = ', num2str(rnk(k))])
end
set(gcf, 'position', [100 100 450 300]); set(gcf,'color','w');

% % mean removed 
% figure;
% subplot(2,3,1)
% imshow(reshape(uint8(x_0mean(:,imnum)),[nrow,ncol]))
% title('original')
% for k = 1:length(rnk)
%     Xc_approx = Uu_0(:,1:rnk(k))*Su_0(1:rnk(k),1:rnk(k))*Vu_0(:,1:rnk(k))';
%     C = reshape(uint8(Xc_approx(:,imnum)),[nrow,ncol]);
%     subplot(2,3,k+1); imshow(C);
%     title(['Rank = ', num2str(rnk(k))])
% end

%% Plot the top four dominant modes 
m = 4;
% figure; 
% for k = 1:m
%     subplot(1,m,k); 
%     imshow(reshape(Uu_0(:,k),[nrow,ncol]),[min(Uu_0(:,k)); max(Uu_0(:,k))])
%     title(['Mode ', num2str(k)])
% end

figure; 
for k = 1:m    
    subplot(1,m,k); 
    imshow(reshape(Uu(:,k),[nrow,ncol]),[min(Uu(:,k)) max(Uu(:,k))])
    title(['Eigenface ', num2str(k)])
end
set(gcf, 'position', [100 100 600 150]); set(gcf,'color','w');

% Comparison:
% Uncropped images require fewer modes to capture 95% of the energy, 
% but those dominant modes are focused on capturing edges of the head
% position, rather than edges of the facial features. 
% Hence, it is not surprising that cropped figure achieves a better
% approximation with fewer modes. 
