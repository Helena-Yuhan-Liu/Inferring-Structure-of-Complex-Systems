%% Identification for Cropped Faces 
% As seen from SVD, cropped images yield better approximations at 
% lower rank, so they will be used for the identification tasks.

% Data are not separated based on lighting, so that the classifier can 
% be trained and tested for images with varying degree of lighting 

% TODO: 
% analyze error cases, imshow the image, look at the subject number
% gscatter for PC

% setup
clear; close all; clc;
plot_en = 0; 

% Load data
% xdata, # pixels by # images, contains all the cropped faces image data  
load('cropped_data.mat'); 
xdata = double(xdata); 

N = size(xdata,2); % number of images
x_0mean = xdata - mean(xdata,2); % Remove baseline (mean) for each pixel

[Uc,Sc,Vc] = svd(xdata,0); 

%% Gender identification using supervised learning
% Various classifiers are tried to identify gender from face images

% Cross-validation by random split of data into training and test sets, 
% repeat for Ntrials times, and then take the average error rate 

rnks = 2:5:500; % rank to keep
Ntrials = 5; % number of cross-validation runs
error_svm = ones(length(rnks),Ntrials); 
error_svm_lin = ones(length(rnks),Ntrials);
error_nb = ones(length(rnks),Ntrials); 
error_lda = ones(length(rnks),Ntrials); 
error_knn = ones(length(rnks),Ntrials); 
error_tree = ones(length(rnks),Ntrials); 

for r = 1:length(rnks)
    rnk = rnks(r); 
    for ntrial = 1:Ntrials
        % Split data into train and test, also get the PC components for train
        % and test
        [xtrain, ctrain, xtest, ctest, Vtrain, Vtest]= ...
            get_train_test(xdata_cell, Vc, rnk, 'gender'); 

        % Run SVM with RBF kernel
        % selecting a good kernel is crucial, for example, high error rate with
        % mlp or linear kernel,              
        svm = fitcsvm(Vtrain,ctrain','KernelFunction','gaussian',...
            'KernelScale','auto');
        pre_svm = svm.predict(Vtest); 
        error_svm(r, ntrial) = sum(pre_svm~=ctest')/length(ctest');   
        % plot
        if plot_en
            figure; subplot(2,1,1); bar(pre_svm,'r'); ylabel('predicted labels');
            subplot(2,1,2); bar(ctest','b'); ylabel('actual labels');
            xlabel('test instance'); title('SVM with RBF');
        end 

        % SVM with linear kernel
        try
            svm = svmtrain(Vtrain, ctrain','kktviolationlevel',0.01); 
        catch 
            svm = svmtrain(Vtrain, ctrain','kktviolationlevel',0.1); 
        end 
        pre_svm_lin = svmclassify(svm, Vtest); 
        error_svm_lin(r, ntrial) = sum(pre_svm_lin~=ctest')/length(ctest');  

        % KNN, k found by trial and error
        neighb_idx = knnsearch(Vtrain,Vtest,'K',5);
        neighb_class = ctrain(neighb_idx); 
        pre_knn = mode(neighb_class,2); 
        error_knn(r, ntrial) = sum(pre_knn~=ctest')/length(ctest'); 
    %     neighb_idx = knnsearch(double(xtrain'),double(xtest'),'K',5);
    %     neighb_class = ctrain(neighb_idx); 
    %     pre_knn = mode(neighb_class,2); 
    %     error_knn(ntrial) = sum(pre_knn~=ctest')/length(ctest');
        % plot
        if plot_en
            figure; subplot(2,1,1); bar(pre_knn,'r'); ylabel('predicted labels');
            subplot(2,1,2); bar(ctest','b'); ylabel('actual labels');
            xlabel('test instance'); title('KNN'); 
        end 

        % Run LDA    
        pre_lda = classify(Vtest,Vtrain, ctrain'); 
        error_lda(r, ntrial) = sum(pre_lda~=ctest')/length(ctest');     
        % plot
        if plot_en
            figure; subplot(2,1,1); bar(pre_lda,'r'); ylabel('predicted labels');
            subplot(2,1,2); bar(ctest','b'); ylabel('actual labels');
            xlabel('test instance'); title('LDA');   
        end 

        % Run Naive Bayes
        nb = fitNaiveBayes(Vtrain, ctrain'); 
        pre_nb = nb.predict(Vtest); 
        error_nb(r, ntrial) = sum(pre_nb~=ctest')/length(ctest'); 
        % plot
        if plot_en
            figure; subplot(2,1,1); bar(pre_nb,'r'); ylabel('predicted labels');
            subplot(2,1,2); bar(ctest','b'); ylabel('actual labels');
            xlabel('test instance'); title('Naive Bayes'); 
        end 

        % Run Decision tree
        tree = fitctree(Vtrain, ctrain'); 
        pre_tree = predict(tree, Vtest); 
        error_tree(r, ntrial) = sum(pre_tree~=ctest')/length(ctest');  
    end 
end 

% Plot the error rate versus rank
figure; 
plot(rnks, mean(error_svm,2), 'r'); hold on; 
plot(rnks, mean(error_svm_lin,2), 'g'); 
plot(rnks, mean(error_nb,2), 'b');
plot(rnks, mean(error_lda,2), 'c'); 
plot(rnks, mean(error_knn,2), 'k');
plot(rnks, mean(error_tree,2), 'm'); hold off; 
xlabel('Number of PCs'); ylabel('Error Rate'); 
legend({'SVM with RBF kernel', 'SVM with linear kernel', 'Naive Bayes', ...
    'LDA', 'KNN', 'Tree'}); 
ylim([0 0.4]); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');
% For KNN and SVM RBF, better results when the rank is low
% For LDA and SVM linear, better results when the rank is high
% The choice of kernel function matters a lot for SVM 

save('class_gender.mat','rnks','error_svm','error_svm_lin','error_nb',...
    'error_lda','error_knn','error_tree');

%% Subject identification using supervised learning
% fitcecoc() is used to apply the above classifiers to multi-classes case

rnks = [2 5:10:500]; % rank to keep
Ntrials = 3; % number of cross-validation runs
error_svm = ones(length(rnks),Ntrials); 
error_svm_lin = ones(length(rnks),Ntrials);
error_nb = ones(length(rnks),Ntrials); 
error_lda = ones(length(rnks),Ntrials); 
error_knn = ones(length(rnks),Ntrials); 
error_tree = ones(length(rnks),Ntrials); 

for r = 1:length(rnks)
    rnk = rnks(r); 
    for ntrial = 1:Ntrials
        % Split data into train and test, also get the PC components for train
        % and test
        [xtrain, ctrain, xtest, ctest, Vtrain, Vtest]= ...
            get_train_test(xdata_cell, Vc, rnk, 'subject'); 
        
        % SVM with Gaussian kernel
        t = templateSVM('Standardize',true,'KernelFunction','gaussian',...
            'KernelScale','auto');
        svm = fitcecoc(Vtrain,ctrain','Learners',t);
        pre_svm = svm.predict(Vtest); 
        error_svm(r, ntrial) = sum(pre_svm~=ctest')/length(ctest');  

        % SVM with linear kernel
        t = templateSVM('Standardize',true,'KernelFunction','linear');
        svm = fitcecoc(Vtrain,ctrain','Learners',t);
        pre_svm_lin = svm.predict(Vtest); 
        error_svm_lin(r, ntrial) = sum(pre_svm_lin~=ctest')/length(ctest');  

        % KNN, k found by trial and error
        t = templateKNN('NumNeighbors',5,'Standardize',1);
        knn = fitcecoc(Vtrain,ctrain','Learners',t);
        pre_knn = knn.predict(Vtest); 
        error_knn(r, ntrial) = sum(pre_knn~=ctest')/length(ctest');

        % Run LDA    
        lda = fitcecoc(Vtrain,ctrain','Learners','discriminant');
        pre_lda = lda.predict(Vtest); 
        error_lda(r, ntrial) = sum(pre_lda~=ctest')/length(ctest');     
        % plot
        if plot_en
            figure; subplot(2,1,1); bar(pre_lda,'r'); ylabel('predicted labels');
            subplot(2,1,2); bar(ctest','b'); ylabel('actual labels');
            xlabel('test instance'); title('LDA');   
        end 

        % Run Naive Bayes
        nb = fitcecoc(Vtrain,ctrain','Learners','naivebayes');
        pre_nb = nb.predict(Vtest); 
        error_nb(r, ntrial) = sum(pre_nb~=ctest')/length(ctest'); 
        % plot
        if plot_en
            figure; subplot(2,1,1); bar(pre_nb,'r'); ylabel('predicted labels');
            subplot(2,1,2); bar(ctest','b'); ylabel('actual labels');
            xlabel('test instance'); title('Naive Bayes'); 
        end 

        % Run Decision tree
        tree = fitcecoc(Vtrain,ctrain','Learners','tree');
        pre_tree = tree.predict(Vtest); 
        error_tree(r, ntrial) = sum(pre_tree~=ctest')/length(ctest');  
    end 
end 

% Plot the error rate versus rank
figure; 
plot(rnks, mean(error_svm,2), 'r'); hold on; 
plot(rnks, mean(error_svm_lin,2), 'g'); 
plot(rnks, mean(error_nb,2), 'b');
plot(rnks, mean(error_lda,2), 'c'); 
plot(rnks, mean(error_knn,2), 'k');
plot(rnks, mean(error_tree,2), 'm'); hold off; 
xlabel('Number of PCs'); ylabel('Error Rate'); 
legend({'SVM with RBF kernel', 'SVM with linear kernel', 'Naive Bayes', ...
    'LDA', 'KNN', 'Tree'}); 
ylim([0 0.7]); 
set(gcf, 'position', [100 100 450 350]); set(gcf,'color','w');

save('class_subject.mat','rnks','error_svm','error_svm_lin','error_nb',...
    'error_lda','error_knn','error_tree');

%% K-means Clustering 
% For unsupervised learning, there is no teaching signal, so no 
% precise way of evaluating the performance. Usually domain knowledge 
% would be useful in evaluating if the clusters make sense 

% GMM was skipped b/c the dimension of the data far exceeds the 
% number of observations, making the cov matrix non-invertible  

% Try k=2 clusters 
[~, C2] = kmeans(xdata',2); 
mean_diff = abs(C2(1,:)-C2(2,:)); 
figure; 
subplot(1,3,1); imshow(reshape(uint8(mean_diff),[nrow,ncol])); 
title('Mean Difference'); 
subplot(1,3,2); imshow(reshape(uint8(C2(1,:)),[nrow,ncol])); 
title('Cluster 1 Mean'); 
subplot(1,3,3); imshow(reshape(uint8(C2(2,:)),[nrow,ncol])); 
title('Cluster 2 Mean'); 
set(gcf, 'position', [100 100 600 150]); set(gcf,'color','w');
% Pixels that really separate the clusters turn out to be edges of 
% facial parts, capturing features such as nose width 
% Lighting also got picked up, as seen in the two different cluster means

% Try k=5 clusters 
[~, C5] = kmeans(xdata',5); 
mean_diff = abs(C5(1,:)-C5(5,:));
for ii=2:5
    mean_diff = mean_diff+abs(C5(ii,:)-C5(ii-1,:));
end 
figure; 
subplot(2,3,1); imshow(reshape(uint8(mean_diff),[nrow,ncol])); 
title('Mean Difference'); 
for ii=1:5
    subplot(2,3,ii+1); imshow(reshape(uint8(C5(ii,:)),[nrow,ncol])); 
    title(['Cluster ' num2str(ii) '  Mean']); 
end 
set(gcf, 'position', [100 100 600 300]); set(gcf,'color','w');
% Again, pixels that really matter are those that picks up edges
% Also clustering picks up lighting, as seen in different cluster means
% These most important pixels can be used for more efficient
% classification, which is skipped in this work 

save('kmeans_data.mat','C2','C5'); 

        