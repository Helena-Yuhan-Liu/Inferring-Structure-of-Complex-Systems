function [xtrain, ctrain, xtest, ctest, Vtrain, Vtest]...
    = get_train_test(xdata_cell, V, rnk, task)
% Split data for 80% for training, 20% for validation. The split is done 
% for every subject, which allow the training data to represent the general 
% data distribution and perform better than the 80-20 split for the 
% entire dataset as a whole. 

% Input: all cropped face image data; task can be 'gender' or 'subject' 
% Output: 80% of the data for every subject is used for training, Also 
% returns the split for PC coordinate V with specified rank
% 20% of the data for every subject is used for testing. 

xtrain=[]; ctrain=[]; xtest=[]; ctest=[]; Vtrain=[]; Vtest=[]; 
count = 0; 
for sn=1:39
    if sn~=14   % subject 14 data missing
        % get image index for training and testing 
        npic = size(xdata_cell{sn},2); 
        idx = randperm(npic); 
        train_idx = idx(1:round(0.8*npic)); 
        test_idx = idx(round(0.8*npic)+1:end); 
        
        % split data and labels 
        if strcmp(task, 'gender')            
            ctrain = [ctrain get_gender(sn)*ones(1,length(train_idx))];           
            ctest = [ctest get_gender(sn)*ones(1,length(test_idx))]; 
        elseif strcmp(task, 'subject')
            ctrain = [ctrain sn*ones(1,length(train_idx))];           
            ctest = [ctest sn*ones(1,length(test_idx))];
        end 
        xtrain = [xtrain xdata_cell{sn}(:,train_idx)];  
        xtest = [xtest xdata_cell{sn}(:,test_idx)]; 
        
        Vtrain = [Vtrain; V(count+train_idx,1:rnk)];
        Vtest = [Vtest; V(count+test_idx,1:rnk)];
        count = count+npic; 
    end 
end 
        