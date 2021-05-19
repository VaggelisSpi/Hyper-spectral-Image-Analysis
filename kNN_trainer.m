function best_k = kNN_trainer(Train_array, Train_array_response,... 
                              l, classes, train_set_size, test_set_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% ret = kNN_trainer(Train_array, Train_array_response, l)
% This function splits the Train array into 5 random subarrays. Then for
% each value of k it uses 4 of the sub arrays as training set and one as
% test set and classifies it. It does this 5 times each time with different
% train and test sets and calclulates the error rate for each value of k.
% The function will return the best value of k
%
% INPUT ARGUMENTS:
% Train_array: an lxN dimensional matrix whose columns are the data 
% vectors to be used as training and test set
% Train_array_response: an 1xN dimensional matrix whose columns are 
% l: The dimension of each chracterisic
% classes: class of the i-th element
% train_set_size: The size of the train set
% test_set_size: The size of the test set
%
% OUTPUT ARGUMENTS
% ret: The value of the neighbours that give the least error
%
% (c) 2019 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Shuffle the point indexes randomly so when we are gonna split our set 
% the new sets will be random
point_idxs = randperm(train_set_size);  

%err(1:)=0;
best_err = -1;
best_k = 0;
%err_idx = 1;

% Check all k values to find out which one is the best
for k=1:2:17    
    % Split the original train set in 5 parts, use 4 as train set and one 
    % as test set
    test_set_start = 1;
    interv = floor(train_set_size/5) - 1; % Where the next set will start
    test_set_end = test_set_start + interv; % Where current set will end
    cor_class = 0; % Correct classified for current k
    % The arrays contain the index of each set
    new_test_set_size = test_set_end - test_set_start + 1;
    
    for r = 1:5  % 5 itterations for each different test set
        test_count = 1;
        train_count = 1;
        
        % Add all the points before the start of test set to the train set
        % if the train set won't start from the beggining
        if (test_set_start ~= 1)
            for i=1:test_set_start - 1  
                train_set_idxs(train_count) = point_idxs(i);
                train_count = train_count + 1;
            end
        end

        % Make the test set
        for i=test_set_start:test_set_end  
            test_set_idxs(test_count) = point_idxs(i);
            test_count = test_count + 1;
        end

        % Add all the points after the start of test set to the train set
        for i=test_set_end+1:train_set_size  
            train_set_idxs(train_count) = point_idxs(i);
            train_count = train_count + 1;
        end

        % Get how many were correctly classified for current train and test
        % sets
        cor_class = cor_class + kNN_training_classifier(train_set_idxs,...
            test_set_idxs, Train_array, Train_array_response, l,...
            classes, new_test_set_size, k);
        test_set_start = test_set_end + 1;
        test_set_end = test_set_end + interv;
    end 

    % Find the error for the curent value of k and keep the best
    err = cor_class/train_set_size;
    if (best_err == -1)
        best_err = err;
        best_k = k;
    else 
       if (err < best_err) 
          best_err = err;
          best_k = k;
       end
    end
    
end