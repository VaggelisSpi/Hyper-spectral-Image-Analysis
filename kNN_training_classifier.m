function ret = kNN_training_classifier(train_set_idxs, test_set_idxs,...
                            Train_array, Train_array_response, ~, classes,...
                            train_set_size, k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% [out_kNN] = kNN_training_classifier(train_set_idxs, test_set_idxs, 
%                       Train_array, Train_array_response, l, classes, k)
%
% This function classifies a set of data vectors in one out of c possible
% classes, according to the k nearest neighbours classifier
%
% INPUT ARGUMENTS:
% train_set_idxs: an array with the index of each point in the trainign
% set. The indexes correspond to a column in the Train array or a line in
% response array
% test_set_idxs: an array with the index of each point in the test
% set. The indexes correspond to a column in the Train array or a line in
% response array
% Train_array: an lxN dimensional matrix whose columns are the data vectors 
% to be used as training and test set
% Train_array_response: an 1xN dimensional matrix whose columns are class
% of the i-th element
% l: The dimension of each chracterisic
% classes: The number of the classes
% train_array_size: The size of the train set
% test_array_size: The size of the test set
% k: The number of the neighbors we'll check
%
% OUTPUT ARGUMENTS
% ret: how many points were classified correctly
%
% (c) 2019 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cor_class = 0; % Amount of corrrect classified points
res(1:train_set_size) = 0;
for i = 1:train_set_size 
    % Calculate distance for each point in test set from each point in train 
    % set
    for j=1:train_set_size
        test_elem = test_set_idxs(i);
        train_elem = train_set_idxs(j);
        vects_to_calc = [(Train_array(:,train_elem))';...
                          Train_array(:,test_elem)'];
        dist(j) = pdist(vects_to_calc, 'euclidean');
    end

    % Sort the distances and keep the sorted indexes
    [~, sorted_idxs] = sort(dist);
    neighbors(1:classes)=0;
    for j = 1:k  % Check the k nearest neighbors
        % Get the index of th j-th nearest neighbor
        cur_elem = train_set_idxs(sorted_idxs(j));  
        cur_class = Train_array_response(cur_elem);  % Get its class
        % Count how many neighbors we have from each class
        neighbors(cur_class) = neighbors(cur_class) + 1; 
    end
    
    % Sort the amount of neigbors for each class and get the indexes
    [~, idxs] = sort(neighbors);
    
    %[~, classes] = size(neighbors);  % Get the number of classes

    % The class for i-th element will be the class with the most neighbors.
    % Which is the last in the sorted array.
    res(i) = idxs(classes);

    % Count how many points were classified correctly
    if (res(i) == Train_array_response(test_set_idxs(i)))
        cor_class = cor_class + 1; 
    end

end

ret = cor_class;