function res = kNN_classifier(Train_array, Test_array,...
                                Train_array_response, ~,...
                                classes, train_set_size, test_set_size, k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% [res, kNN_success_rate, kNN_conf_matrix] = kNN_classifier(Train_array,
%                       Test_array, Train_array_response, l, classes, k)
%
% This function classifies a set of data vectors using the kNN classifier 
% for k neighbors in one out of c possible classes.
%
% INPUT ARGUMENTS:
% Train_array: an lxN dimensional matrix whose columns are the data vectors
% to be used as training set
% Test_array: an lxN dimensional matrix whose columns are the data vectors
% to be classified.
% Train_array_response: an 1xN dimensional matrix whose columns are class
% of the i-th element
% l: The dimension of each chracterisic
% classes: The number of the classes
% train_array_size: The size of the train set
% test_array_size: The size of the test set
% k: The number of the neighbors we'll check
%
% OUTPUT ARGUMENTS
% res: an N-dimensional vector whose i-th component contains the label
% of the class where the i-th data vector has been assigned.
% kNN_success_rate: Succes rate of euclidean classifier 
% kNN_conf_matrix: The confusion matrix given for the current train and
% test set
%
% (c) 2019 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Get the amount of the classes we have
%[~, classes] = size(unique(Train_array_response_init));
% Get the number of the points we have in train set
%[~, train_set_size]=size(Train_array_response_init);

res(1:train_set_size) = 0;
for i = 1:test_set_size 
    % Calclate distance for each point in test set from each point in
    % train set
    dist(1:train_set_size) = 0;
    for j=1:train_set_size
        vects_to_calc = [(Train_array(:,j))'; Test_array(:,i)'];
        dist(j) = pdist(vects_to_calc, 'euclidean');
    end

    % Sort the distances and keep the indexes as well
    [~, sorted_idxs] = sort(dist);
    neighbors(1:classes)=0;
    for j = 1:k  % Check the k nearest neighbors
        % Get the index of th j-th nearest neighbor
        cur_elem = sorted_idxs(j);
        cur_class = Train_array_response(cur_elem);  % Get its class
        % Count how many neighbors we have from each class
        neighbors(cur_class) = neighbors(cur_class) + 1;
    end
    
    % Sort the amount of neigbors for each class and get the indexes
    [~, idxs] = sort(neighbors);
    % The class for i-th element will be the class with the most neighbors.
    % Which is the last in the sorted array.
    res(i) = idxs(classes);
end