function [means, eucl_succes_rate, eucl_conf_matrix]=euclidean_trainer(...
     Train_array, Test_array, Train_array_response, Test_array_response, l,...
     classes, train_set_size, test_set_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% [means, eucl_succes_rate, eucl_conf_matrix] = euclidean_trainer(
% Train_array, Test_array, Train_array_response, 
% Test_array_response, l)
%
% This function trains a euclidean classifier and classifies a set of data 
% vectors in one out of c possible classes. 
% It also caclulates the success rate and the confusion matrix
%
% INPUT ARGUMENTS:
% Train_array: an lxN dimensional matrix whose columns are the data vectors 
% to be used as training set
% Test_array: an lxN dimensional matrix whose columns are the data vectors 
% to be used as test set
% Train_array_response: an 1xN dimensional matrix whose columns are the 
% class of the i-th element
% Test_array_response: an 1xN dimensional matrix whose columns are the
% class of the i-th element
% l: The dimension of each chracterisic
% classes: The number of the classes to classify our data
% train_set_size: The size of the train set
% test_set_size: The size of the test set
%
% OUTPUT ARGUMENTS
% means: The means of each class
% eucl_success_rate: Succes rate of euclidean classifier 
% eucl_conf_matrix: The confusion matrix given for the current train and
% test set
%
% (c) 2019 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sums is gonna be a classesxl array holding the sum of the values for each
% characteristic for each class
sums(1:classes,1:l)=0;      
class_count(1:classes)=0;  % elements in each class
for i=1:train_set_size
    cur_cat = Train_array_response(i);
    class_count(cur_cat) = class_count(cur_cat) + 1;
    % For each element in train array get the total sum of the values of 
    % its characteristics
    for j=1:l
        % sum the value of each characteristic
        sums(cur_cat, j) = sums(cur_cat, j) + Train_array(j, i);
    end
end

% Means is gonna be a number of classes*l array holding the
% mean for each characteristic for each class
means(1:classes,1:l)=0;

% Get the total average to calculate the means
for i=1:classes
    for j=1:l
        means(i, j) = sums(i, j)/class_count(i);
    end
end

out_eucl = euclidean_classifier(means, Test_array, classes, test_set_size);

[eucl_succes_rate, eucl_conf_matrix] = calc_success_rate(out_eucl,...
                                        Test_array_response, classes,...
                                        test_set_size);