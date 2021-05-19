function [success_rate, conf_matrix] = calc_success_rate(out_class,...
                                        Test_array_response, classes,...
                                        test_set_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% [success_rate, conf_matrix] = calc_succes_rate(out_class,
%                                                train_array_response)
% This function calculates the success rate of a classifier and its
% confusion matrix
%
% INPUT ARGUMENTS:
% out_class: an 1xN dimensional matrix, whose i-th column corresponds to 
% the class of the i-th point resulted from the classifier.
% Test_array_response: an 1xN dimensional matrix, whose i-th column 
% corresponds to the correct class of the i-th point.
% classes: The number of the classes
% test_set_size: The size of the test set
%
% OUTPUT ARGUMENTS
% success_rate: The percentage of the points that were classified correctly
% conf_matrix: The confusion matrix for the current sets. It's a CxC matrix
% whre C is the number of classes. i,j element show how many points come
% from class i but are classified in class j
%
% (c) 2019 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get the amount of the classes we have
%[~, classes] = size(unique(Test_array_response));
% Get the number of the points we have in train set
%[~, test_set_size] = size(Test_array_response);

% Make the confusion matrix
conf_matrix(1:classes, 1:classes) = 0;
for i=1:test_set_size
    right_class = Test_array_response(i);
    res_class = out_class(i);
    conf_matrix(right_class, res_class) = conf_matrix(right_class,...
                                                      res_class) + 1;
end

% Calculate the sum of the diagonal
diag_sum = 0;
for i=1:classes
   diag_sum = conf_matrix(i,i) + diag_sum;  
end

% Calculate the sum of all the elemnets of the confusion matrix
total_sum = 0;
for i=1:classes
    for j=1:classes
        total_sum = total_sum + conf_matrix(i,j);
    end
end

success_rate = diag_sum/total_sum;