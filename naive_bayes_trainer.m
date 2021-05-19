function [bayes_means, bayes_sv, a_priori, bayes_success_rate,...
    bayes_conf_matrix] = naive_bayes_trainer(Train_array, Test_array,...
    Train_array_response, Test_array_response, l, classes, train_set_size,...
    test_set_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% [out_bayes] = naive_bayes_trainer(Train_array, Train_array_response, l)
% This function trains a naive bayes classifier amd classifies a set of 
% data vectors in one out of c possible classes, according to the Naive 
% Bayes classifier, assuming normal probabilities.
% It also caclulates the success rate and the confusion matrix
%
% INPUT ARGUMENTS:
% Train_array: an lxN dimensional matrix whose columns are the data vectors
% to be used as training set
% Test_array: an lxN dimensional matrix whose columns are the data vectors 
% to be used as test set
% Train_array_response: an 1xN dimensional matrix whose columns are class
% of the i-th element
% Test_array_response: an 1xN dimensional matrix whose columns are class
% of the i-th element
% l: The dimensions of each chracterisic
%
% OUTPUT ARGUMENTS
% bayes_means: an lxc dimensional matrix, whose i,j cell corresponds to the
% mean of the probability of the j-th cahractersitic to be in the i-th 
% class.
% bayes_sv: an lxc dimensional matrix, whose i,j cell corresponds to the 
% standard deviation of probability of the j-th cahractersitic to be in the
% i-th class.
% a_priori: contains the a_priori probabilities for each class
% bayes_sucess_rate: The success rate of the classifier
% bayes_conf_matrix: The confusion matrix given for the current train and
% test set
%
% (c) 2019 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get the amount of the classes we have
%[~, classes] = size(unique(Train_array_response));
% Get the number of the points we have in train set
%[~, train_set_size]=size(Train_array_response);


% Make two arrays with number of classes*characteristics size. One with 
% the means and one with the standard deviations for each distribution. We
% have one distribution per class per characteristic. Initialize them to
% zeroes
bayes_means(1:classes, 1:l) = 0;
bayes_sv(1:classes, 1:l) = 0;

% Get all the distributions for each class
for i=1:classes
    cur_set=[];
    cur_set_count=1;    % Lines of the current set, one for each point.

    % Each column corresponds to each characteristic
    for j=1:train_set_size
        % For each element of Train array in the current class, add each
        % characteristic to the corresponding set so we can
        % calculate the parametters of the distribution
        if (Train_array_response(j) == i)
            for k=1:l
                cur_set(cur_set_count, k) = Train_array(k, j);
            end
            cur_set_count = cur_set_count + 1;
        end
    end
    % Calculate the distribution parameters for each cahracteristic for current class
    for j=1:l
        dist_res = mle(cur_set(:, j)');
        bayes_means(i, j) = dist_res(1);
        bayes_sv(i, j) = dist_res(2);
    end
end

% Calculate the a-priori probabilities
sums(1:classes)=0;
for i=1:train_set_size
    % Get the current class
    cur_class = Train_array_response(i);
    % Increase the amount of the points in our current class by 1
    sums(cur_class) = sums(cur_class) + 1;
end
a_priori = sums./train_set_size;

% Classify the test set based on the pareameters we just caclulated
out_bayes = naive_bayes_classifier(bayes_means, bayes_sv, a_priori,...
                                    Test_array, l, classes, test_set_size);
% Find the success rate and the confusion matrix of the classifed test set
[bayes_success_rate, bayes_conf_matrix] = calc_success_rate(out_bayes,...
                                    Test_array_response, classes, test_set_size);
