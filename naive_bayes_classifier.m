function [res] = naive_bayes_classifier(bayes_means, bayes_sv, a_priori,...
                                     Test_array, l, classes, test_set_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% [res] = naive_bayes_classifier(bayes_means, bayes_var, Test_array)
% This function classifies a set of data vectors in one out of c possible
% classes, according to the Naive bayes classifier.
%
% INPUT ARGUMENTS:
% bayes_means: an lxc dimensional matrix, whose i,j cell corresponds to the
% mean of the probability of the j-th cahractersitic to be in the i-th 
% class.
% bayes_sv: an lxc dimensional matrix, whose i,j cell corresponds to the 
% standard deviation of probability of the j-th cahractersitic to be in the
% i-th class.
% a_priori: contains the a_priori probabilities for each class
% l: The dimension of each chracterisic
%
% OUTPUT ARGUMENTS
% res: an N-dimensional vector whose i-th component contains the label
% of the class where the i-th data vector has been assigned.
%
% (c) 2019 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

res(1:test_set_size)=0;

% Classify each point based on the Gaussian distributons and using the
% means and standard deviations we have gotten as arguments
for i=1:test_set_size
    prob(1:classes)=0;  % Probability to belong to each one of the classes
    
    % Calculate the probabilty of each class
    for j=1:classes   
        for k=1:l
            prob(j) = prob(j) +...
            log( (1/sqrt(2*pi*(bayes_sv(j, k)^2))) *...
            exp(-(Test_array(k,i) - bayes_means(j, k))^2/...
            (2*(bayes_sv(j, k)^2))));
        end
    end

    % Multiply the probabilities of the Gaussian distribution to the 
    % a-priory probability of each class
    prob = prob.*a_priori;

    % The point will belong to the class with the highest probability
    [~, res(i)] = max(prob);
end