%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% [means] = calc_means(Train_array, classes)
% This function calculate the means of each class from a given training set
%
% INPUT ARGUMENTS:
% Train_array: an lxN dimensional matrix whose columns are the data vectors to
% be classified.
%classes: The ammount of classes
%
% OUTPUT ARGUMENTS
% res: an N-dimensional vector whose i-th component contains the label
% of the class where the i-th data vector has been assigned.
%
% (c) 2018 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [means] = calc_means(Train_array, Train_array_response, classes)


