function [res]=euclidean_classifier(means, data_array, classes, data_size)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
% [res]=euclidean_classifier(means,Train_array)
% This function classifies a set of data vectors in one out of c possible
% classes, according to the Euclidean classifier.
%
% INPUT ARGUMENTS:
% means: an lxc dimensional matrix, whose i-th column corresponds to the
% mean of the i-th class.
% data_array: an lxN dimensional matrix whose columns are the data vectors 
% to be classified.
%
% OUTPUT ARGUMENTS
% res: an N-dimensional vector whose i-th component contains the label
% of the class where the i-th data vector has been assigned.
%
% (c) 2019 V. Spithas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:data_size
    for j=1:classes
       vects_to_calc = [(data_array(:,i))'; means(j,:)];
       dist(j) = pdist(vects_to_calc, 'euclidean');
    end
    [~, res(i)] = min(dist);
end
