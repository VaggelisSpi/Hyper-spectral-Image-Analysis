% This is a supporting MATLAB file for the project

clear
format compact
close all

load Salinas_hyperspectral % Load the Salinas hypercube called "Salinas_Image"
% p,n define the spatial resolution of the image, while l is the number of 
% bands (number of features for each pixel)
[p,n,l] = size(Salinas_Image);

load classification_labels 
% This file contains three arrays of dimension 22500x1 each, called
% "Training_Set", "Operational_Set" and "Test_array_Set". In order to bring them
% in an 150x150 image format we use the command "reshape" as follows:
Training_Set_Image=reshape(Training_Set, p,n); % In our case p=n=150 (spatial dimensions of the Salinas image).
Test_Set_Image = reshape(Test_Set, p,n);
Operational_Set_Image = reshape(Operational_Set, p,n);

%Depicting the various bands of the Salinas image
for i=1:l
    %figure(1), imagesc(Salinas_Image(:,:,i))
    pause(0.05) % This command freezes figure(1) for 0.05sec. 
end

% Depicting the training, test and operational sets of pixels (for the
% pixels depicted with a dark blue color, the class label is not known.
% Each one of the other colors in the following figures indicate a class).
%figure(2), imagesc(Training_Set_Image)
%figure(3), imagesc(Test_Set_Image)
%figure(4), imagesc(Operational_Set_Image)

%%%%%%%%%%%% Make Train set %%%%%%%%%%%%
% Constructing the 204xN array whose columns are the vectors corresponding to the
% N vectors (pixels) of the training set (similar codes can be used for
% the test and the operational sets).
Train=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     % Multiply elementwise each band of the Salinas_Image with the mask 
     % "Training_Set_Image>0", which identifies only the training vectors.
    Train(:,:,i)=Salinas_Image(:,:,i).*(Training_Set_Image>0);
    %figure(5), imagesc(Train(:,:,i)) % Depict the training set per band
    pause(0.05)
end

Train_array=[]; %This is the wanted 204xN array
Train_array_response=[]; % This vector keeps the label of each of the training pixels
Train_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Training_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Train_array = [Train_array squeeze(Train(i,j,:))];
            Train_array_response = [Train_array_response Training_Set_Image(i,j)];
            Train_array_pos = [Train_array_pos; i j];
        end
    end
end

%%%%%%%%%%%% Make Test set %%%%%%%%%%%%
% Constructing the 204xN array whose columns are the vectors corresponding to the
% N vectors (pixels) of the test set (similar codes cane be used for
% the test and the operational sets).
Test=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     %Multiply elementwise each band of the Salinas_Image with the mask
     % "Test_Set_Image>0", which identifies only the training vectors.
    Test(:,:,i)=Salinas_Image(:,:,i).*(Test_Set_Image>0);
    %figure(6), imagesc(est(:,:,i)) % Depict the training set per band
    pause(0.05)
end

Test_array=[]; %This is the wanted 204xN array
Test_array_response=[]; % This vector keeps the label of each of the training pixels
Test_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Test_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Test_array = [Test_array squeeze(Test(i,j,:))];
            Test_array_response = [Test_array_response Test_Set_Image(i,j)];
            Test_array_pos = [Test_array_pos; i j];
        end
    end
end

%%%%%%%%%%%% Make Operational set %%%%%%%%%%%%
% Constructing the 204xN array whose columns are the vectors corresponding to the
% N vectors (pixels) of the test set (similar codes cane be used for
% the test and the operational sets).
Operational=zeros(p,n,l); % This is a 3-dim array, which will contain nonzero values only for the training pixels
for i=1:l
     %Multiply elementwise each band of the Salinas_Image with the mask
     % "Operational_Set_Image>0", which identifies only the training vectors.
    Operational(:,:,i)=Salinas_Image(:,:,i).*(Operational_Set_Image>0);
    %figure(7), imagesc(est(:,:,i)) % Depict the training set per band
    pause(0.05)
end

Operational_array=[]; %This is the wanted 204xN array
Operational_array_response=[]; % This vector keeps the label of each of the training pixels
Operational_array_pos=[]; % This array keeps (in its rows) the position of the training pixels in the image.
for i=1:p
    for j=1:n
        if(Operational_Set_Image(i,j)>0) %Check if the (i,j) pixel is a training pixel
            Operational_array = [Operational_array squeeze(Train(i,j,:))];
            Operational_array_response = [Operational_array_response Operational_Set_Image(i,j)];
            Operational_array_pos = [Operational_array_pos; i j];
        end
    end
end

% Get the amount of the classes we have
[~, classes] = size(unique(Train_array_response));
% Get the number of the points we have in train set
[~, train_set_size] = size(Train_array_response);
% Get the number of the points we have in test set
[~, test_set_size] = size(Test_array_response);

%%%%%%%%%%%%%% Train the classifiers and test them %%%%%%%%%%%%%%%%%
% Euclidean Classification
[means, eucl_success_rate, eucl_conf_matrix] = euclidean_trainer(...
    Train_array, Test_array, Train_array_response, Test_array_response, l,...
    classes, train_set_size, test_set_size);


% Naive Bayes Clasification
[bayes_means, bayes_sv, a_priori, bayes_success_rate, bayes_conf_matrix] =...
       naive_bayes_trainer(Train_array, Test_array, Train_array_response,...
       Test_array_response, l, classes, train_set_size, test_set_size);
%[bayes_means, bayes_sv, out_bayes] = naive_bayes_trainer(Train_array, Test_array, Train_array_response, l);
%[bayes_success_rate, bayes_conf_matrix] = calc_success_rate(out_bayes, Train_array_response);

% kNN Classification
% best_k = kNN_trainer(Train_array, Train_array_response, l, classes,...
%                   train_set_size);
% 17 was the best value returned from above. Comment this line and 
% uncomment the one above to check it
best_k = 5; 
% k is used hardcoded because the above call takes a lot of time to calculate
out_kNN = kNN_classifier(Train_array,...
                    Test_array, Train_array_response,...
                    l, classes, train_set_size, test_set_size, best_k);

[kNN_success_rate, kNN_conf_matrix] = calc_success_rate(out_kNN,...
                       Test_array_response, classes, test_set_size);

%%%%%%%%%%%%%% Classify the operational set %%%%%%%%%%%%%%%%%
% res_eucl = euclidean_classifier(means, Operational_array);

% res_bayes = naive_bayes_classifier(bayes_means, bayes_sv, a_priori,...
                                    % Operational_array, l);

% res_kNN = kNN_classifier(Train_array, Operational_array,...
                          % Train_array_response, l, classes, best_k);