clc;
clear all;
close all;

%prepare trainset & testset
database = [pwd '/PIE/'];
ClassNum = 25;

TrainSet = [];
TestSet = [];
TrainLabel = [];
TestLabel = [];
for i = 1:ClassNum
    for j = 1:170
        if j <= 119
            trainsamples = double(imread(strcat(database,num2str(i),'/',num2str(j),'.jpg')));            
            TrainSamples = trainsamples(1:32*32);
            TrainSamples = double(TrainSamples);
            TrainSet = [TrainSet;TrainSamples];
            TrainLabel = [TrainLabel;i];
        else        
            testsamples = double(imread(strcat(database,num2str(i),'/',num2str(j),'.jpg')));
            TestSamples = testsamples(1:32*32);
            TestSamples = double(TestSamples);
            TestSet = [TestSet;TestSamples];
            TestLabel = [TestLabel;i];
        end
    end
end

SelfieSet = [];
SelfieLabel = zeros(10,1);
for k = 1:10
    selfie = double(imread(strcat(database,num2str(0),'/',num2str(k),'.jpg')));
    SelfieSamples = selfie(1:32*32);
    SelfieSamples = double(SelfieSamples);
    SelfieSet = [SelfieSet;SelfieSamples];
end

TrainSet = [TrainSet;SelfieSet(1:7,:)]';
TrainLabel = [TrainLabel;SelfieLabel(1:7,:)];
TestSet = [TestSet;SelfieSet(8:10,:)]';
TestLabel = [TestLabel;SelfieLabel(8:10,:)];

%PCA
train_diff = [];
test_diff = [];
train_mean = mean(TrainSet,2);
train_diff = TrainSet - repmat(train_mean,1,size(TrainSet,2));
test_diff = TestSet - repmat(train_mean,1,size(TestSet,2));

%generate S matrix
s_matrix = (train_diff * train_diff') / size(TrainSet,2);
[U, ~, ~] = svd(s_matrix);

%reduce dimension
d = 80;
train_d = U(:,1:d)' * train_diff;
test_d = U(:,1:d)' * test_diff;

[label, NegativeLogLikelihood, NumIterations]=gmm(train_d', 3);

figure()
scatter(train_d(1,label==1),train_d(2,label==1),10,'r+');
hold on
scatter(train_d(1,label==2),train_d(2,label==2),10,'yo');
hold on
scatter(train_d(1,label==3),train_d(2,label==3),10,'c*');
title('result');


function [label, NegativeLogLikelihood, NumIterations]=gmm(X, K)
[N,D]=size(X);
para_sigma_inv=zeros(D, D, K);
N_pdf=zeros(N, K); 
RegularizationValue=0.001; 
MaxIter=100; 
TolFun=1e-8;   
gmm=fitgmdist(X, K, 'RegularizationValue', RegularizationValue, 'CovarianceType', 'diagonal', 'Start', 'plus', 'Options', statset('Display', 'final', 'MaxIter', MaxIter, 'TolFun', TolFun));
NegativeLogLikelihood=gmm.NegativeLogLikelihood;
NumIterations=gmm.NumIterations; 
mu=gmm.mu;  
Sigma=gmm.Sigma;  
ComponentProportion=gmm.ComponentProportion;  
for k=1:K
    sigma_inv=1./Sigma(:,:,k); 
    para_sigma_inv(:, :, k)=diag(sigma_inv);  
end
for k=1:K
    coefficient=(2*pi)^(-D/2)*sqrt(det(para_sigma_inv(:, :, k))); 
    X_miu=X-repmat(mu(k,:), N, 1);  
    N_pdf(:,k)=coefficient*exp(-(sum((X_miu*para_sigma_inv(:, :, k)).*X_miu,2)/2));
end
rep=N_pdf.*repmat(ComponentProportion,N,1); 
rep=rep./repmat(sum(rep,2),1,K); 
[~,label]=max(rep,[],2);
end
