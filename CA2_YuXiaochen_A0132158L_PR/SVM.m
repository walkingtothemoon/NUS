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

train_diff = [];
test_diff = [];
train_mean = mean(TrainSet,2);
train_diff = TrainSet - repmat(train_mean,1,size(TrainSet,2));
test_diff = TestSet - repmat(train_mean,1,size(TestSet,2));

%generate S matrix
s_matrix = (train_diff * train_diff') / size(TrainSet,2);
[U, D, V] = svd(s_matrix);

train_80d = U(:,1:80)' * train_diff;
train_200d = U(:,1:200)' * train_diff;
test_80d = U(:,1:80)' * test_diff;
test_200d = U(:,1:200)' * test_diff;

addpath("libsvm-3.25/matlab");

ptrain_80 = mapminmax(train_80d',0,1);
ptest_80 = mapminmax(test_80d',0,1);
ptrain_200 = mapminmax(train_200d',0,1);
ptest_200 = mapminmax(test_200d',0,1);


c = [0.01,0.1,1];

model_linear_80 = svmtrain(TrainLabel,ptrain_80,'-t 0 -c 1');
model_linear_200 = svmtrain(TrainLabel,ptrain_200,'-t 0 -c 1');
[label_80, accuracy_80, values_80] = svmpredict(TestLabel,ptest_80,model_linear_80);
[label_200, accuracy_200, values_200] = svmpredict(TestLabel,ptest_200,model_linear_200);

