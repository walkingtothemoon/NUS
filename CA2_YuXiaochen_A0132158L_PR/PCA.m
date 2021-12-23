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
[U, D, V] = svd(s_matrix);

%reduce dimension
d = 200;
train_d = U(:,1:d)' * train_diff;
test_d = U(:,1:d)' * test_diff;


distance = zeros(size(TestSet,2),size(TrainSet,2));
match = 0;
s = 0;
for i = 1:size(TestSet,2)
    for j = 1: size(TrainSet,2)
        for k = 1:d
        distance(i,j)=distance(i,j)+ sum(abs(test_d(k,i)-train_d(k,j)));
        end
    end
end

for i = 1:length(TestLabel)
    [~,label_index(i)] = min(distance(i,:));
    if i <= length(TestLabel)-3
        if(TestLabel(i) == TrainLabel(label_index(i)))
            match = match + 1;
        end
    else
        if(TestLabel(i) == TrainLabel(label_index(i)))
             s = s+1;
        end
    end
end

accuracy_PIE = match/(length(TestSet)-3);
accuracy_Selfie = s/3;
