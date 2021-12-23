clc;
clear all;
close all;

%prepare trainset & testset
database = [pwd '/PIE/'];
ClassNum = 25;
n = randperm(170,29);

TrainSet = [];
TestSet = [];
TrainLabel = [];
TestLabel = [];
for i = 1:ClassNum
    for j = 1:29
        if j <= 20
            trainsamples = double(imread(strcat(database,num2str(i),'/',num2str(n(j)),'.jpg')));            
            TrainSamples = trainsamples(1:32*32);
            TrainSamples = double(TrainSamples);
            TrainSet = [TrainSet;TrainSamples];
            TrainLabel = [TrainLabel;i];
        else        
            testsamples = double(imread(strcat(database,num2str(i),'/',num2str(n(j)),'.jpg')));
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
TrainLabel = [TrainLabel;SelfieLabel(1:7,:)]';
TestSet = [TestSet;SelfieSet(8:10,:)]';
TestLabel = [TestLabel;SelfieLabel(8:10,:)]';

%PCA Visualization
train_diff = [];
train_mean = mean(TrainSet,2);
train_diff = TrainSet - repmat(train_mean,1,size(TrainSet,2));

%generate S matrix
s_matrix = (train_diff * train_diff') / size(TrainSet,2);
[U, D, V] = svd(s_matrix);

train_2d = U(:,1:2)' * train_diff;
train_3d = U(:,1:3)' * train_diff;


figure(1);
for n = 1:size(TrainSet,2)
    if(TrainLabel(n) == 0)
        scatter(train_2d(1,n),train_2d(2,n),'r*');
        hold on;
    else
        scatter(train_2d(1,n),train_2d(2,n));
        hold on;
    end
end

hold off

figure(2);
for n = 1:size(TrainSet,2)
    if(TrainLabel(n) == 0)
        scatter3(train_3d(1,n),train_3d(2,n),train_3d(3,n),'r*');
        hold on;
    else
        scatter3(train_3d(1,n),train_3d(2,n),train_3d(3,n));
        hold on;
    end
end
hold off;

figure(3);title('eigenface-40');
hold on;
eigenface_40d = U(:,40)*U(:,40)'*train_diff(:,40)+train_mean;
eigenface_40d = reshape(eigenface_40d,[32,32]);
imshow (mat2gray(eigenface_40d))
hold off;

figure(4);title('eigenface—80');
hold on;
eigenface_80d = U(:,80)*U(:,80)'*train_diff(:,80)+train_mean;
eigenface_80d = reshape(eigenface_80d,[32,32]);
imshow (mat2gray(eigenface_80d))
hold off;


figure(5);title('eigenface-200');
hold on;
eigenface_200d = U(:,200)*U(:,200)'*train_diff(:,200)+train_mean;
eigenface_200d = reshape(eigenface_200d,[32,32]);
imshow (mat2gray(eigenface_200d))
hold off;


