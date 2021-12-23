clc;
clear all;
close all;

%prepare trainset & testset
database = [pwd '/PIE/'];
%TrainSize = 20;
%TestSize = 9;
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
SelfieLabel = repmat(26,10,1);
for k = 1:10
    selfie = double(imread(strcat(database,num2str(0),'/',num2str(k),'.jpg')));
    SelfieSamples = selfie(1:32*32);
    SelfieSamples = double(SelfieSamples);
    SelfieSet = [SelfieSet;SelfieSamples];
end

TrainSet = [TrainSet;SelfieSet(1:7,:)];
TrainLabel = [TrainLabel;SelfieLabel(1:7,:)];

miu = mean(TrainSet)';
miu_i = labelmean(TrainSet',TrainLabel');

sb = size(1024,1024);
sw = size(1024,1024);
for i = 1 : 26
    sb = sb + (miu_i(:,i) - miu)*(miu_i(:,i) - miu)';
    [row, column,~]=find(TrainLabel == i);
    sw = sw + cov(TrainSet(row,:))*length(row)/length(TrainLabel);
end

[EigVector, EigValue]=eig((pinv(sw)*sb));
[~,EigValue_idx]=sort(diag(EigValue),'descend');

W{1} = EigVector(:,EigValue_idx(1:2));
W{2} = EigVector(:,EigValue_idx(1:3));
W{3} = EigVector(:,EigValue_idx(1:9));

train_2d = TrainSet* W{1};
train_3d = TrainSet* W{2};

%LDA Visualization
figure(1);
for n = 1:length(TrainLabel)
    if(TrainLabel(n) == 26)
        scatter(train_2d(n,1),train_2d(n,2),'r*');
        hold on;
    else
        scatter(train_2d(n,1),train_2d(n,2));
        hold on;
    end
end

hold off

figure(2);
for n = 1:length(TrainLabel)
    if(TrainLabel(n) == 26)
        scatter3(train_3d(n,1),train_3d(n,2),train_3d(n,3),'r*');
        hold on;
    else
        scatter3(train_3d(n,1),train_3d(n,2),train_3d(n,3));
        hold on;
    end
end
hold off;


function out = labelmean(data,label)
out=[];
for i=unique(label)
    out = [out, mean(data(:,label==i),2)];
end 
end
