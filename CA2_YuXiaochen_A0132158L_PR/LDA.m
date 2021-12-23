%prepare trainset & testset
database = [pwd '/PIE/'];
ClassNum = 25;

TrainSet = [];
TestSet = [];
TrainLabel = [];
TestLabel = [];
for i = 1:ClassNum
    for x = 1:170
        if x <= 119
            trainsamples = double(imread(strcat(database,num2str(i),'/',num2str(x),'.jpg')));            
            TrainSamples = trainsamples(1:32*32);
            TrainSamples = double(TrainSamples);
            TrainSet = [TrainSet;TrainSamples];
            TrainLabel = [TrainLabel;i];
        else        
            testsamples = double(imread(strcat(database,num2str(i),'/',num2str(x),'.jpg')));
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
TestSet = [TestSet;SelfieSet(8:10,:)];
TestLabel = [TestLabel;SelfieLabel(8:10,:)];


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


for i =1:3
    match = 0;
    s = 0;
    [idx, ~]= knnsearch(TrainSet * W{i},TestSet * W{i});
    for x = 1:length(TestLabel)
        if x <= length(TestLabel)-3
            if TestLabel(x,1) == TrainLabel(idx(x,1))
             match = match+1;
            end
        else
            if TestLabel(x,1) == TrainLabel(idx(x,1))
             s = s+1;
            end
        end
    end
    accuracy_PIE(i) = match/(length(TestSet)-3);
	accuracy_Selfie(i) = s/3;
end

function out = labelmean(data,label)
out=[];
for i=unique(label)
    out = [out, mean(data(:,label==i),2)];
end 
end

