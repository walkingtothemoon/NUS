load('net-epoch-22.mat');
class=1;index=1;
subdir=dir(datadir);
imgfiles=dir(fullfile(datadir,subdir(class+4).name));
img=imread(fullfile(datadir,subdir(class+4).name,imgfiles(index+2).name));
imshow(img);
net=load(netpath);
net=net.net;
im_=single(img);
im_=imresize(im_,net.meta.inputSize(1:2));
im_=im_ - net.meta.normalization.averageImage;
opts.batchNormalization = false ;
net.layers{end}.type = 'softmax';
res=vl_simplenn(net,im_);
scores=squeeze(gather(res(end).x));
[bestScore,best]=max(scores);
str=[subdir(best+2).name ':' num2str(bestScore)];
title(str);