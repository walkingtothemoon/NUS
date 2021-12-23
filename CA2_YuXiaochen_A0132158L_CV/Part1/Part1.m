clc;
clear all;

addpath '../GCMex'

img_o = imread('bayes_in.jpg');
img_s = double(reshape(img_o,[],3));
[H, W, ~] = size(img_o);

source = [0, 0, 255]; % blue foreground
sink = [245, 210, 110]; % yellow background

class = zeros(1, H * W);
unary = zeros(2, H * W);

% change this value to change the weight of the smoothness or prior term
lambda = 200;
labelcost = [0 1; 1 0] * lambda;

% data term
for i = 1 : H * W
    unary(1, i) = dist(source, img_s(i,:));
    unary(2, i) = dist(sink, img_s(i,:));
    if unary(1,i) > unary(2,i)
        class(1,i) = 1;     
    end
end

% prior term
below = sparse(repmat([ones(1,H-1),0],1,W));
below = below(1:end-1);
right = sparse(ones(1,H * (W - 1)));
pairwise = diag(below,1) + diag(right,H);
pairwise = pairwise + pairwise';


[labels, ~, ~] = GCMex(class, single(unary), pairwise, single(labelcost));

img_new = zeros(H * W, 3);
for i = 1 : H * W
    if labels(i) == 0
        img_new(i,:) = source;
    else
        img_new(i,:) = sink;
    end
end
img_new = uint8(reshape(img_new,[H,W,3]));
imshow(img_new);
title('lambda=200');

function distance = dist(pixel1 , pixel2)
distance = (abs(pixel1(1) - pixel2(1)) + abs(pixel1(2) - pixel2(2)) + abs(pixel1(3) - pixel2(3))) / 3;
end