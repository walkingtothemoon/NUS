addpath '../GCMex'

% load images
img2 = imread('im2.png');
img6 = imread('im6.png');

% get the image size
[H, W, ~] = size(img2);

img2 = double(img2)/255;
img6 = double(img6)/255;

img2 = reshape(img2,[],3);
img6 = reshape(img6,[],3);

N_classes = 50;
class = zeros(1, H * W);
unary = ones(N_classes, H * W);

% data term
for D = 1:N_classes 
    img2_new = img2((H * D+1):end,:);
    img6_new = img6(1:(end - H * D),:);
    unary(D,(H * D + 1):end) = (sum(abs(img2_new - img6_new),2))';
end

lambda = 0.02;
labelcost = min(abs((meshgrid(1:N_classes) - meshgrid(1:N_classes)')),10) * lambda;

% prior term
below = sparse(repmat([ones(1,H-1),0],1,W));
below = below(1:end-1);
right = sparse(ones(1,H * (W - 1)));
pairwise = diag(below,1) + diag(right,H);
pairwise = pairwise + pairwise';

[labels, ~, ~] = GCMex(class, single(unary), pairwise, single(labelcost));

labels = reshape(labels, H, W);

img_new = uint8(labels * 255/(N_classes-1));
figure();
imshow(img_new);
title(['lambda = ', num2str(lambda)]);