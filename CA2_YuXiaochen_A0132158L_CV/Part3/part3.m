addpath('../GCMex');

% load images
img1 = imread('test00.jpeg');
img2 = imread('test09.jpeg');

[H, W, ~] = size(img1);

img1 = double(img1)/255;
img2 = double(img2)/255;

img1 = reshape(img1,[],3);
img2 = reshape(img2,[],3);

c_matrix = readmatrix('cameras.txt');
K1 = c_matrix(1:3, 1:3);
R1 = c_matrix(4:6, 1:3);
T1 = c_matrix(7:7, 1:3);
K2 = c_matrix(8:10, 1:3);
R2 = c_matrix(11:13, 1:3);
T2 = c_matrix(end, 1:3);

N_classes = 50;
class = zeros(1, H * W);
unary = ones(N_classes, H * W);
disparity = 0.0001:0.0002:0.01;

[X,Y] = meshgrid(1:W,1:H);
term1 = [X(:),Y(:),ones(H*W,1)]';

i = 0;
for  d = disparity
    i = i+1;
    term2 = K2 * R2' * R1 / K1 * term1 + d * K2 * R2' * (T1 - T2)';
    term2 = round(term2(1:2,:) ./ term2(3,:));
    v_l = all(term2(1,:) <= W & term2(2,:) <= H & term2(1:2,:) > 0); 
    idx = sub2ind([H, W], term2(2,v_l), term2(1,v_l));
    unary(i, v_l) = (sum(abs(img1(v_l,:) - img2(idx,:)),2))';
end

lambda = 0.01;
labelcost = min(abs((meshgrid(1:N_classes) - meshgrid(1:N_classes)')),5) * lambda;

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