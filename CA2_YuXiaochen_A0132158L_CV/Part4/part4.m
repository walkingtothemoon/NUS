clc;
clear all;

disparity = 0.0001:0.0002:0.01;
N_class = 50;
num_frame = 10;
sigma = 10;
lambda = 0.01;

for frame_all = 1:141

files = dir('Road/src/*.jpg');
images = fullfile('Road', 'src' , {files.name});
[H, W, ~] = size(imread(images{1}));

for i=1:length(files)
    im{i} = reshape(permute(double(imread(images{i})),[2 1 3]),1,[],3);
end

c_matrix = textread('Road/cameras.txt','%s');
c_matrix = cellfun(@str2double, c_matrix);
total_frame = c_matrix(1);
c_matrix(1) = [];

for i = 1: total_frame
    index = (21*(i-1) +1):21*i;
    K{i} = reshape(c_matrix(index(1:9)),[3 3])';
    R{i} = reshape(c_matrix(index(10:18)),[3 3])';
    T{i} = reshape(c_matrix(index(19:21)),[1 3])';
end

N = H * W;
Unary_all = zeros(N_class, N);


ln = [repmat(1:W,1,H); reshape(repmat(1:H,W,1),1,N); ones(1,N)];
[~, idx] = sort(pdist2((1:total_frame)',(1:total_frame)'));
idx(1,:) = [];
    
%data term
next = idx(1:num_frame,frame_all); 
for x = 1:length(next)
    j = next(x);
    coord1 = K{j}*R{j}'*(T{frame_all}-T{j})*disparity;
    coord2 = K{j}*R{j}'*R{frame_all}*inv(K{frame_all})*ln;
    proj = [];
    for i=1:length(disparity)
        xtemp = coord2 + repmat(coord1(:,i),1,N);
        xtemp = round(xtemp(1:2,:) ./ xtemp(3,:));
        xtemp(2,xtemp(2,:) <1) = 1;
        xtemp(2,xtemp(2,:)>H) = H;
        xtemp(1,xtemp(1,:) <1) = 1;
        xtemp(1,find(xtemp(1,:)>W)) = W;
        proj = [proj; xtemp(1,:)+(xtemp(2,:)-1)*W;]; 
    end
    proj_img = im{j};
    proj_img = reshape(proj_img(1, proj, :), N_class, N, []);
    unary = sqrt(sum((proj_img - repmat(im{frame_all}, N_class, 1)).^2, 3)); 
    unary = sigma./(sigma + unary);
    Unary_all = unary + Unary_all;
end

Unary_all = 1- (Unary_all./repmat(max(Unary_all), N_class, 1));    
img = permute(double(imread(images{frame_all})), [2 1 3]);
[H, W] = size(img(:, :, 1));

% prior term
below = sparse(repmat([ones(1,H-1),0],1,W));
below = below(1:end-1);
right = sparse(ones(1,H * (W - 1)));
pairwise = diag(below,1) + diag(right,H);
pairwise = pairwise + pairwise';
addpath('../GCMex')

class = disparity(end) * ones(1, N);

% labelcost
labelcost = min(abs((meshgrid(1:N_class) - meshgrid(1:N_class)')),25) * lambda;

[labels, ~, ~] = GCMex(class, single(Unary_all), pairwise, single(labelcost));

depth_map = mat2gray(reshape(disparity(labels+1), H, W)');
imshow(depth_map);
f_name = [int2str(frame_all),'.png'];
imwrite(depth_map, f_name)

end