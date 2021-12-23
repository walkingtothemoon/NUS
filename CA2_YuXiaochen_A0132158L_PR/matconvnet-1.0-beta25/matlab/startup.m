%将下面的路径设置为你的MatconvNet安装路径
run vl_setupnn;
datadir='PIE';
opts.expDir = fullfile(vl_rootnn, 'data', 'plate-baseline') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');