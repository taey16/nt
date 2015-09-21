%% vgg / caffe spec

PJT_ROOT = '/works/neuraltalk/';
CAFFE_ROOT = '/works/caffe/';
addpath(genpath([CAFFE_ROOT 'matlab/']));
INPUT_IMG_LIST_FILENAME = 'tasks.txt';
MODEL_DEPLOY_FILENAME = sprintf('%s/models/vgg/vgg_layer16_deploy_feature.prototxt', CAFFE_ROOT);
MODEL_WEIGHT = sprintf('/%s/models/vgg/vgg_layer16.caffemodel', CAFFE_ROOT);
batch_size = 10;

%% set net
caffe.set_mode_cpu();
net = caffe.Net(MODEL_DEPLOY_FILENAME, MODEL_WEIGHT, 'test')

%% input files spec
root_path = sprintf('%s/example_images/', PJT_ROOT);
fs = textread([root_path INPUT_IMG_LIST_FILENAME], '%s');
N = length(fs);

% iterate over the iamges in batches
feats = zeros(4096, N, 'single');
for b=1:batch_size:N
    % enter images, and dont go out of bounds
    Is = {};
    for i = b:min(N,b+batch_size-1)
        I = imread([root_path fs{i}]);
        if ndims(I) == 2
            I = cat(3, I, I, I); % handle grayscale edge case. Annoying!
        end
        Is{end+1} = I;
    end
    input_data = prepare_images_batch(Is);

    tic;
    scores = net.forward({input_data});
    scores = squeeze(scores{1});
    tt = toc;

    nb = length(Is);
    feats(:, b:b+nb-1) = scores(:,1:nb);
    fprintf('%d/%d = %.2f%% done in %.2fs\n', b, N, 100*(b-1)/N, tt);
end

%% write to file

save([root_path 'vgg_feats_hdf5.mat'], 'feats', '-v7.3');
save([root_path 'vgg_feats.mat'], 'feats');
