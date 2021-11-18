metrics = {'psnr'; 'ssim'; 'multissim'; 'brisque'; 'pique'; 'niqe'};
qualmetrics = @(img) [brisque(img); piqe(img)];

lrpath = 'LR_high_SNR';
hrpath = 'HR_high_SNR';
restpath = 'gans_high_SNR_ENH';

liver1fnames = sort({dir("LR_high_SNR/brain*").name});

% 400 steps per epoch
qscores100_SR = zeros(length(liver1fnames), 2, 3); % liverXpeakY, metrics, category
for s=1:length(liver1fnames)
    fname=liver1fnames{s};
    if mod(s,100) == 0 
    fprintf("Peak [%d/%d] %.2f%%\t", s, length(liver1fnames), 100*s/length(liver1fnames))
    end
    lr = imread( strjoin({lrpath; fname}, "/") ); 
    LR = double(lr - min(lr(:))); LR = LR ./ max(LR(:));
    rest = imread( strjoin({restpath; fname}, "/") ); 
    REST = double(rest - min(rest(:))); REST = REST ./ max(REST(:));
    hr = imread( strjoin({hrpath; fname}, "/") ); 
    HR = double(hr-min(hr(:))); HR = HR ./ max(HR(:));
    
    qscores100_SR(s, :, 1) = qualmetrics(LR);
    qscores100_SR(s, :, 2) = qualmetrics(REST);
    qscores100_SR(s, :, 3) = qualmetrics(HR);
end

save('brain_ENH_gans.mat', 'qscores100_SR')