function mapAll2FER()
%ccc
imgdir = ...
    'E:\emotiw2015\Datasets\histeq\Emotiw_Test_aligned_faces_histeq\'; %change me%
savedir = ...
    'E:\emotiw2015\Datasets\registered48x48\Emotiw15_Test_reg\'; %change me%
mkdir(savedir);
try
    rmdir(savedir,'s'); 
    pause(1);
    mkdir(savedir);
catch
end
FERavg = 'FER_48x48_points_avg.mat';
load(FERavg); base_points = pavg;

OTHERavg = 'EMOTI15_TRAIN_128x128_points_avg.mat'; %change me%
load(OTHERavg); input_points = pavg+48; %add border of 48 for EMOTIW (32 is enough for TFD which results in 160 image size)

TFORM = cp2tform(input_points, base_points, 'similarity')

imagefiles = dir([imgdir '*.jpg']); %change me%
for i=1:length(imagefiles)
    i
    noisyim = uint8(randi([0,255],[224,224])); %if TFD second argument is [160,160]
    img = imread([imgdir imagefiles(i).name]);
    if size(img,3)==3
        img = rgb2gray(img);
    end
    x=(size(noisyim,1)-size(img,1))/2;
    y=(size(noisyim,2)-size(img,2))/2;
    noisyim(x+1:x+size(img,1),y+1:y+size(img,2))=...
        img;
    
    [imgtransformed,xdata,ydata] = imtransform(noisyim,TFORM,...
        'XData',[1 48], 'YData',[1 48]);
    %imshow(imgtransformed)
    imwrite(imgtransformed,[savedir imagefiles(i).name(1:end-4) '.png'],'png')
end
%verify
FERavg = 'FER_48x48_points_avg.mat';
load(FERavg); base_points = pavg;

imagefiles = dir([savedir '*.png']);
for i=1:10:length(imagefiles)
    i
    img = imread([savedir imagefiles(i).name]);
    clf;
    imshow(img)
    hold on;
    plot(base_points(:,1),base_points(:,2),'.r');
    pause(0.0001)
end
end
