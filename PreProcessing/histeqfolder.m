%ccc
originaldir = ...
    'E:/emotiw2015/Datasets/EmotiW_2015/Test_aligned_faces/';
destinationdir = ...
    'E:/emotiw2015/Datasets/histeq/Emotiw_Test_aligned_faces_histeq/';

mkdir(destinationdir)

allimages = dir([originaldir '*.jpg']);
for i=1:length(allimages)
    allimages(i).name
    originalimg = [originaldir allimages(i).name];
    img=imread(originalimg);
    img=histeq(img);
    imwrite(img,[destinationdir allimages(i).name]);
end