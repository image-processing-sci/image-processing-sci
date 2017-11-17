function [ tform ] = transformation()
img = imread('betterCheckb.png');
gray = rgb2gray(img);
imshow(img)
% got these by taking the points of the photoshopped in checkerboard
% from = [707 428;810 435; 939 444;636 467;752 481;905 499;531 524;660 552;846 593];

% xCol yCol
myFrom = [1042 406; 1299 406; 1172 529; 903 791; 356 790; 822 529]
sL = 40; % square side length
tW = 2000;
tH = 2000;

myTo = [tW/2-3*sL, tH/2-20*sL;
 tW/2+3*sL, tH/2-20*sL;
 tW/2+3*sL, tH/2;
 tW/2+3*sL, tH/2+20*sL;
 tW/2-3*sL, tH/2+20*sL;
  tW/2-3*sL, tH/2]

% in my top view, I want those points of the checkerboard to be transformed
% to these pixels near the center of a ~1000x1000 image.
% to = [450;450;450;500;500;500;550;550;550]; %x coord
% x = [450;500;550;450;500;550;450;500;550]; %y coord
%
% to = [x,to];
outputView = imref2d([tW,tH]);
[tform, inlierPtsDistorted, inlierPtsOriginal] = estimateGeometricTransform(myFrom,myTo,'projective');
display(tform)
Ir = imwarp(gray,tform,'OutputView',outputView);
figure;imshow(Ir)
end

