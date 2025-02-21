close all;

restoredefaultpath;    % 모든 사용자 추가 경로 초기화
addpath(genpath(pwd));  % 현재 폴더와 모든 하위 폴더 경로에 추가

%% Data Preparation
input_path = 'data/input/laser_cylinder';
cylRadius = 45;
data = load('stereoParams_0123.mat');
cameraParams = data.stereoParams_0123;

imgInfo = getUniqueName(input_path);
numImages = length(imgInfo);
angles = parseImgInfo(imgInfo);
angles = deg2rad(angles);

nAngles = size(angles, 1);
Pts3s = cell(nAngles, 1);
ln = cell(nAngles, 1);
Xcyl = cell(nAngles, 1);
Ycyl = cell(nAngles, 1);
Zcyl = cell(nAngles, 1);


%%
lengthCyl = 400;
progBar = ProgressBar(numImages);
for i = 1:numImages
    [ln{i}, Pts3s{i}] = getInfo3dCylinder(gridPtsP{i,1}, gridPtsP{i,2}, cameraParams, cylRadius);
    [Xcyl{i}, Ycyl{i}, Zcyl{i}] = getCylinder(ln{i}, Pts3s{i}, cylRadius, lengthCyl);
    progBar([], [], []);
end

progBar.release();

%%
imgName = {'-1-8'};
cylInfo = structureCylInfo(length(imgName), Xcyl, Ycyl, Zcyl);

% close all;

% Define colors for each cylinder in idx
colors = {'red', 'blue', 'green'};

% Pts3s만 그리기 (색상 적용)
plotCylinders3D(imgInfo, imgName, Pts3s, cylInfo, 'Pts3s', colors);

% Pts3s + 실린더 그리기 (색상 적용)
% plotCylinders3D(imgInfo, imgName, Pts3s, cylInfo, 'cylinder', colors);