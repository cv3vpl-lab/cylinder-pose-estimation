close all;

restoredefaultpath;    % 모든 사용자 추가 경로 초기화
addpath(genpath(pwd));  % 현재 폴더와 모든 하위 폴더 경로에 추가

%% setting path 
input_path = 'data/input/laser_cylinder';

% 환경이 같다면 최초 설정 후 불필요 (다른 사람것 써도 무방. 바꿀필요X) 
% dllPath = 'C:/Users/parkchaeho/Anaconda3/envs/grid/Library/bin'; % chaeho
% dllPath = 'C:/Users/choik/anaconda3/envs/CPE/Library/bin'; %kschoi
dllPath = 'C:/Users/kmb30/anaconda3/envs/AGV_py10/Library/bin'; %MB

utilsDir = 'utils'; % utils for makePyGridPts

moduleName = 'python_grid_detection_cylinder'; % name of python-makePyGridPts

data = load('data/stereoParams_0123.mat');
cameraParams = data.stereoParams_0123;

%% initialization & setting env 
imgInfo = getUniqueName(input_path); % get unique img name (ex. 00, 01, 02, ...)

[K1, K2, ~, ~, T_C2_C1, ~] = getCamParams(cameraParams);

angles = parseImgInfo(imgInfo); % get angle from imgInfo (ex. imgInfo: 01 => angle: pan=0, tilt=1)
angles = deg2rad(angles);

nAngles = size(angles, 1);

Pts3 = cell( nAngles, 1 );
cylT0 = cell( nAngles, 1 );
meanError = cell( nAngles, 1 );
fvals = cell( nAngles, 1 );

numImages = length(imgInfo);
gridPtsP = cell(numImages, 2);       
output_imgs = cell(numImages, 2);  
cylRadius = 45;

idx = struct('start', 1, 'end', 45);

pyEnv = struct(...
    'utilsDir', utilsDir, ...
    'dllPath', dllPath, ...
    'modulename', moduleName);

setPyenv(pyEnv.dllPath) % set python env

%% finding grid points
progBar = ProgressBar(numImages);
imgUndist = cell(numImages, 2);
imgHisteq = cell(numImages, 2);

for i = 1:numImages
    img_info = imgInfo{i};

    I1 = imread( fullfile( input_path, sprintf('%sL.png', imgInfo{i}) ) );
    I2 = imread( fullfile( input_path, sprintf('%sR.png', imgInfo{i}) ) );

    [I1_undist, I2_undist, I1_histeq, I2_histeq] = preProcessing(I1, I2, cameraParams); % img pre-processing 

    imgUndist(i,:) = {I1_undist, I2_undist};  % undistortion img
    imgHisteq(i,:) = {I1_histeq, I2_histeq};  % undistortion + histeq

    try
        [gridPtsP{i,1}, output_imgs{i,1}] = makePyGridPts(imgUndist{i,1}, pyEnv);
        [gridPtsP{i,2}, output_imgs{i,2}] = makePyGridPts(imgUndist{i,2}, pyEnv);
    catch ME
        warning('makePyGridPts error: %s', ME.message);
    end
    progBar([], [], []);
end

progBar.release();

%% Single-cylinder fitting
for i = idx.start:idx.end 
    fignum = i*2 + [-1, 0];
    [Pts3{i}, cylT0{i}, fvals{i}, meanError{i}] = fitSingleCylinder( i, gridPtsP(i,:), cylRadius, imgHisteq(i,:), fignum, imgInfo{i}, cameraParams, false );
end
%%
drawFvals(fvals)
drawFvals(meanError)

%% Multi-cylinder fitting 
[T_Cam_AGV, fval] = fitCylinderWPts3sAngs( Pts3(idx.start:idx.end,:), mat2cell(angles(idx.start:idx.end,:), ones(1, length(idx.start:idx.end)), 2), cylRadius ); 

Clrs = [1, 1, 0; 1, 0, 1];
for i = idx.start:idx.end 
    T_Cam_cyl = T_Cam_AGV * getTAGVcyl( angles(i, 1), angles(i, 2));
    figure(i*2-1), drawCylinder( T2vec(T_Cam_cyl), cylRadius, K1, Clrs )
    figure(i*2), drawCylinder( T2vec(T_C2_C1 * T_Cam_cyl), cylRadius, K2, Clrs )
end
