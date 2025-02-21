function vars = initializeVar(input_path, cameraParams, numImages, imgInfo, utilsDir, dllPath, moduleName)
% initializeVar - 입력 인수를 기반으로 모든 관련 변수를 하나의 구조체(vars) 내에 초기화합니다.
%
% 입력:
%   input_path   : 이미지 파일 또는 폴더 경로 (예: 'C:\images')
%   cameraParams : 스테레오 카메라 파라미터 (stereoParameters 객체 또는 관련 데이터)
%   numImages    : 이미지 개수 (정수)
%   imgInfo      : 고유 이미지 이름을 담은 cell array (예: {'00', '01', ...})
%   utilsDir     : 파이썬 유틸리티 디렉토리 경로
%   dllPath      : DLL 파일 경로
%   moduleName   : 파이썬 모듈 이름
%
% 출력:
%   vars: 후속 처리에 필요한 모든 변수를 포함하는 구조체로, 주요 필드는 다음과 같습니다.
%         .imgInfo, .K1, .K2, .T_C2_C1, .angles, .nAngles, 
%         .Pts3, .cylT0, .meanError, .fvals, .cylRadius,
%         .gridPtsP, .output_imgs, .numImages, .idx, .pyEnv, 
%         .imgUndist, .imgHisteq

    % 이미지 이름은 이미 imgInfo로 전달되었다고 가정
    vars.imgInfo = imgInfo; 
    
    % 카메라 파라미터 추출
    [vars.K1, vars.K2, ~, ~, vars.T_C2_C1, ~] = getCamParams(cameraParams);
    
    % 각도 정보 파싱 (예: '01' -> [pan, tilt]) 후 라디안 변환
    angles = parseImgInfo(vars.imgInfo);
    vars.angles = deg2rad(angles);
    
    % 이미지 개수 및 각도 관련 변수
    vars.nAngles = size(vars.angles, 1);
    vars.numImages = numImages;
    
    % Single-cylinder fitting 관련 cell 배열 초기화
    vars.Pts3 = cell(vars.nAngles, 1);
    vars.cylT0 = cell(vars.nAngles, 1);
    vars.meanError = cell(vars.nAngles, 1);
    vars.fvals = cell(vars.nAngles, 1);
    
    % 원통 반경 (예: 45)
    vars.cylRadius = 45;
    
    % grid 점과 출력 이미지 저장용 cell 배열 (numImages×2)
    vars.gridPtsP = cell(numImages, 2);
    vars.output_imgs = cell(numImages, 2);
    
    % undistorted 이미지, 히스토그램 equalization 이미지 저장용 cell 배열
    vars.imgUndist = cell(numImages, 2);
    vars.imgHisteq = cell(numImages, 2);
    
    % 인덱스 범위 구조체: 이미지 전체 범위
    vars.idx = struct('start', 1, 'end', numImages);
    
    % 파이썬 환경 설정 구조체
    vars.pyEnv = struct('utilsDir', utilsDir, 'dllPath', dllPath, 'modulename', moduleName);
    
    %% 이미지 전처리 및 grid 점 검출
    progBar = ProgressBar(numImages);
    for i = 1:numImages
        curImgInfo = vars.imgInfo{i};
        
        % 좌우 이미지 읽기
        I1 = imread(fullfile(input_path, sprintf('%sL.png', curImgInfo)));
        I2 = imread(fullfile(input_path, sprintf('%sR.png', curImgInfo)));
        
        % 이미지 전처리 (undistortion, 히스토그램 equalization)
        [I1_undist, I2_undist, I1_histeq, I2_histeq] = preProcessing(I1, I2, cameraParams);
        
        vars.imgUndist(i,:) = {I1_undist, I2_undist};
        vars.imgHisteq(i,:) = {I1_histeq, I2_histeq};
        
        try
            [vars.gridPtsP{i,1}, vars.output_imgs{i,1}] = makePyGridPts(I1_undist, vars.pyEnv);
            [vars.gridPtsP{i,2}, vars.output_imgs{i,2}] = makePyGridPts(I2_undist, vars.pyEnv);
        catch ME
            warning('makePyGridPts error: %s', ME.message);
        end
        progBar([], [], []);
    end
    progBar.release();
end
