function [gridPts, output_img] = makePyGridPts(input_img, pyEnv)
    % 입력된 이미지를 처리하고 Python을 통해 그리드 포인트를 검출

    % Python 환경 설정
    utilsDir = pyEnv.utilsDir;
    modulename = pyEnv.modulename;

    % utilsDir를 sys.path에 추가 (중복 확인)
    if ~any(strcmp(py.sys.path, utilsDir))
        insert(py.sys.path, int32(0), utilsDir);
    end

    % Python 모듈 임포트
    try
        py_func = py.importlib.import_module(modulename);
        py.importlib.reload(py_func);
    catch ME
        disp('final_grid module import failed:');
        disp(ME.message); 
        output_img = [];
        gridPts = struct();   
        return;
    end    

    % 이미지 처리
    processed_img = py.numpy.array(input_img);
    
    try
        outputs = py_func.detect_grid(processed_img);
        output = outputs{1};
        
        % 이미지 포맷 변환 (BGR -> RGB)
        if ndims(output) == 3 && size(output,3) == 3
            output = output(:, :, [3, 2, 1]);
        end
        output_img = output;

        % JSON 문자열 처리
        jsonStr = char(outputs{2});
        gridPts = jsondecode(jsonStr);
        gridPts.points = pointsStruct2mat(gridPts.points);
        
    catch ME
        fprintf('Error makePyGridPts\n');
        disp(ME.message);
    end
end