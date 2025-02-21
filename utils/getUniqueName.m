function uniqueNames = getUniqueName(inputPath)
    % 입력 경로의 모든 PNG 파일 목록 가져오기
    files = dir(fullfile(inputPath, '*.png'));
    
    % 유효한 파일 이름을 저장할 셀 배열 초기화
    baseNames = {};
    
    for k = 1:length(files)
        fname = files(k).name;
        % 파일 이름이 최소 5자 이상이고 'L.png'로 끝나는지 확인
        if length(fname) >= 5 && strcmp(fname(end-4:end), 'L.png')
            % 마지막 5자 ('L.png') 제거
            baseName = fname(1:end-5);
            baseNames{end+1} = baseName; %#ok<AGROW>
        end
    end
    
    % 유니크한 이름 추출
    uniqueNames = unique(baseNames);
end
