function indices = findMatchingIdx(imgInfo, imgName)
    % ...
    indices = [];
    for i = 1:length(imgInfo)
        [~, filename] = fileparts(imgInfo{i});
        candidate = filename(1:end);  % Extract prefix before 'L' or 'R'
        if any(cellfun(@(x) strcmp(candidate, x), imgName))
            indices = [indices, i];
        end
    end
end