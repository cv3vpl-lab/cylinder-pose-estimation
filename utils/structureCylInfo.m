function cylInfo = structureCylInfo(idx, Xcyl, Ycyl, Zcyl)
    % cylInfo 구조체 생성
    nIdx = length(idx);
    cylInfo = struct('X', cell(nIdx, 1), 'Y', cell(nIdx, 1), 'Z', cell(nIdx, 1));

    for i = 1:nIdx
        cylInfo(i).X = Xcyl{idx(i)};
        cylInfo(i).Y = Ycyl{idx(i)};
        cylInfo(i).Z = Zcyl{idx(i)};
    end
end