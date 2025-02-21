function M = numpy2mat(nd)
    % nd: py.numpy.ndarray
    % 1. numpy 배열의 shape 정보를 MATLAB 배열로 변환
    shp = cellfun(@double, cell(nd.shape));
    % 2. numpy 배열을 평탄화하여 iterator로 얻기
    it = py.numpy.nditer(nd);
    % 3. iterator를 사용해 평탄화된 데이터를 MATLAB double 벡터로 변환
    Mvec = double(py.array.array('d', it));
    % 4. 원래 shape대로 재구성
    M = reshape(Mvec, shp);
end
