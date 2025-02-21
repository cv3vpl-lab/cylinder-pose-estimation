function plotCylinders3D(imgInfo, imgName, Pts3s, cylInfo, drawOption, colors, varargin)
    % 3D 시각화 함수
    % drawOption: 'Pts3s' 또는 'cylinder'
    % colors: 각 데이터의 색상 (예: {'red', 'green', 'blue'})
    % varargin: Parent axes specification

    % Default colors (인자 순서에 따라 nargin 검사)
    if nargin < 6 || isempty(colors)
        colors = {'red', 'green', 'blue'};
    end

    idx = findMatchingIdx(imgInfo, imgName);
    
    % Parent axes 결정
    parentAxes = [];
    if any(strcmp('Parent', {varargin{:}}))
        parentAxesIdx = find(strcmp('Parent', {varargin{:}}));
        parentAxes = varargin{parentAxesIdx + 1};
    else
        figure;
        parentAxes = gca;
    end

    view(parentAxes, 3); % 3D 뷰 설정
    hold(parentAxes, 'on');
    % (Axes의 ZDir를 바꾸지 않고, 데이터 자체를 반전시킵니다.)
    
    switch drawOption
        case 'Pts3s'
            % 포인트 클라우드만 그리기
            for k = 1:length(idx)
                i = idx(k);
                % 파일 이름 추출 (candidate는 imgName과 일치하는 값)
                [~, candidate] = fileparts(imgInfo{i});
                pts = Pts3s{i}';
                % 포인트 클라우드 데이터의 z 좌표 반전
                pts(:,3) = -pts(:,3);
                % colors 배열의 요소가 부족할 경우 순환 사용
                curColor = colors{ mod(k-1, numel(colors)) + 1 };
                pastelColor = getPastelColorRGB(curColor);
                numPts = size(pts,1);
                rgb = repmat(pastelColor, numPts, 1);
                ptCloud = pointCloud(pts, 'Color', rgb);
                pcshow(ptCloud, 'MarkerSize', 100, 'Parent', parentAxes);
                % 범례용 dummy plot (범례 텍스트는 흰색으로 설정할 예정)
                plot3(parentAxes, NaN, NaN, NaN, 'o', 'Color', double(pastelColor)/255, ...
                    'MarkerSize', 10, 'DisplayName', candidate);
            end
        case 'cylinder'
            % 실린더와 포인트 클라우드를 함께 그리기
            for k = 1:length(idx)
                i = idx(k);
                [~, candidate] = fileparts(imgInfo{i});
                curColor = colors{ mod(k-1, numel(colors)) + 1 };
                pastelColor = getPastelColorRGB(curColor);
                
                % 실린더의 Z 데이터 반전
                surf(parentAxes, cylInfo(k).X, cylInfo(k).Y, -cylInfo(k).Z, ...
                    'FaceColor', double(pastelColor)/255, ...
                    'FaceAlpha', 0.2, ...
                    'EdgeColor', 'none', ...
                    'DisplayName', candidate);
                
                if ~isempty(Pts3s{i})
                    pts = Pts3s{i}';
                    pts(:,3) = -pts(:,3);
                    numPts = size(pts,1);
                    rgb = repmat(pastelColor, numPts, 1);
                    ptCloud = pointCloud(pts, 'Color', rgb);
                    pcshow(ptCloud, 'MarkerSize', 100, 'Parent', parentAxes);
                end
            end
        otherwise
            error('Invalid drawOption: Use ''Pts3s'' or ''cylinder''.');
    end

    xlabel(parentAxes, 'X');
    ylabel(parentAxes, 'Y');
    zlabel(parentAxes, 'Z');
    title(parentAxes, sprintf('imgName = %s', strjoin(imgName, ', ')));
    
    % 범례 추가 후 텍스트 색상을 흰색으로 설정
    hLegend = legend(parentAxes, 'show');
    set(hLegend, 'TextColor', 'white');
    
    hold(parentAxes, 'off');
end

function pastelRGB = getPastelColorRGB(colorName)
    % getPastelColorRGB: 입력된 색상 문자열에 대해 파스텔톤 RGB (0-255 범위) 반환
    switch lower(colorName)
        case 'red'
            pastelRGB = uint8([255, 182, 193]); % light pink
        case 'green'
            pastelRGB = uint8([152, 251, 152]); % pale green
        case 'blue'
            pastelRGB = uint8([173, 216, 230]); % light blue
        case 'yellow'
            pastelRGB = uint8([255, 255, 224]); % light yellow
        case 'magenta'
            pastelRGB = uint8([255, 182, 255]); % light magenta
        case 'cyan'
            pastelRGB = uint8([224, 255, 255]); % light cyan
        otherwise
            pastelRGB = uint8([200, 200, 200]); % 기본 파스텔 회색
    end
end
