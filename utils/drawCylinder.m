function drawCylinder( cylPose, cylRadius, K, Clrs )

if nargin < 4
    Clrs = [0, 1, 0 ; 1, 0, 0]; % green circles, red line
end

T = vec2T(cylPose);

angles = 0:2*pi/30:2*pi;
ncp = length(angles);
circlepts = [cos(angles) * cylRadius ; zeros(1, ncp) ; sin(angles) * cylRadius];
circlepts = repmat(circlepts, 1, 3);
circlepts(2,ncp+1:end) = [ones(1, ncp) * 50, ones(1, ncp) * 100];

co = [0, 0, 0]';
cn = [0, 100, 0]';

circlepts_cam = transformEuclid(T, [circlepts, co, cn]);
circlepts_img = projPts3(circlepts_cam, K);

ncp3 = ncp*3;
scatter( circlepts_img(1,1:ncp3), circlepts_img(2,1:ncp3), 5, Clrs(1,:), 'filled' )
line( circlepts_img(1, ncp3+[1,2]), circlepts_img(2, ncp3+[1,2]), 'Color', Clrs(2,:) )
