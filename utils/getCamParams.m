function [K1, K2, T_C1_Ps, T_C2_Ps, T_C2_C1, calibPts] = getCamParams(sParams)

cParam1 = sParams.CameraParameters1;
cParam2 = sParams.CameraParameters2;

K1 = cParam1.K;
K2 = cParam2.K;

T_C2_C1 = sParams.PoseCamera2.A;

nPatterns = length(cParam1.PatternExtrinsics);
T_C1_Ps = zeros(4, 4, nPatterns);
T_C2_Ps = zeros(4, 4, nPatterns);

for i = 1:nPatterns
    T_C1_Ps(:,:,i) = cParam1.PatternExtrinsics(i).A;
    T_C2_Ps(:,:,i) = cParam2.PatternExtrinsics(i).A;
end

calibPts = cParam1.WorldPoints;