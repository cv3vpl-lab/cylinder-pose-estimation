function setPyenv(dllPath)
    % checking path. if don't, update it.

    % Check if KMP_DUPLICATE_LIB_OK is not set or is not 'TRUE'
    currentKmp = getenv('KMP_DUPLICATE_LIB_OK');
    if isempty(currentKmp) || ~strcmp(currentKmp, 'TRUE')
        setenv('KMP_DUPLICATE_LIB_OK', 'TRUE');
    end

    % Check and update PATH environment variable
    matlabRuntimePath = fullfile(matlabroot, 'runtime', 'win64');
    newPaths = {dllPath, matlabRuntimePath};
    
    % Check if PATH is already set
    currentPath = getenv('PATH');
    if ~isempty(currentPath)
        % Split the current PATH into parts
        pathParts = strsplit(currentPath, ';');
        
        % Check if dllPath and matlabRuntimePath are already present
        for k = 1:length(newPaths)
            newPath = newPaths{k};
            if ~ismember(newPath, pathParts)
                currentPath = [currentPath ';' newPath];
            end
        end
    else
        % If PATH is not set, initialize it with the new paths
        currentPath = strjoin(newPaths, ';');
    end
    
    % Update PATH if new paths were added
    if ~strcmp(currentPath, getenv('PATH'))
        setenv('PATH', currentPath);
    end
