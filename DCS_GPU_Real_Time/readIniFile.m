function config = readIniFile(fileName)

% Read the configuration file
fileID = fopen(sprintf('%s.ini',fileName), 'r');

% Initialize a structure to store the configuration data
config = struct();
currentSection = '';

% Read the file line by line
while ~feof(fileID)
    line = fgetl(fileID);
    
     % Skip lines starting with a semicolon (;)
    if startsWith(line, ';')
        continue;
    end

    % Check for section headers
    if startsWith(line, '[') && endsWith(line, ']')
        currentSection = strtrim(line(2:end-1));
        config.(currentSection) = struct();
    elseif contains(line, '=')
        % Key-value pair, parse and store in the current section
        parts = split(line, '=');
        key = strtrim(parts{1});
        value = strtrim(parts{2});
        
        % Store in the current section
        config.(currentSection).(key) = value;
    end
end

% Close the file
fclose(fileID);