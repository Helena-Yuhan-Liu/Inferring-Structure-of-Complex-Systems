% Recursively go through the folders and load all data 

%% Cropped 

clear; 

ldir = './CroppedYale';
subdirs = dir(ldir);

xdata = []; subnum = []; gender=[]; first_im = 1; 
xdata_cell=cell(39,1); % each cell contains all images pertaining to one subject
for ii = 1:length(subdirs)
    subdir = subdirs(ii); 
    if ~strcmp(subdir.name, '.') && ~strcmp(subdir.name, '..')
        subsubdirs = dir([subdir.folder '/' subdir.name]);
        for jj = 1:length(subsubdirs)
            subsubdir = subsubdirs(jj);
            if ~strcmp(subsubdir.name, '.') && ~strcmp(subsubdir.name, '..')
                % Load data
                importfile([subsubdir.folder '/' subsubdir.name]);
                % Stack the image and subject number
                var_name = split(subsubdir.name,'.'); 
                if ~strcmp(var_name(end),'bad')
                    var_name=char(var_name(1));
                    var_name = strrep(var_name,'+','_');
                    var_name = strrep(var_name,'-','_');
                    eval(['x = ' var_name ';']);
                    xdata = [xdata reshape(x, [length(x(:)) 1])]; 
                    sn = str2double(subsubdir.name(6:7)); 
                    subnum = [subnum; sn]; 
                    gender = [gender; get_gender(sn)]; 
                    xdata_cell{sn} = [xdata_cell{sn} reshape(x, [length(x(:)) 1])];
                    % Get the image dimension
                    if first_im
                        nrow = size(x,1);
                        ncol = size(x,2);
                        first_im = 0; 
                    end 
                end 
            end
        end
    end
end 

save('cropped_data.mat', 'xdata', 'xdata_cell', 'subnum', 'gender', 'nrow', 'ncol'); 

%% Uncropped 

clear; 

ldir = './yalefaces';
subdirs = dir(ldir);

xdata = []; subnum = []; emotion=[]; first_im = 1; 
for ii = 1:length(subdirs)
    subdir = subdirs(ii); 
    if ~strcmp(subdir.name, '.') && ~strcmp(subdir.name, '..')
        subsubdirs = dir([subdir.folder '/' subdir.name]);
        for jj = 1:length(subsubdirs)
            subsubdir = subsubdirs(jj);
            if ~strcmp(subsubdir.name, '.') && ~strcmp(subsubdir.name, '..')
                % Load data
                importfile_uncropped([subsubdir.folder '/' subsubdir.name]);
                % Stack the image, subject number and emotion
                var_name = split(subsubdir.name,'.');
                emotion = [emotion; var_name(2)]; 
                subnum = [subnum; num2str(subsubdir.name(8:9))];
                x = cdata;                 
                xdata = [xdata reshape(x, [length(x(:)) 1])];
                if first_im
                    nrow = size(x,1);
                    ncol = size(x,2);
                    first_im = 0; 
                end                    
            end
        end
    end
end 

save('uncropped_data.mat', 'xdata', 'subnum', 'emotion', 'nrow', 'ncol');
