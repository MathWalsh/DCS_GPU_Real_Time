close all; clear all; clc;

convOut = 0;
ref_multOut = 0;
PC_out =1;
mult1 = 1;
mult2 = 2;

IniFileName = 'FilterSinus_multiChannel';
config = readIniFile(IniFileName);

% filepath = 'C:\Users\mawal47\source\CompuScope\CompuScope C SDK\C Samples\Advanced\GPU\FilterSinus1MHz';
n = 31;
fs = str2double(config.Acquisition.SampleRate);
b = fir1(n, 5e6*2/fs,"low");
Shift1 = 50e6;
bS = single(b.*exp(1j.*2*pi*Shift1/fs*(0:n)));
% 
% filename = 'FilterCOSR.bin';
% fileID = fopen(fullfile(filepath,filename),'w', 'l');
% fwrite(fileID,flip(real(bS)),'single', 'l');
% fclose(fileID);
% 
% filename = 'FilterCOSI.bin';
% fileID = fopen(fullfile(filepath,filename),'w', 'l');
% fwrite(fileID,flip(imag(bS)),'single', 'l');
% fclose(fileID);




if strcmp(config.Acquisition.Mode, 'Single') == 1
    Nch = 1;
elseif strcmp(config.Acquisition.Mode, 'Dual') == 1
    Nch = 2;
elseif strcmp(config.Acquisition.Mode, 'Quad') == 1
    Nch = 4;
elseif strcmp(config.Acquisition.Mode, 'Octal') == 1
    Nch = 8;
end

Nbits = 16;
NBytesPerPoint = Nbits / 8;
BufferSize =str2double(config.StmConfig.BufferSize);
NBufferSize = BufferSize / NBytesPerPoint;

NumberOfBuffers = 2;
Npts = str2double(config.StmConfig.NptsTot) *  NumberOfBuffers / Nch;



filename = sprintf('%s_I1.dat',config.StmConfig.DataFile);
fid = fopen(filename, 'rb');

% fid = fopen('output_phi_dfr.bin', 'rb');

% Check if the file is successfully opened


if fid == -1
    error('Error: Could not open the file.');
end
% Read the floats from the binary file
temp = fread(fid, 'int16');
fclose(fid);


input = zeros(Npts,Nch, 'single');
inputF = complex(zeros(Npts, Nch, 'single'), zeros(Npts, Nch, 'single'));
multref = complex(zeros(Npts, 1, 'single'), zeros(Npts, 1, 'single'));
PC_IGMs = complex(zeros(Npts, 1, 'single'), zeros(Npts, 1, 'single'));
Nbuffer = ceil(Nch*Npts/NBufferSize);

if convOut == 1
   for i=1:Nch
    Fullrange = str2double(config.(sprintf('Channel%d',i)).Range);
    input(1:Npts,i) = single(temp(i:Nch:end))/2^(Nbits)*Fullrange;
    inputF(1:Npts,i) = circshift(conv(input(1:Npts,i),bS,'same'),length(bS)/2);
    end
elseif ref_multOut ==1

   for i=1:Nch
    Fullrange = str2double(config.(sprintf('Channel%d',i)).Range);
    input(1:Npts,i) = temp(i:Nch:end);
    inputF(1:Npts,i) = circshift(conv(input(1:Npts,i),bS,'same'),length(bS)/2);
    multref = (inputF(1:Npts,mult1) .* inputF(1:Npts,mult2))/2^(Nbits)*Fullrange;
    end
elseif PC_out == 1
    for i=1:Nch
    Fullrange = str2double(config.(sprintf('Channel%d',i)).Range);
    input(1:Npts,i) = temp(i:Nch:end);
    inputF(1:Npts,i) = circshift(conv(input(1:Npts,i),bS,'same'),length(bS)/2);
    multref = (inputF(1:Npts,mult1) .* inputF(1:Npts,mult2));
    end

    PC_IGMs = inputF(:,mult1).*exp(1j.*angle(multref))/2^(Nbits)*Fullrange;

end









filename = sprintf('%s_O1.dat',config.StmConfig.DataFile);
fid = fopen(filename, 'rb');


% Check if the
% file is successfully opened
if fid == -1
    error('Error: Could not open the file.');
end
% Read the floats from the binary file

if convOut == 1
    temp = fread(fid, [2*Npts*Nch], 'single');
    % temp = fread(fid, 'single');
    fclose(fid);

    % output = zeros(Npts, Nch, 'single');
    output = complex(zeros(Npts, Nch, 'single'), zeros(Npts, Nch, 'single'));
    NBufferSizeO = NBufferSize*2
    NBufferSizeCh = NBufferSizeO/Nch;
    for i=1:Nch
        Fullrange = str2double(config.(sprintf('Channel%d',i)).Range);
        for j=1:Nbuffer
            idxi = NBufferSizeO*(j-1) + 1 + (i-1)*NBufferSizeO/Nch;
            idxf = NBufferSizeO*(j-1) + i*NBufferSizeCh;
            idxiO = NBufferSizeCh*(j-1)/2 + 1;
            idxfO = (NBufferSizeCh)/2 *j;
            output(idxiO:idxfO,i) = complex(temp(idxi:2:idxf,1), temp(idxi+1:2:idxf));
        end
        output(:,i) = output(:,i)/2^(Nbits)*Fullrange;
    end
    
elseif ref_multOut ==1

    temp = fread(fid, 2*Npts, 'single');
    % temp = fread(fid, 'single');
    fclose(fid);

    % output = zeros(Npts, Nch, 'single');
    output = complex(single(temp(1:2:end)), single(temp(2:2:end)))/2^(Nbits)*Fullrange;

elseif PC_out == 1
    temp = fread(fid, 2*Npts, 'single');
    % temp = fread(fid, 'single');
    fclose(fid);

    % output = zeros(Npts, Nch, 'single');
    output = complex(single(temp(1:2:end)), single(temp(2:2:end)))/2^(Nbits)*Fullrange;

end

%%


if convOut == 1
    for i = 1:Nch
        figure(i);cla;
        ax(1)=subplot(311);cla;
        plot(input(:,i));
        title(sprintf("Input channel: %d",i));
        ylabel('Tension [mV]');
        ax(2)=subplot(312);cla;
        plot(real(output(:,i)));
        hold on;
        plot(imag(output(:,i)));
        ylabel('Tension [mV]');
        title(sprintf("Output convolution channel: %d",i));

        ax(3)=subplot(313);cla;
        plot(real(output(:,i)-inputF(:,i)));
        hold on;
        plot(imag(output(:,i)-inputF(:,i)));
        title(sprintf("Convolution GPU - Convolution Matlab channel: %d",i));
        linkaxes(ax,'x');
        ylabel('Tension [mV]');
        xlabel('Npts [-]')
    end
elseif ref_multOut ==1
    figure(i);cla;
    ax(1)=subplot(411);cla;
    plot(real(output));
    title("Real ref1")
    ylabel('Tension [mV]');
    ax(2)=subplot(412);cla;
    plot(real(output-multref));
    title("Real ref1 : GPU - Matlab");
    ax(3)=subplot(413);cla;
    plot(imag(output));
    title("Imag ref1")
    ylabel('Tension [mV]');
    ax(4)=subplot(414);cla;
    plot(imag(output-multref));
    title("Imag ref1 : GPU - Matlab");
    ylabel('Tension [mV]');
    linkaxes(ax,'x');
elseif PC_out ==1
    figure(i);cla;
    ax(1)=subplot(411);cla;
    plot(real(output));
    title("Real IGMs phase corrected")
    ylabel('Tension [mV]');
    ax(2)=subplot(412);cla;
    plot(real(output-PC_IGMs));
    title("Real IGMs phase corrected : GPU - Matlab");
    ax(3)=subplot(413);cla;
    plot(imag(output));
    title("Imag IGMs phase corrected")
    ylabel('Tension [mV]');
    ax(4)=subplot(414);cla;
    plot(imag(output-PC_IGMs));
    title("Imag IGMs phase corrected: GPU - Matlab");
    ylabel('Tension [mV]');
    linkaxes(ax,'x');
end