clear all; close all; clc

%% define tankpath, block, and output file name
tankpath = 'C:\Users\anupam\Desktop\salk data\VIPCR\20150915\20150915';
block = 'ORIxLR-2';
output_file_path = 'C:\Users\anupam\Desktop\salk data\VIPCR\20150915\ORIxLR-2.dat';

%% start activeX control
figure(88)
TTX = actxcontrol('TTank.X');
% connect to server
TTX.ConnectServer('Local', 'Me');
% open tank
TTX.OpenTank(tankpath, 'R');
% select experiment ('block')
TTX.SelectBlock(block);

%% To get raws for each channel
for i = 1:32
NumRecs = TTX.ReadEventsV(1000000,'Raws',i,0,0,0,'ALL')
e = TTX.ParseEvV(0, NumRecs);
e_res = reshape(e, numel(e), 1);
raws(:,i) = e_res;
end

%% To get LED timestamps (time led turns on)
N = TTX.ReadEventsV(100000,'LeOn',0,0,0,0,'ALL');
aux_LeON=TTX.ParseEvV(0,N); % LED On times
LeON_info=TTX.ParseEvInfoV(1,0,0); 
LeON_timestamps = TTX.ParseEvInfoV(0,N,6); 

%% write raw data to 16-bit binary file
raws = raws';
output_file = fopen(output_file_path,'w');
fwrite(output_file,raws,'int16');
fclose(output_file);