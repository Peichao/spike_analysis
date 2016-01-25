function [lfpsi, time_index, epocs] = tdt_LFP_py(tank_path, block)
%% start activeX control
figure(88)
TTX = actxcontrol('TTank.X');
% connect to server
TTX.ConnectServer('Local', 'Me');
% open tank
TTX.OpenTank(tank_path, 'R');
% select experiment ('block')
TTX.SelectBlock(block);

N = TTX.ReadEventsV(1000000,'LFPs',1,0,0,0,'ALL');
aux_LFP = TTX.ParseEvV(0,N); % get LFP values
LFP = double(reshape(aux_LFP, 1, numel(aux_LFP)));

%get the sampling interval and create a time index from the lfp
lfpsi = 1/(TTX.ParseEvInfoV(0,1, 9)); %sampling interval for lfps
time_index = (0:length(LFP)-1)*lfpsi; %time index is created from the lfps to be more accurate when comparing spikes for ppc

disp('Find timing using epoc')
TTX.CreateEpocIndexing;
TTX.ResetFilters;

epocs =  TTX.GetEpocsV('Epoc',0,0,1000); %3/18/13 changed to our TDT event epoc
% remove any unfinished epocs
if epocs(3,end)==0
    epocs(:,end) = [];
end
end