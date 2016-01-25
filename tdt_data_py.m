function raws = tdt_data_py(tankpath, block)
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
NumRecs = TTX.ReadEventsV(1000000,'Raws',i,0,0,0,'ALL');
e = TTX.ParseEvV(0, NumRecs);
e_res = reshape(e, numel(e), 1);
raws(:,i) = e_res;
end
raws = raws';
