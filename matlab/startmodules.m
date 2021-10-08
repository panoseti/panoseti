 insert(py.sys.path,int32(0),[getuserdir filesep 'panoseti' filesep 'pythonlib' filesep])


modeacq={'0x02'};

%threshold for modes 1,2, 6:
threshPE= 3.5; % non dual-mode threshold (use dual mode =0)
%threshold for modes 3, 7:
threshPEIma=3.5; % dual Imaging (use dual mode =1)
threshPEPH=15.5; % dual PH (use dual mode=1)

if strncmp(cell2mat(modeacq),'0x07',4) || strncmp(cell2mat(modeacq),'0x03',4)
    dualmode = 1;
else
    dualmode = 0;
end


IPtab=["192.168.0.4","192.168.0.5","192.168.0.6","192.168.0.7",...
   "192.168.3.248","192.168.3.249","192.168.3.250","192.168.3.251" ]

%  IPtab=["192.168.0.4","192.168.0.5","192.168.0.6","192.168.0.7" ]

% IPtab=["192.168.0.4",...
%    "192.168.3.248" ]
% IPtab=["192.168.3.248" ]

%IPtab=[ "192.168.3.248","192.168.3.249","192.168.3.250","192.168.3.251" ]
%IPtab=["192.168.3.248","192.168.3.249" ]
%IP='192.168.0.4';

for IPn=1:8%8%1:size(IPtab,2)
    IP=IPtab(IPn);
 startqNph
  changepeq
 % pauseboard
end






