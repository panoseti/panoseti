cd '/home/panosetigraph/panoseti/PANOapp/'
fpgadir='/home/panosetigraph/panoseti/FPGA/quabo_v0105';
disp('Starting boards initialization...')
%firmware='quabo_0103_22C57D1F.bin';
firmware='quabo_0116C_23CBEAFB.bin';
%% action=0 => reboot
%%        1 => load silver firmware in fpgadir\startloadx.py
%% action=2 => load gold firm in fpga\startloadgoldx.py
action=0; 
IPtab=["192.168.0.4","192.168.0.5","192.168.0.6","192.168.0.7",...
   "192.168.3.248","192.168.3.249","192.168.3.250","192.168.3.251" ]
%IPtab=["192.168.0.7","192.168.3.251"]

if action==0
    loadfirm='';
    gold='';
elseif action ==1
    loadfirm='load';
    gold='';
elseif action ==2
      loadfirm='load';
    gold='gold';
end

for IPn=1:8%1:size(IPtab,2)
commandStr = ['cd ' fpgadir ' ; python startq' loadfirm gold 'x.py ' char(IPtab(IPn)) ' ' firmware];
 [statusx, commandOutx] = system(commandStr);
 if statusx==1
     disp(['Board IP ' IPtab(IPn) ' initialized...'])
   %  fprintf('squared result is %s\n',(commandOut248));
    else
     disp(commandOutx)
 end

end

% commandStr = ['cd ' fpgadir ' & python startq248.py'];
%  [status248, commandOut248] = system(commandStr);
%  if status248==1
%      disp('Board IP 248 initialized...')
%    %  fprintf('squared result is %s\n',(commandOut248));
%     else
%      disp(commandOut248)
%  end
%  
%  commandStr =  ['cd ' fpgadir ' & python startq249.py'];
%  [status249, commandOut249] = system(commandStr);
%  if status249==1
%      disp('Board IP 249 initialized...')
%    %  fprintf('squared result is %s\n',(commandOut249));
%     else
%      disp(commandOut249)
%  end
%  
%  commandStr =  ['cd ' fpgadir ' & python startq250.py'];
%  [status250, commandOut250] = system(commandStr);
%  if status250==1
%      disp('Board IP 250 initialized...')
%    %  fprintf('squared result is %s\n',(commandOut250));
%     else
%      disp(commandOut250)
%  end
%  
% commandStr =  ['cd ' fpgadir ' & python startq251.py'];
%  [status251, commandOut251] = system(commandStr);
%  if status251==1
%      disp('Board IP 251 initialized...')
%     % fprintf('squared result is %s\n',(commandOut251));
%      else
%      disp(commandOut251)
%  end
%  
%  
% commandStr =  ['cd ' fpgadir ' & python startq4.py'];
%  [status4, commandOut4] = system(commandStr);
%  if status4==1
%      disp('Board IP 4 initialized...')
%     % fprintf('squared result is %s\n',(commandOut4));
%      else
%      disp(commandOut4)
%  end
% 
%  commandStr =  ['cd ' fpgadir ' & python startq5.py'];
%  [status5, commandOut5] = system(commandStr);
%  if status5==1
%      disp('Board IP 5 initialized...')
%     % fprintf('squared result is %s\n',(commandOut5));
%      else
%      disp(commandOut5)
%  end
% 
%  
%  commandStr =  ['cd ' fpgadir ' & python startq6.py'];
%  [status6, commandOut6] = system(commandStr);
%  if status6==1
%      disp('Board IP 6 initialized...')
%     % fprintf('squared result is %s\n',(commandOut6));
%      else
%      disp(commandOut6)
%  end
%  
%  commandStr =  ['cd ' fpgadir ' & python startq7.py'];
%  [status7, commandOut7] = system(commandStr);
%  if status7==1
%      disp('Board IP 7 initialized...')
%  %    fprintf('squared result is %s\n',(commandOut7));
%   else
%      disp(commandOut7)
%  end
%  
  disp('Boards initializations finished!!!')