% The code is used for converting Panoconfig.mat to q_info.json
%

clear;
clc;
close all;

% define how many quabos are on each mobo,
% and how many detectors are on each quabo
QuabosonMobo =4;
DetectorsonQubao = 4;

% get the uids in the system
quabo_uids = jsondecode(fileread('quabo_uids.json'));
% get dome number in the system
DomeNum = size(quabo_uids.domes, 1);
% create an array, which contains IP and UID
% the first column is ip addr of quabo0; the following four columns contains uids of the four quabos.
n = 1;
for i = 1:DomeNum
    % get module number in each dome
    dome = quabo_uids.domes(i);
    ModuleNum = size(dome.modules,1);
    for j = 1:ModuleNum
        module = dome.modules(j);
        ipaddr = module.ip_addr;
        quabos = module.quabos;
        for k = 1:QuabosonMobo
            ip_uid{n,1} = ipaddr;
            ip_uid{n,k+1} = quabos(k).uid;
        end
        n = n + 1;
    end
end

% load the config tables
Panoconfig = load(['.' filesep 'Panoconfig.mat']);

% get the size of Panoconfig.config
% in this table the meanings of columns are:
% ModuleName MoboSN 4*(QuaboSN, 4*DetectorSN) Loc 4*IP_Addresses fixedaltaz and two values
% (what do the last 3 columns mean??)
[r,c] = size(Panoconfig.config);

n = 0;
for i = 1:r
    if(strncmp(Panoconfig.config{i,23}, 'LICK', 4) == 1)
        quabo0_ip = Panoconfig.config{i, 24};
        for j = 1:size(ip_uid , 1)
            if(strcmp(quabo0_ip,ip_uid{j,1}) == 1)
               % index is a bit complicated in Panoconfig.config
               for k = 1:QuabosonMobo
                    n = n + 1;
                    q_info(n).uid = ip_uid{j,k+1};
                    q_serialno = Panoconfig.config{i, k*(DetectorsonQubao + 1)-2};
                    serialno_str = strsplit(q_serialno,'0');
                    q_info(n).serialno = serialno_str{2};
                    if(str2num(q_info(n).serialno) < 30) 
                        q_info(n).board_version='qfp';
                    else
                        q_info(n).board_version='bga';
                    end
                    q_info(n).detector_serialno = [];
                    for m = 1:DetectorsonQubao
                        q_info(n).detector_serialno = [q_info(n).detector_serialno, ...
                                                       Panoconfig.config{i, k*(DetectorsonQubao + 1) -2 + m}];
                    end
               end
            end
        end
    end
end

content = [];
for i = 1:n
    content = [content, q_info(i)];
end

json = jsonencode(content,"PrettyPrint",true)
% write json content to a json file
fid = fopen('quabo_info.json', 'wt');
fprintf(fid, '%s',json);
fclose(fid);
