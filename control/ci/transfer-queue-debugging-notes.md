# Transfer Queue Debugging notes

### Incorrect Start Transactions

- start does not do the network sweep to check reachability of all quabos, allowing hashpipe to start with no data. This is probably find in the sw-only environment, but in the hardware-software environment this is an issue because we want the quabos to be reachable or the test fails in most cases. Though I can see some value in allowing hashpipe-only hardware-software tests to check the behavior over real router hardware.
    
    ```bash
    [04/26/26 01:09:07] WARNING  Quabo at 192.168.88.152:60003 is UNREACHABLE: Quabo ping failed   
                                 (UDP timeout) (Non-fatal in container/CI environment)             
    [2026-04-26 01:09:07,998] WARNING: Quabo at 192.168.88.152:60003 is UNREACHABLE: Quabo ping failed (UDP timeout) (Non-fatal in container/CI environment)
    [04/26/26 01:09:08] WARNING  Quabo at 192.168.88.152:60002 is UNREACHABLE: Quabo ping failed   
                                 (UDP timeout) (Non-fatal in container/CI environment)             
    [2026-04-26 01:09:08,002] WARNING: Quabo at 192.168.88.152:60002 is UNREACHABLE: Quabo ping failed (UDP timeout) (Non-fatal in container/CI environment)
                        WARNING  Quabo at 192.168.88.152:60000 is UNREACHABLE: Quabo ping failed   
                                 (UDP timeout) (Non-fatal in container/CI environment)             
    [2026-04-26 01:09:08,005] WARNING: Quabo at 192.168.88.152:60000 is UNREACHABLE: Quabo ping failed (UDP timeout) (Non-fatal in container/CI environment)
                        WARNING  Quabo at 192.168.88.152:60001 is UNREACHABLE: Quabo ping failed   
                                 (UDP timeout) (Non-fatal in container/CI environment)             
    [2026-04-26 01:09:08,007] WARNING: Quabo at 192.168.88.152:60001 is UNREACHABLE: Quabo ping failed (UDP timeout) (Non-fatal in container/CI environment)
    ```
    
- When I turned off the quabos and tried to run start, I was able to in the hardware-software tests because the in container flag was active, bypassing checks and preventing us from testing the transaction behavior.
- I can run multiple starts. the first one fails with an aborted but subsequent ones work fine? Oh but the first hashpipe luckily not stopped.
    - Confirmed: there seems to be a race condition in the start up procedure.
    - When I run start multiple times in a row, sometimes the transaction aborts correctly and other times it fails.
    - Sometimes a subset of the quabo config and data flow UDP commands are sent off before the hashpipe check is performed, resulting in unexpected reconfig even if the transaction is aborted.
    - I suspect this may be due to the lock not working or the ledger path not being updated after the state refactor ?
    - Running stop multiple times in a row seems to repeat the stop transaction logic from the top. I’m not sure if this is intended?
        - Maybe this is good for idempotency?
    - The daq node also randomly crashed one time. like the grpc and the entire container stack just died for no reason and I have no idea why and if this is a problem with stale code etc → crashed containers have no logs for testing?

### Incorrect path resolution:

- 2026-04-26 01:09:06,981] INFO: data_packet_destination: 192.168.1.1
Warning: Log directory '/app/logs' is not writable ([Errno 2] No such file or directory: '/app/logs/.perm_test'). Falling back to '/tmp/panoseti_logs'Warning: Log directory '/app/logs' is not writable ([Errno 2] No such file or directory: '/app/logs/.perm_test'). Falling back to '/tmp/panoseti_logs'

### Multiple gRPC server processes

- While we allow multiple grpc services to attach to a grpc server on a given node, we shouldn’t have multiple grpc servers on the same node, as they’ll likely use the same config and compete for packets. We don’t need load balancing on the same node; we need exclusivity so that at most one grpc server is active.
- gRPC: I noticed that it’s possible to have multiple unified server instances running! Then there’s ambiguity about who should handle things. I feel like per node there should be just one panoseti server. Perhaps we need an elegant way for servers to detect if there are other unified servers active and refuse to start with the pseti-grpc server command? Or even just part of the server?

### Incorrect Stop Tranactions

- Also when hashpipe was stopping I noticed this bug; the start daq just accepted the start command without a force option even though it detected an active hashpipe instance! Since all hashpipe processes are children of the grpc process (the daq control service starts and manages them), I feel like this should be a problem.
- Here are the logs:
    
    [04/26/26 01:08:26] INFO     Checking Daq Node status...
    
    INFO daq_control_server — Starting HASHPIPE instance...
    [04/26/26 01:08:34] INFO     Starting HASHPIPE instance...
    
    WARNING  Found 1 HASHPIPE instances running. pids: [70]
    
    WARNING daq_control_server — Found 1 HASHPIPE instances running. pids: [70]
    [04/26/26 01:08:35] INFO     Stop HASHPIPE instance...
    
    INFO daq_control_server — Stop HASHPIPE instance...
    INFO hp_stdout — NET_THREAD Ended
    INFO hp_stdout — COMPUTE_THREAD Ended
    INFO hp_stdout — Returned Compute_thread
    INFO hp_stdout — OUTPUT_THREAD Ended
    INFO hp_stdout — Returned Output_thread
    INFO hp_stdout — Returned Net_thread
    INFO hp_stdout — Joined thread 'output_thread'
    INFO hp_stdout — Joined thread 'compute_thread'
    INFO hp_stdout — Joined thread 'net_thread'
    WARNING asyncio — child process pid 70 exit status already read:  will report returncode 255
    

### UX issues

- UX: the pseti obs transfer command has no progress bar. There’s also no resolved path for the ledgers, making it hard to manually inspect logs. (how to add this without making it cramped?
- The start command dumps the entire validation output onto the screen. I feel like this is nice to have but it can make it hard to see errors.
- The quabo_driver is noisy and spits out warnings and errors for timeouts during reboot. These are expected and I noticed that it’s making me ignore them. So if there’s actually an unexpected error I’ll miss it.
    - Perhaps we should decrease the log level by 1 unless a verbose or log-level flag is given this way timemouts might be warnings only. Though I’m not sure what a higher log level would constitute, as during reboot the worst that can happen is the quabo not respond. We can’t really tell what’s happening…
- I’m finding it helpful to run pseti grpc —host <daq node> status to see the grpc status and things like hashpipe activity on our remote node. But it’s cumbersome to manually hunt down the ip address and account for port forwarding etc. It would be really nice if like the status command could automatically provide rich context on the observing status as the name suggests with flags or subcommands for things like network sweeps, checking reachability. Maybe even having a watch feature so it polls the status and refreshses the output every few seconds (configurable).
- The ledgers are also hard to access. Like it would be very convenient if there were a way to easily inspect ledgers, or at the very least just have full paths so I can cat or vim into the ledgers (without making the screen super cramped). Though we should be careful to avoid letting people manually modify the ledgers unless they know what they’re doing…

### Transfer Queue

- The transfer queue has lots of entries pending and while it seems to be prioritizing the oldest run. However, the oldest bounces bounces back and forth between active and pending without any retries counter being updated. There’s no warning message or errors so I have no idea what’s wrong.
- The transfer queue tail command is broken:
    - root@panoseti-headnode-ucb:/app# pseti obs transfer  tail
    Log file not found: /app/state/logs/transfer_daemon/current.log
- When I s