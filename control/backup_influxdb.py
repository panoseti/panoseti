#! /usr/bin/env python3

"""
Script that creates a backup of the influxdb database. After each backup attempt,
log data is recorded in a file with the following structure:
{
    "backups: [
        {Backup log entry 1},
    ]
}
"""

import os
import datetime
import json


BACKUP_DIR_PATH = './.influxdb_backups'
RESTORE_DIR_PATH = BACKUP_DIR_PATH
backup_log_filename = 'backup_log.json'
backup_log_path = '{0}/{1}'.format(BACKUP_DIR_PATH, backup_log_filename)


def get_backup_folder_path():
    date = datetime.datetime.utcnow().strftime("%Y_%m_%dT%H_%M_%SZ")
    backup_folder_path = '{0}/influx_backup_{1}'.format(BACKUP_DIR_PATH, date)
    return backup_folder_path

def get_data_start_time():
    start = None
    if os.path.exists(backup_log_path):
        with open(backup_log_path) as f:
            s = f.read()
            c = json.loads(s)
            start = c["backups"][-1]["UTC_time_range_end"]
    return start


def get_data_end_time():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def update_backup_log(backup_folder_path, exit_status, start, end):
    """
    Updates the json file (creating it if necessary) and storing backup log data
    """
    new_log_data = {
        "backup_number": 0,
        "UTC_timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backup_abs_path": os.path.abspath(backup_folder_path),
        "backup_status": exit_status,
        "UTC_time_range_start": start,
        "UTC_time_range_end": end,
    }
    if os.path.exists(backup_log_path):
        with open(backup_log_path, 'r+') as f:
            s = f.read()
            c = json.loads(s)
            new_log_data["backup_number"] = c["backups"][-1]["backup_number"] + 1
            c["backups"].append(new_log_data)
            json_obj = json.dumps(c, indent=4)
            f.seek(0)
            f.write(json_obj)
    else:
        c = {
            "backups": [
                new_log_data
            ]
        }
        with open(backup_log_path, 'w+') as f:
            json_obj = json.dumps(c, indent=4)
            f.write(json_obj)


def do_backup(backup_folder_path, start, end):
    """
    1. Creates a new directory: 'influx_backup_{current date in year-month-day-format},
    2. Creates a backup of the influxdb data generated since the last backup, and
    3. Adds a log entry to the log file.
    """
    # Get and run commands
    make_dir_command = 'mkdir -p {0}'.format(backup_folder_path)
    os.system(make_dir_command)
    if start:
        backup_command = 'influxd backup -portable -start {0} -end {1} {2} '.format(start, end, backup_folder_path)
    else:
        backup_command = 'influxd backup -portable -end {0} {1}'.format(end, backup_folder_path)
    exit_status = os.system(backup_command)
    exit_status = 'SUCCESS' if not exit_status else 'FAILED'
    # Add log entry for this backup.
    update_backup_log(backup_folder_path, exit_status, start, end)
    return exit_status


def do_restore():
    """
    Restore the metadata database using the files stored in the restore directory.
    """
    backup_directories = [name for name in os.listdir(RESTORE_DIR_PATH)]
    backup_directories.remove(backup_log_filename)
    restore_command = 'influxd restore -portable -db metadata -newdb testdb {0}/{1}'
    print(backup_directories)
    for dir in backup_directories:
        os.system(restore_command.format(RESTORE_DIR_PATH, dir))
        print('Restoring: {0}...'.format(dir))


def main():
    backup_folder_path = get_backup_folder_path()
    start, end = get_data_start_time(), get_data_end_time()
    exit_status = do_backup(backup_folder_path, start, end)
    if exit_status == 'SUCCESS':
        msg = 'Successfully backed up the database at {0}'.format(backup_folder_path)
        print(msg)
    else:
        msg = 'Failed to back up the database.'
        print(msg)


if __name__ == '__main__':
    main()