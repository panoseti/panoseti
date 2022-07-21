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


# Time between backups in days.
BACKUP_INTERVAL = 30
BACKUP_DIR_PATH = './.influxdb_backups'
backup_log_filename = '{0}/backup_log.json'.format(BACKUP_DIR_PATH)


def get_backup_folder_path():
    date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    backup_folder_path = '{0}/influx_backup_{1}'.format(BACKUP_DIR_PATH, date)
    return backup_folder_path

def get_data_start_time():
    return (datetime.datetime.today()- datetime.timedelta(days=BACKUP_INTERVAL)).strftime("%Y-%m-%dT%H:%M:%SZ")

def get_data_end_time():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

def update_backup_log(backup_folder_path, exit_status, start, end):
    new_log_data = {
        "backup_number": 0,
        "backup_timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backup_path": backup_folder_path,
        "exit_status": exit_status,
        "data_start_time": start,
        "data_end_time": end,
    }
    if os.path.exists(backup_log_filename):
        with open(backup_log_filename, 'r+') as f:
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
        with open(backup_log_filename, 'w+') as f:
            json_obj = json.dumps(c, indent=4)
            f.write(json_obj)


def do_backup(backup_folder_path, start, end):
    """Creates a new directory: 'influx_backup_{current date in year-month-day-format},
    creates a backup of the influxdb data generated in the past BACKUP_INTERVAL days,
    and adds a log entry to the log file."""
    # Get and run commands
    make_dir_command = 'mkdir -p {0}'.format(backup_folder_path)
    backup_command = 'influxd backup -portable -start {0} -end {1} {2} '.format(start, end, backup_folder_path)
    os.system(make_dir_command)
    exit_status = os.system(backup_command)
    exit_status = 'SUCCESS' if not exit_status else 'FAILED'
    # Add log entry for this backup.
    update_backup_log(backup_folder_path, exit_status, start, end)
    return exit_status


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