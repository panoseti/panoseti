#! /usr/bin/env python3

"""
Script that creates a backup of the influxdb database. After each backup attempt,
log data is recorded in a file with the following structure:
{

}
"""

import os
import datetime
import json


# Time between backups in days.
BACKUP_INTERVAL = 30
BACKUP_DIR_PATH = './.influx_backups'
backup_log_filename = '{0}/backup_log.json'.format(BACKUP_DIR_PATH)


def get_backup_folder_path():
    date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    backup_folder_path = '{0}/influx_backup_{1}'.format(BACKUP_DIR_PATH, date)
    return backup_folder_path


def update_backup_log(backup_folder_path, exit_status, data_start_time):
    new_log_data = {
        "backup_timestamp": datetime.datetime.now().isoformat(),
        "backup_path": backup_folder_path,
        "exit_status": exit_status,
        "data_start_time": data_start_time,
        "data_end_time": datetime.datetime.now().isoformat()
    }
    if os.path.exists(backup_log_filename):
        print('\t BRANCH 1')
        with open(backup_log_filename, 'r+') as f:
            s = f.read()
            c = json.loads(s)
            c["backups"].append(new_log_data)
            json_obj = json.dumps(c, indent=4)
            print('after dump','\t', json_obj)
            f.seek(0)
            f.write(json_obj)
    else:
        print('\t BRANCH 2')
        c = {
            "backups": [
                new_log_data
            ]
        }
        with open(backup_log_filename, 'w+') as f:
            print(c)
            json_obj = json.dumps(c, indent=4)
            print(json_obj)
            f.write(json_obj)


def do_backup(backup_folder_path):
    """Creates a new directory: 'influx_backup_{current date in year-month-day-format},
    then creates a backup of the entire influxdb database."""
    data_start_time = (datetime.datetime.today() - datetime.timedelta(days=BACKUP_INTERVAL)).isoformat()
    make_dir_command = 'mkdir -p {0}'.format(backup_folder_path)
    backup_command = 'influx backup -portable -start {0} {1} '.format(data_start_time, backup_folder_path)
    os.system(make_dir_command)
    exit_status = 0#os.system(backup_command)

    update_backup_log(backup_folder_path, exit_status, data_start_time)

    return exit_status


def main():
    backup_folder_path = get_backup_folder_path()
    exit_status = do_backup(backup_folder_path)
    if exit_status == 0:
        msg = 'Successfully backed up the database at {0}'.format(backup_folder_path)
        print(msg)
    else:
        msg = 'Failed to back up the database.'
        print(msg)


if __name__ == '__main__':
    main()