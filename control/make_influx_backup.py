#! /usr/bin/env python3

"""
Script that creates a backup of the influxdb database.
"""

import os
import datetime


BACKUP_DIR_PATH = '~/influx_backups'
# Logs all backup activity.
BACKUP_LOG = 'influxdb_backups.json'


def get_backup_folder_path():
    date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    backup_folder_path = '{0}/influx_backup_{1}'.format(BACKUP_DIR_PATH, date)
    return backup_folder_path


def do_backup(backup_folder_path):
    """Creates a new directory: 'influx_backup_{current date in year-month-day-format},
    then creates a backup of the entire influxdb database."""
    make_backup_dir_command = 'mkdir {0}'.format(backup_folder_path)
    backup_command = 'influx backup -portable {0}'.format(backup_folder_path)
    exit_status = os.system(make_backup_dir_command), os.system(backup_command)
    return exit_status


def main():
    backup_folder_path = get_backup_folder_path()
    exit_status = do_backup()
    if exit_status == (0, 0):
        msg = 'Successfully backed up the database at {0}'.format(backup_folder_path)
        print(msg)
    else:
        msg = 'Failed to back up the database.'
        print(msg)


if __name__ == '__main__':
    main()