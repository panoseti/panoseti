#! /usr/bin/env python3

"""
Script that creates a backup of the influxdb database. After each backup attempt,
log data is recorded in a file with the following structure:
{
    "backups: [
        {Backup log entry},
    ]
}

This script can be run on the 1st and 15th of every month at 9am with a cronjob:
With email notifications:
MAILTO="" # Add email address
0 9 1,15 * * /path/to/this/script/backup_influxdb.py --backup

Without email notifications:
0 9 1,15 * * /path/to/this/script/backup_influxdb.py --backup >/dev/null 2>&1
"""

import datetime
import json
import os
import sys
import time

# Globals
BACKUP_DIR_PATH = '/tmp/influxdb_backups'
RESTORE_DIR_PATH = BACKUP_DIR_PATH
backup_log_filename = 'backup_log.json'
backup_log_path = f'{BACKUP_DIR_PATH}/{backup_log_filename}'


def get_backup_folder_path(date: datetime.datetime) -> str:
    backup_folder_path = '{}/influx_backup_{}'.format(BACKUP_DIR_PATH, date.strftime("%Y_%m_%dT%H_%M_%SZ"))
    return backup_folder_path


def get_last_backup_date() -> str | None:
    """Returns the last backup date, or None if this is the first backup."""
    last_backup_date = None
    if os.path.exists(backup_log_path):
        with open(backup_log_path) as f:
            s = f.read()
            c = json.loads(s)
            last_backup_date = str(c["backups"][-1]["timestamp"])
    return last_backup_date


def update_backup_log(backup_folder_path: str, date: datetime.datetime, exit_status: str) -> None:
    """Updates the backup json file (creating it if necessary) with a new log entry."""
    new_log_data = {
        "backup_number": 0,
        "backup_path": os.path.abspath(backup_folder_path),
        "timestamp": date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backup_status": exit_status,
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


def do_backup() -> None:
    """
    1. Creates a new directory: 'influx_backup_{current date in year-month-day-format}',
    2. Creates a backup of the influxdb data generated since the last backup, and
    3. Calls update_backup_log to add a log entry for this backup.
    """
    date = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    backup_folder_path = get_backup_folder_path(date)
    # Create backup directory.
    make_dir_command = f'mkdir -p {backup_folder_path}'
    os.system(make_dir_command)
    # Get and run backup command.
    last_backup_date = get_last_backup_date()
    if last_backup_date:
        backup_command = f'influxd backup -portable -db metadata -since {last_backup_date} {backup_folder_path} '
    else:
        backup_command = f'influxd backup -portable -db metadata {backup_folder_path}'
    exit_status = os.system(backup_command)
    status_msg = 'SUCCESS' if not exit_status else 'FAILED'
    # Add log entry for this backup.
    update_backup_log(backup_folder_path, date, status_msg)
    # Report success or failure of backup.
    if status_msg == 'SUCCESS':
        msg = f'Successfully backed up the database to {backup_folder_path}.'
        print(msg)
    else:
        msg = 'Failed to back up the database.'
        print(msg)


def restore_one_backup(path_to_backup: str) -> None:
    """
    Restores one backup. Note that InfluxDB does not allow us to directly restore backups to
     an existing database, so we must do the following:
        1) Restore the backup to a temporary database,
        2) Write the data from the temporary database into the metadata database, and
        3) Delete the temporary database.
    """
    print("Restoring the backup to the temporary database 'metadata-tmp'...")
    command_1 = f'influxd restore -portable -db "metadata" -newdb "metadata-tmp" {path_to_backup}'
    os.system(command_1)

    print("Querying data from 'metadata-tmp' and writing it into 'metadata'...")
    time.sleep(1)
    command_2 = '''influx -execute 'SELECT * INTO "metadata".autogen.:MEASUREMENT FROM "metadata-tmp".autogen./.*/ GROUP BY *' '''
    os.system(command_2)

    print("Deleting 'metadata-tmp'...")
    command_3 = '''influx -execute 'DROP DATABASE "metadata-tmp"' '''
    os.system(command_3)


def do_restore() -> None:
    """Restore the metadata database from the backups in the specified restore directory."""
    try:
        backup_directories = [name for name in os.listdir(RESTORE_DIR_PATH)]
        backup_directories.remove(backup_log_filename)
    except FileNotFoundError as ferr:
        msg = "backup_influxdb.py: {0}\n\t{1} may not exist or does not contain a usable backup directory."
        msg += "\n\tPlease assign RESTORE_DIR_PATH to a different path."
        msg += "\n\tError msg: {2}\n"
        print(msg.format(datetime.datetime.now(), RESTORE_DIR_PATH, ferr))
        raise
    create_db_command = '''influx -execute 'CREATE DATABASE "metadata"' '''
    os.system(create_db_command)

    print('Attempting to restore the following backups:')
    for name in backup_directories:
        print(f'\t* {name}')

    for name in backup_directories:
        print('\n\n\t' + '**' * 3, name, '**' * 3)
        path_to_backup = f'{RESTORE_DIR_PATH}/{name}'
        restore_one_backup(path_to_backup)
    print("\nRestored all backups.")


def usage() -> None:
    msg = 'Usage:'
    msg += '\n\t--backup\tcreate a backup of the influxdb data since the last update and save it in the directory specified by BACKUP_DIR_PATH.'
    msg += '\n\t--restore\trestore the metadata database from backups in the directory specified by RESTORE_DIR_PATH.'
    print(msg)

# Facilitates command-line use.
if __name__ == '__main__':
    argv = sys.argv
    op = ''
    nops = 0
    i = 1
    while i < len(argv):
        if argv[i] == '--backup':
            nops += 1
            op = 'backup'
        elif argv[i] == '--restore':
            nops += 1
            op = 'restore'
        i += 1
    if nops == 0:
        usage()
    elif nops > 1:
        print('must specify a single op')
        usage()
    else:
        if op == 'backup':
            do_backup()
        elif op == 'restore':
            do_restore()
