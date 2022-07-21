#! /usr/bin/env python3

"""
Script that creates a backup of the influxdb database. After each backup attempt,
log data is recorded in a file with the following structure:
{
    "backups: [
        {Backup log entry},
    ]
}
"""

import os, sys
import datetime
import json
import time


BACKUP_DIR_PATH = '/tmp/influxdb_backups'
RESTORE_DIR_PATH = BACKUP_DIR_PATH
backup_log_filename = 'backup_log.json'
backup_log_path = '{0}/{1}'.format(BACKUP_DIR_PATH, backup_log_filename)


def get_backup_folder_path(date):
    backup_folder_path = '{0}/influx_backup_{1}'.format(BACKUP_DIR_PATH, date.strftime("%Y_%m_%dT%H_%M_%SZ"))
    return backup_folder_path


def get_last_backup_date():
    last_backup_date = None
    if os.path.exists(backup_log_path):
        with open(backup_log_path) as f:
            s = f.read()
            c = json.loads(s)
            last_backup_date = c["backups"][-1]["timestamp"]
    return last_backup_date


def update_backup_log(backup_folder_path, date, exit_status):
    """
    Updates the json file (creating it if necessary) and storing backup log data
    """
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


def do_backup():
    """
    1. Creates a new directory: 'influx_backup_{current date in year-month-day-format},
    2. Creates a backup of the influxdb data generated since the last backup, and
    3. Adds a log entry to the log file.
    """
    # Get and run commands
    date = datetime.datetime.utcnow()
    backup_folder_path = get_backup_folder_path(date)

    make_dir_command = 'mkdir -p {0}'.format(backup_folder_path)
    os.system(make_dir_command)
    last_backup_date = get_last_backup_date()
    if last_backup_date:
        print("BRANCH 1")
        backup_command = 'influxd backup -portable -db metadata -since {0} {1} '.format(last_backup_date, backup_folder_path)
    else:
        backup_command = 'influxd backup -portable -db metadata {0}'.format(backup_folder_path)
    exit_status = os.system(backup_command)
    exit_status = 'SUCCESS' if not exit_status else 'FAILED'
    # Add log entry for this backup.
    update_backup_log(backup_folder_path, date, exit_status)

    if exit_status == 'SUCCESS':
        msg = 'Successfully backed up the database at {0}'.format(backup_folder_path)
        print(msg)
    else:
        msg = 'Failed to back up the database.'
        print(msg)


def restore_one_backup(path_to_backup):
    # Restore backup to a temporary database 'metadata-tmp'
    print('COMMAND 1')
    restore_to_tmp_command = 'influxd restore -portable -db "metadata" -newdb "metadata-tmp" {0}'.format(path_to_backup)
    os.system(restore_to_tmp_command)
    # Copy data from 'metadata-tmp' and write it into 'metadata'
    print('COMMAND 2')
    time.sleep(1)
    copy_from_tmp_command = '''influx -execute 'SELECT * INTO "metadata".autogen.:MEASUREMENT FROM "metadata-tmp".autogen./.*/ GROUP BY *' '''
    os.system(copy_from_tmp_command)
    # Delete temporary database
    #time.sleep(5)
    drop_tmp_db_command = '''influx -execute 'DROP DATABASE "metadata-tmp"' '''
    os.system(drop_tmp_db_command)


def do_restore():
    """
    Restore the metadata database using the files stored in the restore directory.
    """
    try:
        backup_directories = [name for name in os.listdir(RESTORE_DIR_PATH)]
    except FileNotFoundError as ferr:
        msg = "backup_influxdb.py: {0}\n\t{1} may not exist or does not contain a usable backup directory."
        msg += "\n\tPlease assign RESTORE_DIR_PATH to a different path."
        msg += "\n\tError msg: {2}\n"
        print(msg.format(datetime.datetime.now(), RESTORE_DIR_PATH, ferr))
        raise
    backup_directories.remove(backup_log_filename)
    print('Attempting to restore the following backups:', backup_directories)
    create_db_command = '''influx -execute 'CREATE DATABASE "metadata"' '''
    os.system(create_db_command)
    for name in backup_directories:
        print('\n\t' + '**' * 3, name, '**' * 3)
        path_to_backup = '{0}/{1}'.format(RESTORE_DIR_PATH, name)
        restore_one_backup(path_to_backup)
    print("Restored all backups.")


def usage():
    print('''Usage:
    --backup\t\t create a backup of the influxdb data since the last update, and write it to the directory specified by BACKUP_DIR_PATH.
    --restore\t\t restore the metadata database from backups stored in the directory specified by RESTORE_DIR_PATH.
    ''')


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
