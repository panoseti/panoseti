import hashlib
import pandas as pd
import os
import shutil

# Database formats

def get_dataframe_formats():
    dataframe_formats = {
        'user': {
            'columns': ['user_uid', 'name'],
        },
        'img': {
            # batch_data_subdir is defined as the dir relative to the data batch with id 'batch_id' containing the img file.
            'columns': ['img_uid', 'fname', 'unix_t', 'camera_type', 'batch_id', 'batch_data_subdir'],
        },
        'unlabeled': {
            'columns': ['img_uid', 'is_labeled'],
        },
        'labeled': {
            'columns': ['img_uid', 'user_uid', 'label'],
        },
        'user-batch-log': {
            'columns': ['user_uid', 'batch_id'],
        }
    }
    return dataframe_formats

# Database IO routines
def get_data_export_dir(task, batch_id, user_uid, root):
    dir_name = "task_{0}.batch-id_{1}.user-uid_{2}".format(task, batch_id, user_uid)
    dir_path = f'{root}/{dir_name}'
    return dir_path

def get_df_save_name(task, batch_id, df_type, user_uid, is_temp):
    if user_uid is not None:
        save_name = "task_{0}.batch-id_{1}.type_{2}.user-uid_{3}".format(task, batch_id, df_type, user_uid)
    else:
        save_name = "task_{0}.batch-id_{1}.type_{2}".format(task, batch_id, df_type)
    if is_temp:
        save_name += f'.TEMP'
    return save_name + '.csv'


def save_df(df, df_type, user_uid, batch_id, task, is_temp, save_dir, overwrite_ok=True):
    """Save a pandas dataframe as a csv file. If is_temp is True, add the postfix TEMP to the filename."""
    #os.makedirs(save_dir, exist_ok=True)

    df_path = f'{save_dir}/{get_df_save_name(task, df_type, batch_id, user_uid, is_temp)}'
    if os.path.exists(df_path) and not overwrite_ok:
        raise FileExistsError(f'{df_path} exists. Aborting save.')
    else:
        with open(df_path, 'w'):
            df.to_csv(df_path)


def load_df(user_uid, batch_id, df_type, task, is_temp, save_dir):
    df_path = f'{save_dir}/{get_df_save_name(task, df_type, batch_id, user_uid, is_temp)}'
    if os.path.exists(df_path):
        with open(df_path, 'r') as f:
            df = pd.read_csv(f, index_col=0)
            return df
    else:
        return None

def extend_df(df: pd.DataFrame, df_type: str, data: dict, verify_columns=False, require_complete=True, verify_dtype=True):
    """Extends a df of df_type with entries in the data dictionary.
    If verify_columns=True, raise an error if the given keys do not match the
    Missing entries are filled with NaNs.
    If require_complete=True, raise a KeyError if the given data does not contain all keys specified in
    the df_format for df_type.
    """
    df_format = get_dataframe_formats()[df_type]
    if verify_columns:
        for column in data:
            if column not in df_format['columns']:
                raise KeyError(f"'{column}' is not a valid key for {df_type}. Valid keys are {df_format['columns']}.")
    if require_complete:
        for column in df_format['columns']:
            if column not in data:
                raise KeyError(f"An entry for '{column}' is required for {df_type}, but was not provided in the given data.")
    df_extended = pd.DataFrame.from_dict(data)
    if verify_dtype and len(df) > 0:
        if df.dtypes.tolist() != df_extended.dtypes.tolist():
            raise ValueError(f"dtypes of new data do not match dtypes of existing columns."
                             f"\nGiven dtypes:\n{df_extended.dtypes} \nExpected dtypes: \n{df.dtypes}")
    return pd.concat([df, df_extended], keys=df_format, ignore_index=True)



# Database interface routines

def get_uid(data: str):
    """Returns SHA1 hash of a string input data."""
    data_bytes = data.encode('utf-8')
    data_uid = hashlib.sha1(data_bytes).hexdigest()
    return data_uid


def add_user(user_df, user_uid, name, verbose=False):
    """Adds new user to user_df."""
    if not user_df.loc[:, 'user_uid'].str.contains(user_uid).any():
        data = {
            'user_uid': [user_uid],
            'name': [name]
        }
        extend_df(user_df, 'user', data)
        #user_df.loc[len(user_df)] = [user_uid, name]
    elif verbose:
        print(f'An entry for "{name}" already exists')
    return user_uid

def add_user_batch_log(ubl_df, user_uid, batch_id, verbose=False):
    """Adds new (user-uid, batch-id) entry to user_df."""
    if not (ubl_df.loc[ubl_df['user_uid'] == user_uid, 'batch_id'] == batch_id).any():
        data = {
            'user_uid': [user_uid],
            'batch_id': [batch_id]
        }
        extend_df(ubl_df, 'user-batch-log', data)
        #ubl_df.loc[len(ubl_df)] = [user_uid, batch_id]
    elif verbose:
        print(f'An entry for "{batch_id}" already exists')
    return user_uid


def add_skycam_img(img_df, fname, skycam_type, timestamp, batch_id, batch_data_subdir, verbose=False):
    """Add a skycamera img to the img_df."""
    img_uid = get_uid(fname)
    if not img_df.loc[:, 'img_uid'].str.contains(img_uid).any():
        data = {
            'img_uid': [img_uid],
            'fname': [fname],
            'unix_t': [timestamp],
            'camera_type': [skycam_type],
            'batch_id': [batch_id],
            'batch_data_subdir': [batch_data_subdir]
        }
        extend_df(img_df, 'img', data)
        #img_df.loc[len(img_df)] = [img_uid, fname, timestamp, skycam_type]
    elif verbose:
        print(f'An entry for "{fname}" already exists')
    return img_uid


def add_unlabeled_data(unlabeled_df, img_uid, verbose=False):
    if not unlabeled_df.loc[:, 'img_uid'].str.contains(img_uid).any():
        data = {
            'img_uid': [img_uid],
            'is_labeled': [False]
        }
        extend_df(unlabeled_df, 'unlabeled', data)
        #unlabeled_df.loc[len(unlabeled_df)] = [img_uid, False]
    elif verbose:
        print(f'An entry for "{img_uid}" already exists')


def add_labeled_data(labeled_df, unlabeled_df, img_uid, user_uid, label):
    # labeled_df.loc[(labeled_df['img_uid'] == img_uid), ['user_uid', 'label']] = [user_uid, label]
    data = {
        'img_uid': [img_uid],
        'user_uid': [user_uid],
        'label': [label]
    }
    extend_df(labeled_df, 'labeled', data, verify_columns=True)
    # labeled_df.loc[len(labeled_df)] = [img_uid, user_uid, label]
    unlabeled_df.loc[(unlabeled_df['img_uid'] == img_uid), 'is_labeled'] = True


def get_dataframe(df_type):
    dataframe_formats = get_dataframe_formats()
    assert df_type in dataframe_formats, f"'{df_type}' is not a supported dataframe format"
    return pd.DataFrame(columns=dataframe_formats[df_type]['columns'])


# Unzip files
def unpack_batch_data(batch_data_root_dir='batch_data'):
    """Unpack image files from batch data gztar file."""
    downloaded_fname = ''
    batch_dir = ''

    for fname in os.listdir(batch_data_root_dir):
        if fname.endswith('.tar.gz'):
            downloaded_fname = fname
            batch_dir = fname.rstrip('.tar.gz')
    if downloaded_fname:
        downloaded_fpath = f'{batch_data_root_dir}/{downloaded_fname}'
        batch_dir_path = f'{batch_data_root_dir}/{batch_dir}'
        print(f'Unzipping {downloaded_fpath}. This may take a minute...')
        shutil.unpack_archive(downloaded_fpath, batch_dir_path, 'gztar')
        os.remove(downloaded_fpath)

"""
pd.set_option('display.max_columns', None)
data = {'user_uid': ['23423'], 'img_uid': ['123423423'], 'label': [0]}
df = pd.read_csv('/Users/nico/panoseti/panoseti-software/cloud-detection/data_labeling/dataset_cloud-detection/user_labeled_batches/task_cloud-detection.batch-id_0.user-uid_2a8f7d4a0de094708b0e1b74645c4dcfd789068f/task_cloud-detection.batch-id_labeled.type_0.user-uid_2a8f7d4a0de094708b0e1b74645c4dcfd789068f.csv', index_col=0)
#print(df)
print('type', type(df.dtypes.tolist()[0]))
# for key in df.columns:
print(extend_df(df, 'labeled', data, verify_columns=True))
"""

