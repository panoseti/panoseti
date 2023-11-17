import hashlib
import pandas as pd
import numpy as np
import os
import shutil


skycam_imgs_root_dir = 'skycam_imgs'
pano_imgs_root_dir = 'pano_imgs'
batch_data_root_dir = 'batch_data'

data_labels_fname = 'skycam_labels.json'
pano_path_index_fname = 'pano_path_index.json'
skycam_path_index_fname = 'skycam_path_index.json'

# Database formats / Core routines


def get_dataframe_formats():
    dataframe_formats = {
        'user': {
            'columns': ['user_uid', 'name'],
        },
        'skycam': {
            # batch_data_subdir is defined as the dir relative to the data batch with id 'batch_id' containing the img file.
            'columns': ['skycam_uid', 'batch_id', 'skycam_dir', 'fname', 'unix_t', 'skycam_type'],
        },
        'pano': {
            'columns': ['pano_uid', 'batch_id', 'run_dir', 'fname', 'frame_offset', 'module_id', 'frame_unix_t']
        },
        'feature': {
            'columns': ['feature_uid', 'skycam_uid', 'pano_uid', 'batch_id']
        },
        'unlabeled': {
            'columns': ['feature_uid', 'is_labeled'],
        },
        'labeled': {
            'columns': ['feature_uid', 'user_uid', 'label'],
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

    df_path = f'{save_dir}/{get_df_save_name(task, batch_id, df_type, user_uid, is_temp)}'
    if os.path.exists(df_path) and not overwrite_ok:
        raise FileExistsError(f'{df_path} exists. Aborting save.')
    else:
        with open(df_path, 'w'):
            df.to_csv(df_path)


def load_df(user_uid, batch_id, df_type, task, is_temp, save_dir):
    df_path = f'{save_dir}/{get_df_save_name(task, batch_id, df_type, user_uid, is_temp)}'
    if os.path.exists(df_path):
        with open(df_path, 'r') as f:
            df = pd.read_csv(f, index_col=0)
            return df
    else:
        return None


def extend_df(df: pd.DataFrame, df_type: str, data: dict,
              verify_columns=False, require_complete=True, verify_dtype=True, verify_integrity=False):
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
    if verify_dtype:
        if len(df) == 0:
            # If the df is empty, define column dtypes based on first datapoint.
            for x in df_extended.columns:
                if x in df.columns:
                    df[x] = df[x].astype(df_extended[x].dtypes.name)
        else:
            if df.dtypes.to_dict() != df_extended.dtypes.to_dict():
                raise ValueError(f"dtypes of new data do not match dtypes of existing columns."
                                 f"\nGiven dtypes:\n{df_extended.dtypes} \nExpected dtypes: \n{df.dtypes}")
    return pd.concat([df, df_extended], ignore_index=True, verify_integrity=verify_integrity)


# UID definitions

def get_uid(data: str):
    """Returns SHA1 hash of a string input data."""
    data_bytes = data.encode('utf-8')
    data_uid = hashlib.sha1(data_bytes).hexdigest()
    return data_uid


def get_pano_uid(pano_original_fname, frame_offset):
    return get_uid(pano_original_fname + str(frame_offset))

def get_feature_uid(skycam_uid, pano_uid, batch_id):
    return get_uid(skycam_uid + pano_uid + str(batch_id))

def get_skycam_uid(original_skycam_fname):
    return get_uid(original_skycam_fname)


# Database interface routines

def add_user(user_df, user_uid, name, verbose=False):
    """Returns a df with user added."""
    if not user_df.loc[:, 'user_uid'].str.contains(user_uid).any():
        data = {
            'user_uid': [user_uid],
            'name': [name]
        }
        return extend_df(user_df, 'user', data)
        #user_df.loc[len(user_df)] = [user_uid, name]
    raise ValueError(f'An entry for "{name}" already exists')

def add_user_batch_log(ubl_df, user_uid, batch_id, verbose=False):
    """Adds new (user-uid, batch-id) entry to user_df."""
    if not (ubl_df.loc[ubl_df['user_uid'] == user_uid, 'batch_id'] == batch_id).any():
        data = {
            'user_uid': [user_uid],
            'batch_id': [batch_id]
        }
        return extend_df(ubl_df, 'user-batch-log', data)
        #ubl_df.loc[len(ubl_df)] = [user_uid, batch_id]
    raise ValueError(f'An entry for "{batch_id}" already exists')


def add_skycam_img(skycam_df, skycam_uid, original_fname, skycam_type, timestamp, batch_id, skycam_dir, verbose=False):
    """Add a skycamera img to skycam_df."""
    if not skycam_df.loc[:, 'skycam_uid'].str.contains(skycam_uid).any():
        data = {
            'skycam_uid': [skycam_uid],
            'batch_id': [batch_id],
            'skycam_dir': [skycam_dir],
            'fname': [original_fname],
            'unix_t': [timestamp],
            'skycam_type': [skycam_type],
        }
        return extend_df(skycam_df, 'skycam', data)
        #img_df.loc[len(img_df)] = [skycam_uid, fname, timestamp, skycam_type]
    raise ValueError(f'An entry for "{original_fname}" already exists')

def add_pano_img(pano_df, pano_uid, run_dir, fname, frame_offset, module_id, unix_t, batch_id, verbose=False):
    """Add a panoseti module img to pano_df."""
    if not pano_df.loc[:, 'pano_uid'].str.contains(pano_uid).any():
        data = {
            'pano_uid': [pano_uid],
            'batch_id': [batch_id],
            'run_dir': [run_dir],
            'fname': [fname],
            'frame_offset': [frame_offset],
            'module_id': [module_id],
            'frame_unix_t': [unix_t]
        }
        return extend_df(pano_df, 'pano', data)
    return pano_df
    # raise ValueError(f'An entry for "{fname}" already exists')

def add_feature_entry(feature_df, skycam_uid, pano_uid, batch_id, verbose=False):
    feature_uid = get_feature_uid(skycam_uid, pano_uid, batch_id)
    if not feature_df.loc[:, 'feature_uid'].str.contains(feature_uid).any():
        data = {
            'feature_uid': [feature_uid],
            'skycam_uid': [skycam_uid],
            'pano_uid': [pano_uid],
            'batch_id': [batch_id]
        }
        return extend_df(feature_df, 'feature', data)
    raise ValueError(f'An entry for "{feature_uid}" already exists')


def add_unlabeled_data(unlabeled_df, feature_uid, verbose=False):
    if not unlabeled_df.loc[:, 'feature_uid'].str.contains(feature_uid).any():
        data = {
            'feature_uid': [feature_uid],
            'is_labeled': [False]
        }
        return extend_df(unlabeled_df, 'unlabeled', data)
        #unlabeled_df.loc[len(unlabeled_df)] = [feature_uid, False]
    raise ValueError(f'An entry for "{feature_uid}" already exists')


def add_labeled_data(labeled_df, unlabeled_df, feature_uid, user_uid, label):
    # labeled_df.loc[(labeled_df['feature_uid'] == feature_uid), ['user_uid', 'label']] = [user_uid, label]
    data = {
        'feature_uid': [feature_uid],
        'user_uid': [user_uid],
        'label': [label]
    }
    extended_df = extend_df(labeled_df, 'labeled', data, verify_columns=True)
    # labeled_df.loc[len(labeled_df)] = [feature_uid, user_uid, label]
    unlabeled_df.loc[(unlabeled_df['feature_uid'] == feature_uid), 'is_labeled'] = True
    return extended_df


def get_dataframe(df_type):
    dataframe_formats = get_dataframe_formats()
    assert df_type in dataframe_formats, f"'{df_type}' is not a supported dataframe format"
    return pd.DataFrame(columns=dataframe_formats[df_type]['columns'])


# Unzip files
def unpack_batch_data(batch_data_root_dir):
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

# Misc




"""
pd.set_option('display.max_columns', None)
data = {'user_uid': ['23423'], 'feature_uid': ['123423423'], 'label': [0]}
df = pd.read_csv('/Users/nico/panoseti/panoseti-software/cloud-detection/data_labeling/dataset_cloud-detection/user_labeled_batches/task_cloud-detection.batch-id_0.user-uid_2a8f7d4a0de094708b0e1b74645c4dcfd789068f/task_cloud-detection.batch-id_labeled.type_0.user-uid_2a8f7d4a0de094708b0e1b74645c4dcfd789068f.csv', index_col=0)
#print(df)
print('type', type(df.dtypes.tolist()[0]))
# for key in df.columns:
print(extend_df(df, 'labeled', data, verify_columns=True))
"""

