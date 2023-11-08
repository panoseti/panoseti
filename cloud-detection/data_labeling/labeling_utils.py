import hashlib
import pandas as pd
import os
import shutil

# Database formats

def get_dataframe_formats():
    dataframe_formats = {
        'user': ['user_uid', 'name'],
        'img': ['img_uid', 'fname', 'unix_t', 'type'],
        'unlabeled': ['img_uid', 'is_labeled'],
        'labeled': ['img_uid', 'user_uid', 'label'],
        'user-batch-log': ['user_uid', 'batch_id']
    }
    return dataframe_formats

# Database IO routines
def get_data_export_dir(task, batch_id, user_uid, root):
    dir_name = "task_{0}.batch-id_{1}.user-uid_{2}".format(task, batch_id, user_uid)
    dir_path = f'{root}/{dir_name}'
    return dir_path

def get_batch_label_dir(task, batch_id, root):
    dir_name = "task_{0}.batch-id_{1}".format(task, batch_id)
    dir_path = f'{root}/{dir_name}'
    return dir_path

def get_df_save_name(df_type, user_uid, is_temp):
    save_name = "type_{0}.user-uid_{1}".format(df_type, user_uid)
    if is_temp:
        save_name += f'.TEMP'
    return save_name + '.csv'


def save_df(df, df_type, user_uid, batch_id, task, is_temp, batch_label_dir, overwrite_ok=True):
    """Save a pandas dataframe as a csv file. If is_temp is True, add the postfix TEMP to the filename."""
    os.makedirs(batch_label_dir, exist_ok=True)

    df_path = f'{batch_label_dir}/{get_df_save_name(df_type, user_uid, is_temp)}'
    if os.path.exists(df_path) and not overwrite_ok:
        raise FileExistsError(f'{df_path} exists. Aborting save.')
    else:
        with open(df_path, 'w'):
            df.to_csv(df_path)


def load_df(user_uid, batch_id, df_type, task, is_temp, root='batch_labels', from_export=False):
    if from_export:
        batch_label_dir = get_data_export_dir(task, batch_id, user_uid, root)
    else:
        batch_label_dir = get_batch_label_dir(task, batch_id, root)
    df_path = f'{batch_label_dir}/{get_df_save_name(df_type, user_uid, is_temp)}'
    print(df_path)
    if os.path.exists(df_path):
        with open(df_path, 'r') as f:
            df = pd.read_csv(f, index_col=0)
            return df
    else:
        return None


# Database interface routines

def get_uid(data: str):
    """Returns SHA1 hash of a string input data."""
    data_bytes = data.encode('utf-8')
    data_uid = hashlib.sha1(data_bytes).hexdigest()
    return data_uid


def add_user(user_df, user_uid, name, verbose=False):
    """Adds new user to user_df."""
    if not user_df.loc[:, 'user_uid'].str.contains(user_uid).any():
        user_df.loc[len(user_df)] = [user_uid, name]
    elif verbose:
        print(f'An entry for "{name}" already exists')
    return user_uid

def add_user_batch_log(ubl_df, user_uid, batch_id, verbose=False):
    """Adds new (user-uid, batch-id) entry to user_df."""
    if not (ubl_df.loc[ubl_df['user_uid'] == user_uid, 'batch_id'] == batch_id).any():
        ubl_df.loc[len(ubl_df)] = [user_uid, batch_id]
    elif verbose:
        print(f'An entry for "{batch_id}" already exists')
    return user_uid


def add_skycam_img(skycam_df, fname, skycam_type, timestamp, verbose=False):
    img_uid = get_uid(fname)
    if not skycam_df.loc[:, 'img_uid'].str.contains(img_uid).any():
        skycam_df.loc[len(skycam_df)] = [img_uid, fname, timestamp, skycam_type]
    elif verbose:
        print(f'An entry for "{fname}" already exists')
    return img_uid


def add_unlabeled_data(unlabeled_df, img_uid, verbose=False):
    if not unlabeled_df.loc[:, 'img_uid'].str.contains(img_uid).any():
        unlabeled_df.loc[len(unlabeled_df)] = [img_uid, False]
    elif verbose:
        print(f'An entry for "{img_uid}" already exists')


def add_labeled_data(labeled_df, unlabeled_df, img_uid, user_uid, label):
    # labeled_df.loc[(labeled_df['img_uid'] == img_uid), ['user_uid', 'label']] = [user_uid, label]
    labeled_df.loc[len(labeled_df)] = [img_uid, user_uid, label]
    unlabeled_df.loc[(unlabeled_df['img_uid'] == img_uid), 'is_labeled'] = True

def get_dataframe(df_type):
    dataframe_formats = get_dataframe_formats()
    assert df_type in dataframe_formats, f"'{df_type}' is not a supported dataframe format"
    return pd.DataFrame(columns=dataframe_formats[df_type])


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
