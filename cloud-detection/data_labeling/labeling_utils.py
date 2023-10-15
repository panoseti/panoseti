import hashlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Database formats
def get_dataframe_formats():
    dataframe_formats = {
        'user': ['user_uid', 'name'],
        'img': ['img_uid', 'fname', 'type', 'unix_t'],
        'unlabeled_data': ['img_uid', 'is_labeled'],
        'labeled_data': ['img_uid', 'user_uid', 'label']
    }
    return dataframe_formats

def get_dataframe(df_type):
    dataframe_formats = get_dataframe_formats()
    assert df_type in dataframe_formats, f"'{df_type}' is not a supported dataframe format"
    return pd.DataFrame(columns=dataframe_formats[df_type])

def get_df_save_name(user_uid, batch_id, df_type):
    # TODO: reverse order
    return "user-uid_{0}.batch-id_{1}.type_{2}.csv".format(user_uid, batch_id, df_type)

def save_df(user_uid, batch_id, df_type):
    dataframe_formats = get_dataframe_formats()
    for df_type, df_format in dataframe_formats.items():
        df = get_dataframe(df_type)
        df.to_csv(get_df_save_name())

def load_df(batch_id, df_type):
    ...

# Database IO routines



def load_database_formats(data_batch_dir, batch_id):
    ...


# Database interface routines

def get_uid(data: str):
    """Returns SHA1 hash of a string input data."""
    data_bytes = data.encode('utf-8')
    data_uid = hashlib.sha1(data_bytes).hexdigest()
    return data_uid


def add_user(user_df, name, verbose=False):
    """Adds new user to user_df."""
    user_uid = get_uid(name)
    if not user_df.loc[:, 'user_uid'].str.contains(user_uid).any():
        user_df.loc[len(user_df)] = [user_uid, name]
    elif verbose:
        print(f'An entry for "{name}" already exists')
    return user_uid


def add_skycam_img(skycam_df, fname, skycam_type, timestamp, verbose=False):
    img_uid = get_uid(fname)
    if not skycam_df.loc[:, 'img_uid'].str.contains(img_uid).any():
        skycam_df.loc[len(skycam_df)] = [img_uid, fname, skycam_type, timestamp]
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


# name = "Nicolas Rault-Wang"
# batch_id = 0
#
# #user_uid = add_user(user_df, name)
# session = LabelSession('cloud-detection', name, batch_id)
# session.start()