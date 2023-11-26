
"""
Class for building batch data skycam features.
"""

import os
from dataframe_utils import add_skycam_img
from preprocess_skycam import *
from fetch_skycam_imgs import *
from batch_building_utils import *


class SkycamBatchBuilder(SkycamBatchDataFileTree):
    def __init__(self, task, batch_id, skycam_type, year, month, day, verbose=True, force_recreate=False):

        # self.task = task
        # self.batch_id = batch_id
        # self.batch_path = batch_path
        #
        # self.skycam_root_path = get_skycam_root_path(batch_path)
        # self.skycam_dir = get_skycam_dir(skycam_type, year, month, day)
        # self.skycam_path = f'{self.skycam_root_path}/{self.skycam_dir}'
        # self.skycam_subdirs = get_skycam_subdirs(self.skycam_path)
        self.skycam_meta = {
            'skycam_type': skycam_type,
            'year': year,
            'month': month,
            'day': day
        }
        super().__init__(task, batch_id, get_skycam_dir(**self.skycam_meta))

        self.verbose = verbose
        self.force_recreate = force_recreate

        self.init_skycam_subdirs()

    def init_skycam_subdirs(self):
        """Initialize pre-processing directories."""
        # if self.force_recreate:
        #     shutil.rmtree(self.skycam_path)
        for dir_name in self.skycam_subdirs.values():
            os.makedirs(dir_name, exist_ok=True)
        # if not self.force_recreate:
        #     self.is_uninitialized()

    """Utility functions"""
    @staticmethod
    def get_skycam_img_time(skycam_fname):
        """Returns datetime object based on the image timestamp contained in skycam_fname."""
        if skycam_fname[-4:] != '.jpg':
            raise Warning('Expected a .jpg file')
        # Example: SC2_20230625190102 -> SC2, 20230625190102
        skycam_type, t = skycam_fname[:-4].split('_')
        # 20230625190102 -> 2023, 06, 25, 19, 01, 02
        time_fields = t[0:4], t[4:6], t[6:8], t[8:10], t[10:12], t[12:14]
        year, month, day, hour, minute, second = [int(tf) for tf in time_fields]

        dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        tz = timezone(timedelta(hours=0))
        dt = dt.astimezone(tz)
        return dt

    """State checking routines"""

    def is_uninitialized(self):
        if os.path.exists(self.skycam_path) and len(os.listdir()) > 0:
            is_initialized = False
            for path in os.listdir():
                if path in self.skycam_subdirs:
                    is_initialized |= len(os.listdir()) > 0
                if os.path.isfile(path):
                    is_initialized = False
            if is_initialized:
                raise FileExistsError(
                    f"Expected directory {self.skycam_path} to be uninitialized, but found the following files:\n\t"
                    f"{os.walk(self.skycam_path)}")

    def skycam_data_downloaded(self):
        """Returns True iff data is already downloaded."""
        if os.path.exists(self.skycam_subdirs['original']) and len(os.listdir(self.skycam_subdirs['original'])) > 0:
            return True
        return False

    def skycam_features_created(self):
        """Returns True iff skycam features are already created."""
        ret = False
        if os.path.exists(f'{self.batch_path}/{self.skycam_path_index_fname}'):
            ret = True
        for skycam_subdir in self.skycam_subdirs.values():
            ret &= os.path.exists(skycam_subdir) and len(os.listdir(skycam_subdir)) > 0
        return ret

    def skycam_zipfile_downloaded(self):
        """Returns True iff a skycam zip file exists in self.skycam_path"""
        downloaded_fname = ''
        for fname in os.listdir(self.skycam_path):
            if fname.endswith('.tar.gz'):
                downloaded_fname = fname
        return len(downloaded_fname) > 0

    """Skycam original image download"""

    def get_manual_download_instructions(self):
        skycam_link = get_skycam_link(**self.skycam_meta)
        msg = (f'To manually download the data, please do the following:\n'
               f'\t0. Set manual_skycam_download=True in your call to the build_batch function\n'
               f'\t1. Visit {skycam_link}\n'
               f'\t2. Click the blue "all" text\n'
               f'\t3. Uncheck the .mp4 and .mpg files (should be near the top)\n'
               f'\t4. Click the "Download tarball" button and wait for the download to finish\n\n'
               f'\t(Please DO NOT unzip the file)\n\n'
               f'\t5. Move the downloaded .tar.gz file to \n'
               f'\t\t{os.path.abspath(self.skycam_path)}\n'
               f'\t6. Rerun make_batch.py')
        return msg

    def manual_skycam_download(self):
        """Raises FileNotFoundError if skycam zipfile is not downloaded"""
        if not self.skycam_zipfile_downloaded():
            msg = 'Automatic skycam download is disabled and no skycam data found. \n'
            msg += self.get_manual_download_instructions()
            raise FileNotFoundError(msg)

    def automatic_skycam_download(self):
        try:
            download_skycam_data(**self.skycam_meta, verbose=self.verbose, skycam_path=self.skycam_path)
            if not self.skycam_zipfile_downloaded():
                raise FileNotFoundError(
                    f"Failed to retrieve skycam images! Skycam zipfile missing from {os.path.abspath(self.skycam_path)}.")
        except Warning as wee:
            msg = str(wee)
            msg += self.get_manual_download_instructions()
            raise Warning(wee)
        except Exception as e:
            msg = f'\n\n\nOriginal error: "{str(e)}"\n\n'
            msg += "Failed to automatically download skycam data.\n"
            msg += self.get_manual_download_instructions()
            raise Exception(msg)

    def filter_images(self, first_t: datetime, last_t: datetime):
        """Remove skycam images between t_start and t_end."""
        path_to_orig_skycam_imgs = get_skycam_subdirs(self.skycam_path)['original']
        for fname in sorted(os.listdir(path_to_orig_skycam_imgs)):
            if fname.endswith('.mp4') or fname.endswith('.mpg'):
                os.remove("{0}/{1}".format(path_to_orig_skycam_imgs, fname))
            skycam_t = self.get_skycam_img_time(fname)
            if not (first_t <= skycam_t <= last_t):
                os.remove("{0}/{1}".format(path_to_orig_skycam_imgs, fname))

    def get_skycam_imgs(self, do_manual_skycam_download):
        """Downloads and unpacks original skycam data from https://mthamilton.ucolick.org/data/.
        Raises FileNotFoundError if valid skycam img zipfile could not be found."""
        if self.skycam_data_downloaded():
            return f"Data already downloaded at {self.skycam_subdirs['original']}"

        if do_manual_skycam_download:
            self.manual_skycam_download()
        else:
            self.automatic_skycam_download()

        if self.verbose: print("Unzipping skycam files...")
        unzip_images(self.skycam_path)
        # if self.verbose: print("Filtering skycam images...")
        # filter_images(self.skycam_path, first_t, last_t)

    """Skycam feature generation"""

    def add_skycam_data_to_skycam_df(self, skycam_df):
        """Add entries for each skycam image to skycam_df """
        original_img_dir = self.skycam_subdirs['original']
        for original_skycam_fname in os.listdir(original_img_dir):
            if original_skycam_fname.endswith('.jpg'):
                # Collect image features
                t = self.get_skycam_img_time(original_skycam_fname)
                unix_t = get_unix_from_datetime(t)
                skycam_uid = get_skycam_uid(original_skycam_fname)
                # Add entries to skycam_df
                skycam_df = add_skycam_img(
                    skycam_df,
                    skycam_uid,
                    self.batch_id,
                    self.skycam_dir,
                    original_skycam_fname,
                    unix_t,
                    **self.skycam_meta,
                    verbose=self.verbose)
        return skycam_df

    def create_skycam_image_features(self):
        """Run all preprocessing routines on the skycam data"""
        print('Running pre-processing routines.')
        if self.skycam_features_created():
            return f"Data in {self.skycam_path} already processed"

        corners_4x1x2 = get_corners(self.skycam_meta['skycam_type'])

        for original_fname in os.listdir(self.skycam_subdirs['original']):
            if original_fname[-4:] != '.jpg':
                continue

            original_img = cv2.imread(get_skycam_img_path(original_fname, 'original', self.skycam_path))
            cropped_fpath = get_skycam_img_path(original_fname, 'cropped', self.skycam_path)
            pfov_fpath = get_skycam_img_path(original_fname, 'pfov', self.skycam_path)

            crop_img(original_img, corners_4x1x2, cropped_fpath)
            plot_pfov(original_img, corners_4x1x2, pfov_fpath)

    def build_skycam_batch_data(self, skycam_df, do_manual_skycam_download=False):
        """Dispatch for building skycam features"""
        if self.verbose: print(f'Creating skycam features for {self.skycam_meta}')
        self.get_skycam_imgs(do_manual_skycam_download)
        self.create_skycam_image_features()
        skycam_df = self.add_skycam_data_to_skycam_df(skycam_df)
        return skycam_df
