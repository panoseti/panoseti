"""
PanosetiBatchBuilder creates all panoseti features for a single observing run.
"""
import sys
sys.path.append("../../util")

from panoseti_file_interfaces import ObservingRunInterface
from pano_utils import *
from dataframe_utils import *


class PanoBatchBuilder(ObservingRunInterface, PanoBatchDataFileTree):

    raw_data_shapes = {
        'raw-original': (32, 32),
        'raw-fft': (32, 32),
        'raw-derivative.-60': (32, 32),
        'raw-derivative-fft.-60': (32, 32),
    }

    # Data types every record must exclusively contain for a successful write.
    required_data_types = {'raw-original', 'raw-fft', 'raw-derivative.-60', 'raw-derivative-fft.-60'}

    def __init__(self, task, batch_id, batch_type, panoseti_data_dir, panoseti_run_dir, do_baseline_subtraction=False, force_recreate=False, verbose=False):
        ObservingRunInterface.__init__(self, panoseti_data_dir, panoseti_run_dir, do_baseline_subtraction, verbose=True)
        PanoBatchDataFileTree.__init__(self, batch_id, batch_type, panoseti_run_dir)
        self.force_recreate = force_recreate
        self.verbose = verbose
        # self.pano_dataset_builder = PanoDatasetBuilder(task, batch_id, self.run_dir)

        # Buffer for collecting raw images for features associated with the current pano_uid
        self.raw_data_arrays = dict()

        self.init_pano_subdirs()

    def init_pano_subdirs(self):
        """Initialize pre-processing directories."""
        for dir_name in self.pano_subdirs.values():
            os.makedirs(dir_name, exist_ok=True)
        # if V
        # self.is_uninitialized()

    """State checks"""

    def is_uninitialized(self):
        """Are pano subdirs initialized?"""
        if os.path.exists(self.pano_path) and len(os.listdir()) > 0:
            is_initialized = False
            for path in os.listdir():
                if path in self.pano_subdirs:
                    is_initialized |= len(os.listdir()) > 0
                if os.path.isfile(path):
                    is_initialized = False
            if is_initialized:
                raise FileExistsError(
                    f"Expected directory {self.pano_path} to be uninitialized, but found the following files:\n\t"
                    f"{os.walk(self.pano_path)}")

    def pano_features_created(self):
        """Returns True iff skycam features are already created."""
        ret = False
        if os.path.exists(f'{self.batch_path}/{self.pano_path_index_fname}'):
            ret = True
        for pano_subdir in self.pano_subdirs.values():
            ret &= os.path.exists(pano_subdir) and len(os.listdir(pano_subdir)) > 0
        return ret

    """Raw data array interface"""

    def clear_current_entry(self):
        self.raw_data_arrays = dict()

    def add_img_to_entry(self, data, img_type):
        """Add image data for img_type to the self.data_arrays buffer."""
        assert img_type in self.required_data_types, f'img_type "{img_type}" not supported!'
        assert img_type not in self.raw_data_arrays, f'img_type "{img_type}" already added!'

        self.raw_data_arrays[img_type] = data

    def all_data_valid(self):
        """Returns True iff the data types in self.required_data_types have been correctly created."""
        if set(self.raw_data_arrays.keys()) != self.required_data_types:
            return False
        for data in self.raw_data_arrays.values():
            if data is None:
                return False
        return True


    def write_arrays(self, pano_uid, overwrite_ok=True):
        """Write raw data features to file."""
        for img_type in self.raw_data_arrays:
            fpath = self.get_pano_img_path(pano_uid, img_type)
            if os.path.exists(fpath) and not overwrite_ok:
                raise FileExistsError(f'overwrite_ok=False and {fpath} exists.')
            data = self.raw_data_arrays[img_type]
            if data is None:
                raise ValueError(f'Data for "{img_type}" is None!')
            data = np.array(data)
            shape = self.raw_data_shapes[img_type]
            if data.shape != shape:
                data = np.reshape(data, shape)
            np.save(fpath, data)
        self.clear_current_entry()


    """Feature creation"""

    def img_transform(self, img):
        img = np.reshape(img, (32, 32))
        # img = np.flip(img, axis=0)
        img = np.rot90(img, 2)
        return img

    def make_original_img_features(self, curr_frame_seek_info, module_id, subtract_baseline=True):
        """Create features based on a single stacked panoseti image:
            1. Original image, with the baseline subtracted.
            2. FFT of (1)
        """

        stacked_img, stacked_meta = self.stack_frames(
            curr_frame_seek_info['file_idx'],
            curr_frame_seek_info['frame_offset'],
            module_id,
            subtract_baseline=subtract_baseline
        )
        self.add_img_to_entry(stacked_img, 'raw-original')
        self.add_img_to_entry(apply_fft(stacked_img), 'raw-fft')
        return 'ok', 'ok'

    def make_time_derivative_features(self, curr_frame_seek_info, module_id, delta_ts, subtract_baseline=False):
        """
        Create time derivative features relative to the frame specified by start_file_idx and start_frame_offset.
        Returns None if time derivative calc is not possible.

        Parameters
        @param start_file_idx: file containing reference frame
        @param start_frame_offset: number of frames to the reference frame
        @param module_id: module id number, as computed from its ip address
        @param delta_ts
        """
        assert max(delta_ts) < 0, 'Must specify delta_ts that are strictly in the past.'
        sorted_delta_ts = sorted(delta_ts)
        frame_unix_t = curr_frame_seek_info['frame_unix_t']
        curr_stacked_img, curr_stacked_meta = self.stack_frames(
            curr_frame_seek_info['file_idx'],
            curr_frame_seek_info['frame_offset'],
            module_id,
            subtract_baseline=subtract_baseline
        )
        prev_stacked_imgs = {}
        raw_diff_data = []
        for delta_t in sorted_delta_ts:
            # Get stacked images for each time-derivative specified in delta_ts.
            frame_seek_info = self.module_file_time_seek(module_id, frame_unix_t + delta_t)
            if frame_seek_info is None:
                return None, None
            prev_stacked, prev_stacked_meta = self.stack_frames(
                frame_seek_info['file_idx'],
                frame_seek_info['frame_offset'],
                module_id,
                subtract_baseline=subtract_baseline
            )
            # Compute difference between the current image and each of the delta_t images.
            prev_stacked_imgs[delta_t] = prev_stacked
            diff = curr_stacked_img - prev_stacked
            raw_diff_data.append(diff)

        for i in range(len(sorted_delta_ts)):
            delta_t = sorted_delta_ts[i]
            # Make derivative features
            derivative_type = f'raw-derivative.{delta_t}'
            data = raw_diff_data[i]
            self.add_img_to_entry(np.array(data), derivative_type)
            # Make derivative-fft features
            derivative_fft_type = f'raw-derivative-fft.{delta_t}'
            data = apply_fft(raw_diff_data[i])
            self.add_img_to_entry(np.array(data), derivative_fft_type)
        return 'ok', 'ok'

    def correlate_skycam_to_pano_img(self, skycam_unix_t, module_id):
        # Correlate skycam img to panoseti image, if possible
        skycam_integration_offset = -60
        return self.module_file_time_seek(module_id, skycam_unix_t + skycam_integration_offset)

    def create_training_features(self, feature_df,
                                 pano_df,
                                 module_id,
                                 skycam_df,
                                 skycam_dir,
                                 sample_stride,
                                 upsample_pano_frames,
                                 allow_skip=True,
                                 ):
        """For each original skycam image:
            1. Get its unix timestamp.
            2. Find the corresponding panoseti image frame, if it exists.
            3. Generate a corresponding set of panoseti image features relative to that frame.
        Note: must download skycam data before calling this routine.
        """
        if self.pano_features_created() and not self.force_recreate:
            raise FileExistsError(f"Data in {self.run_dir} already processed")

        module_image_pff_files = self.obs_pff_files[module_id]["img"]

        print(f'Generating features for module {module_id}')
        skycam_info = skycam_df.loc[skycam_df.skycam_dir == skycam_dir, ['skycam_uid', 'unix_t']]
        for index, skycam_row in skycam_info.sort_values(by='unix_t').iloc[::sample_stride, :].iterrows():
            skycam_uid, skycam_unix_t = skycam_row
            if self.verbose: print(f'\nGenerating pano features for skycam_uid {skycam_uid}...')
            # Upsample panoset files to gain more features
            if upsample_pano_frames:
                target_times = np.linspace(skycam_unix_t - 20, skycam_unix_t + 20, 5)
            else:
                target_times = [skycam_unix_t]
            for pano_target_unix_t in target_times:
                pano_frame_seek_info = self.correlate_skycam_to_pano_img(pano_target_unix_t, module_id)
                if pano_frame_seek_info is None:
                    # if self.verbose: print('Failed to find matching panoseti frames. Skipping...')
                    continue
                # Generate features
                self.make_original_img_features(
                    pano_frame_seek_info,
                    module_id
                )
                self.make_time_derivative_features(
                    pano_frame_seek_info,
                    module_id,
                    delta_ts=[-60]
                )

                # Skip this image if not all figs are valid
                if not self.all_data_valid():
                    self.clear_current_entry()
                    continue

                # Write figures to data dirs
                pano_fname = module_image_pff_files[pano_frame_seek_info['file_idx']]['fname']
                frame_offset = pano_frame_seek_info['frame_offset']
                pano_uid = get_pano_uid(pano_fname, frame_offset)

                # Commit entry to data_array for this run_dir
                # Write feature data
                self.write_arrays(pano_uid)
                self.clear_current_entry()
                # Update dataframes
                pano_df = add_pano_img(
                    pano_df,
                    pano_uid,
                    self.run_dir,
                    pano_fname,
                    frame_offset,
                    module_id,
                    pano_frame_seek_info['frame_unix_t'],
                    self.batch_id
                )
                feature_df = add_feature_entry(
                    feature_df,
                    skycam_uid,
                    pano_uid,
                    self.batch_id
                )
        return feature_df, pano_df

    def create_inference_features(self, feature_df, pano_df, module_id, time_step, allow_skip=True):
        """For each original skycam image:
            1. Get its unix timestamp.
            2. Find the corresponding panoseti image frame, if it exists.
            3. Generate a corresponding set of panoseti image features relative to that frame.
        Note: must download skycam data before calling this routine.
        @param time_step: Generate inference features for each time_step interval during the observing run.
        """
        if self.pano_features_created() and not self.force_recreate:
            raise FileExistsError(f"Data in {self.run_dir} already processed")

        module_image_pff_files = self.obs_pff_files[module_id]["img"]
        first_unix_t = module_image_pff_files[0]['first_unix_t']
        last_unix_t = module_image_pff_files[-1]['last_unix_t']
        inference_unix_ts = np.arange(first_unix_t, last_unix_t, time_step)

        print(f'Generating features for module {module_id}')
        for target_unix_t in inference_unix_ts:
            pano_frame_seek_info = self.module_file_time_seek(module_id, target_unix_t)
            if pano_frame_seek_info is None:
                if self.verbose: print('Failed to find matching panoseti frames. Skipping...')
                continue

            # Generate features
            self.make_original_img_features(
                pano_frame_seek_info,
                module_id
            )

            self.make_time_derivative_features(
                pano_frame_seek_info,
                module_id,
                delta_ts=[-60]
            )

            # Skip this image if not all figs are valid
            if not self.all_data_valid():
                self.clear_current_entry()
                continue

            # Write figures to data dirs
            pano_fname = module_image_pff_files[pano_frame_seek_info['file_idx']]['fname']
            frame_offset = pano_frame_seek_info['frame_offset']
            pano_uid = get_pano_uid(pano_fname, frame_offset)

            # Commit entry to data_array for this run_dir
            self.write_arrays(pano_uid)
            self.clear_current_entry()

            # Update dataframes
            pano_df = add_pano_img(
                pano_df,
                pano_uid,
                self.run_dir,
                pano_fname,
                frame_offset,
                module_id,
                pano_frame_seek_info['frame_unix_t'],
                self.batch_id
            )
            feature_df = add_feature_entry(
                feature_df,
                'INFERENCE-NO SKYCAM',
                pano_uid,
                self.batch_id
            )
        return feature_df, pano_df

    def build_pano_training_batch_data(self, feature_df, pano_df, skycam_df, skycam_dir, sample_stride=1, upsample_pano_frames=False):
        print(f'\nCreating panoseti features for {self.run_dir}')
        assert self.batch_type == 'training'
        for module_id in self.obs_pff_files:
            if len(self.obs_pff_files[module_id]["img"]) == 0:# or module_id == 3 or module_id == 254:
                continue
            try:
                feature_df, pano_df = self.create_training_features(
                    feature_df, pano_df, module_id, skycam_df, skycam_dir,
                    sample_stride, upsample_pano_frames
                )
            except FileExistsError:
                continue
        return feature_df, pano_df

    def build_pano_inference_batch_data(self, feature_df, pano_df, time_step):
        print(f'\nCreating panoseti features for {self.run_dir}')
        assert self.batch_type == 'inference'
        for module_id in self.obs_pff_files:
            if len(self.obs_pff_files[module_id]["img"]) == 0:
                continue
            try:
                feature_df, pano_df = self.create_inference_features(
                    feature_df, pano_df, module_id, time_step
                )
            except FileExistsError:
                continue
        return feature_df, pano_df

