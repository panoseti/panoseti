
import os
import numpy as np
import seaborn_image as isns
import matplotlib.pyplot as plt

from panoseti_file_interfaces import ObservingRunFileInterface, ModuleImageInterface


class PanosetiBatchBuilder(ObservingRunFileInterface):
    root_batch_data_dir = 'batch_data'
    def __init__(self, data_dir, run_dir, task, batch_id):
        super().__init__(data_dir, run_dir)
        self.task = task
        self.batch_id = batch_id
        self.batch_dir = ...

    @staticmethod
    def plot_image_grid(imgs, delta_ts):
        for i in range(len(imgs)):
            if imgs[i] is None or not isinstance(imgs[i], np.ndarray):
                print('no image')
                return None
            if imgs[i].shape != (32,32):
                imgs[i] = np.reshape(imgs[i], (32, 32))
        # print(delta_ts)
        # ax = isns.ImageGrid(imgs, height=1.5, col_wrap=1, vmin=-100, vmax=100, cmap="viridis", cbar=False)
        ax = isns.ImageGrid(imgs, height=1.5, col_wrap=1, vmin=-3, vmax=3, cmap="viridis", cbar=False)
        return ax.fig

    @staticmethod
    def plot_image_fft(img):
        if img is None or not isinstance(img, np.ndarray):
            print('no image')
            return None
        if img.shape != (32, 32):
            img = np.reshape(img, (32, 32))
        ax = isns.fftplot(img, cmap="viridis", window_type='cosine')
        return ax.get_figure()

    def is_initialized(self, pano_dir):
        img_subdirs = get_img_subdirs(pano_dir)
        if os.path.exists(pano_dir) and len(os.listdir()) > 0:
            is_initialized = False
            for path in os.listdir():
                if path in img_subdirs:
                    is_initialized |= len(os.listdir()) > 0
                if os.path.isfile(path):
                    is_initialized = False
            if is_initialized:
                raise FileExistsError(
                    f"Expected directory {pano_dir} to be uninitialized, but found the following files:\n\t"
                    f"{os.walk(pano_dir)}")

    def get_panoseti_batch_dir(self):
        ...

    def gen_npy_fname(self):
        npy_fname = f"{self.batch_dir}/data.npy"
        return npy_fname

    def get_empty_data_array(self, file_attrs, step_size):
        data_size = 0
        for i in range(len(file_attrs)):
            data_size += file_attrs[i]['nframes'] // step_size
        return np.zeros(data_size)

    def get_files_to_process(self, data_dir, run_dir, module):
        files_to_process = []
        for fname in os.listdir(f'{data_dir}/{run_dir}'):
            if pff.is_pff_file(fname) and pff.pff_file_type(fname) in ('img16', 'img8'):
                files_to_process.append(fname)
        return files_to_process

    def process_file(self, file_info, data, itr_info, step_size):
        """On a sample of the frames in the file represented by file_info, add the total
        image brightness to the data array beginning at data_offset."""
        with open(f"{self.data_dir}/{self.run_dir}/{file_info['fname']}", 'rb') as f:
            # Start file pointer with an offset based on the previous file -> ensures even frame sampling
            f.seek(
                itr_info['fstart_offset'] * file_info['frame_size'],
                os.SEEK_CUR
            )
            new_nframes = file_info['nframes'] - itr_info['fstart_offset']
            for i in range(new_nframes // step_size):
                j, img = self.get_next_frame(f, file_info['frame_size'], file_info['bytes_per_pixel'], step_size)
                data[itr_info['data_offset'] + i] = np.sum(img)
            itr_info['fstart_offset'] = file_info['nframes'] - (new_nframes // step_size) * step_size

    def get_data(self, file_info_array, analysis_dir, step_size):
        # Save reduced data to file
        npy_fname = self.gen_npy_fname(analysis_dir)
        if os.path.exists(npy_fname):
            data_arr = np.load(npy_fname)
            return data_arr
        itr_info = {
            "data_offset": 0,
            "fstart_offset": 0  # Ensures frame step size across files
        }
        data_arr = self.get_empty_data_array(file_info_array, step_size)
        for i in range(len(file_info_array)):
            print(f"Processing {file_info_array[i]['fname']}")
            file_info = file_info_array[i]
            self.process_file(file_info, data_arr, itr_info, step_size)
            itr_info['data_offset'] += file_info["nframes"] // step_size

        np.save(npy_fname, data_arr)
        return data_arr


if __name__ == '__main__':
    DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'
    # RUN_DIR = 'obs_Lick.start_2023-08-29T04:49:58Z.runtype_sci-obs.pffd'
    RUN_DIR = 'obs_Lick.start_2023-08-01T05:14:21Z.runtype_sci-obs.pffd'

    test_batch_builder = PanosetiBatchBuilder(DATA_DIR, RUN_DIR, 'cloud-detection', 0)
    test_mii = ModuleImageInterface(DATA_DIR, RUN_DIR, 254)
    print(test_mii.module_pff_files[0]['nframes'])
    for i in range(0, len(test_mii.module_pff_files)):
        fpath = test_mii.run_path + '/' + test_mii.module_pff_files[i]['fname']
        delta_t = 1
        step_size = int(delta_t / (test_mii.intgrn_usec * 1e-6))
        print(f"Plotting {test_mii.module_pff_files[i]['fname']}. "
              f'Delta t = {step_size * test_mii.intgrn_usec * 1e-6}s')
        from scipy.fft import fftn, fftshift
        from skimage.filters import window

        with (open(fpath, 'rb') as fp):
            fig = None
            from collections import deque
            maxlen = 120
            hist = []
            for j, img in test_mii.frame_iterator(fp, step_size):
                if fig:
                    plt.close(fig)
                if len(hist) == maxlen:
                    # mean = np.mean(hist)
                    # std = np.std(hist)
                    # diff = (img - prev_img) / std

                    imgs = [(img - np.median(hist)) / np.std(hist)]
                    delta_ts = [0]
                    for i in [10, 30, 60, 90, 120]:
                        #print(i)
                        data = (img - hist[i - 1]) / np.std(hist)
                        imgs.append(data)
                        delta_ts.append(-delta_t * i)
                    hist.pop()
                    prev_img = hist.pop()

                    # fig = test_mii.plot_image((img - np.mean(hist)) / np.std(hist))
                    fig = test_mii.plot_image_grid(imgs, delta_ts)
                    fig.suptitle(f'{delta_ts}, L->R; T->B')
                    plt.pause(0.05)
                hist.insert(0, img)
            plt.close(fig)

