from datetime import timedelta
import os
import numpy as np
import matplotlib.pyplot as plt

from panoseti_file_interfaces import ObservingRunFileInterface, ModuleImageInterface
from panoseti_batch_builder import PanosetiBatchBuilder
from panoseti_batch_utils import *
from skycam_utils import get_skycam_img_time, get_skycam_subdirs, get_unix_from_datetime
import make_batch


skycam_imgs_root_path = '/Users/nico/panoseti/panoseti-software/cloud-detection/data_labeling/batch_data/task_cloud-detection.batch-id_0/skycam_imgs'
skycam_dir = 'SC2_imgs_2023-07-31'


if __name__ == '__main__':
    DATA_DIR = '/Users/nico/Downloads/panoseti_test_data/obs_data/data'
    # RUN_DIR = 'obs_Lick.start_2023-08-29T04:49:58Z.runtype_sci-obs.pffd'
    RUN_DIR = 'obs_Lick.start_2023-08-01T05:14:21Z.runtype_sci-obs.pffd'

    builder = PanosetiBatchBuilder(DATA_DIR, RUN_DIR, 'cloud-detection', 0)
    test_mii = ModuleImageInterface(DATA_DIR, RUN_DIR, 254)

    print(builder.start_utc)
    print(builder.stop_utc)

    make_batch.build_batch('cloud-detection', 0, builder.start_utc, builder.stop_utc)

    skycam_subdirs = get_skycam_subdirs(f'{skycam_imgs_root_path}/{skycam_dir}')

    for fname in sorted(os.listdir(skycam_subdirs['original'])):
        if fname.endswith('.jpg'):
            t = get_skycam_img_time(fname)
            skycam_unix_t = get_unix_from_datetime(t)

            ret = builder.module_file_time_seek(254, get_unix_from_datetime(t))
            if ret is not None:
                fig = builder.time_derivative(
                    ret['file_idx'],
                    ret['frame_offset'],
                    254,
                    1,
                    10,
                    3,
                    True
                )
                plt.pause(1)
                plt.close(fig)
                file_info = builder.obs_pff_files[254][ret['file_idx']]
                #print(builder.start_utc <= t <= builder.stop_utc)
                print('delta_t = ', (file_info['last_unix_t'] - file_info['first_unix_t']))
                print(file_info['seqno'])
                print()

    """
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
                    fig = plot_image_grid(imgs, delta_ts)
                    fig.suptitle(f'{delta_ts}, L->R; T->B')
                    plt.pause(0.05)
                hist.insert(0, img)
            plt.close(fig)
    """