import sys, os
import logging, typing

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import torch

sys.path.append('../util')
import panoseti_file_interfaces as pfi
import pff
from vae_model import *



class PulseHeightDataset(torch.utils.data.Dataset):
    """Interface for retrieving pulse-height images from a specific observing run."""
    MAX_PH_PIXEL_VAL = 2**16 - 1  # Max PH pixel value. PH pixels are typically represented as uint16 values.
    OUTLIER_CUTOFF = MAX_PH_PIXEL_VAL - 5000  # Value defining pixel outlier status: TODO: do some stats to find better cutoff.
    img_cwh = (1, 16, 16) # ph256 image dimensions: 1 channel, 16x16 image.

    @classmethod
    def norm(cls, ph_img):
      """Log-normalize a given PH image with uint16 pixels into the range [-1, 1]."""
      norm_ph_img = 2 * (np.log(ph_img + 1) / np.log(cls.MAX_PH_PIXEL_VAL + 1)) - 1 # [-1, 1]
      # norm_ph_img = np.log(ph_img + 1) / np.log(cls.MAX_PH_PIXEL_VAL + 1) # [0, 1]
      # norm_ph_img = 2 * (ph_img / cls.MAX_PH_PIXEL_VAL) - 1
      # norm_ph_img = (ph_img / cls.MAX_PH_PIXEL_VAL)
      assert -1.0 <= np.min(norm_ph_img) and np.max(norm_ph_img) <= 1.0, "np.min(norm_ph_img)={0}, np.max(norm_ph_img)={1}".format(np.min(norm_ph_img), np.max(norm_ph_img))
      return norm_ph_img
  
    @classmethod
    def inv_norm(cls, norm_ph_img):
      """Invert the log-normalization performed by norm."""
      ph_img = np.exp((norm_ph_img + 1) * np.log(cls.MAX_PH_PIXEL_VAL) / 2) - 1 # [-1, 1]
      # ph_img = np.exp(norm_ph_img * np.log(cls.MAX_PH_PIXEL_VAL)) - 1
      # ph_img = (norm_ph_img + 1) * cls.MAX_PH_PIXEL_VAL / 2
      # ph_img = norm_ph_img * cls.MAX_PH_PIXEL_VAL
      ph_img = np.clip(ph_img, 0, cls.MAX_PH_PIXEL_VAL)
      assert 0.0 <= np.min(ph_img) and np.max(ph_img) <= cls.MAX_PH_PIXEL_VAL, "np.min(ph_img)={0}, np.max(ph_img)={1}".format(np.min(ph_img), np.max(ph_img))
      return ph_img

    @classmethod
    def obs_run_ph_frame_generator(cls, ori: pfi.ObservingRunInterface, module_id: int):
        """Sequentially yields PH frames from the module with given module_id from the obs_run ori."""
        if module_id not in ori.obs_pff_files:
            print(f'No module with ID "{module_id}"\n'
                  f'Available module_ids:\n\t{list(ori.obs_pff_files.keys())}')
            return None
        for ph_file in ori.obs_pff_files[module_id]["ph"]:
            fpath = "{0}/{1}/{2}".format(ori.data_dir, ori.run_dir, ph_file["fname"])
            with open(fpath, 'rb') as fp:
                frame_iterator = ori.pulse_height_frame_iterator(fp, 1)
                for j, img in frame_iterator:
                    j['wr_timestamp (s)'] = pff.wr_to_unix_decimal(j['pkt_tai'], j['pkt_nsec'], j['tv_sec'])
                    j['unix_timestamp'] = pd.to_datetime(float(j['wr_timestamp (s)']), unit = 's', utc=True)
                    yield j, img

    @classmethod
    def init_obs_run(cls, obs_run_config: typing.Dict):
      """Parses obs_run_config and returns a dict containing an ObservingRunInterface instance and module_ids to use from this run."""
      assert {'data_dir', 'run_dir', 'module_ids'}.issubset(set(obs_run_config))
      ori = pfi.ObservingRunInterface(obs_run_config['data_dir'], obs_run_config['run_dir'])
      
      # Get list of modules to use in the dataset
      dataset_module_ids = []
      if obs_run_config['module_ids'] == 'all':
        dataset_module_ids = list(ori.obs_pff_files.keys())
      elif set(obs_run_config['module_ids']).issubset(set(ori.obs_pff_files)) and len(obs_run_config['module_ids']) > 0:
        dataset_module_ids = config['module_ids']
      else:
        raise ValueError(
          f"'module_ids' must be either 'all' or a subset of valid module_ids: {set(ori.obs_pff_files)} "\
          f"in the specified observing run: {obs_run_config['run_dir']}."
        )
      # Initialize ph frame generators for this observing run
      ph_frame_generators = {}
      for module_id in dataset_module_ids:
        ph_frame_generators[module_id] = PulseHeightDataset.obs_run_ph_frame_generator(ori, module_id)
      
      obs_run = {
        'ori': ori,
        'dataset_module_ids': dataset_module_ids,
        'ph_generators': ph_frame_generators,
        'frame_gen_counter': 0,
      }
      return obs_run

    def __init__(self, ph_dataset_config: typing.Dict, transform=None, target_transform=None, log_level=logging.ERROR):
        assert 'observing_runs' in ph_dataset_config, "Could not find 'observing_runs' in ph_dataset_config."
        super().__init__()
        self.transform = transform
        self.obs_runs = []
        
        # Parse dataset configs and initialize observing file interfaces.
        for obs_run_config in ph_dataset_config['observing_runs']:
          obs_run = PulseHeightDataset.init_obs_run(obs_run_config)
          self.obs_runs.append(obs_run)
        
        # Initialize PH frame generator
        self.ph_gen = self.dataset_ph_frame_generator()
        
        # Initialize logger
        self.logger = logging.getLogger("PulseHeightDataset")
        self.logger.setLevel(log_level)
        
        # Compute available PH frames
        self.total_ph_frames = 0
        for obs_run in self.obs_runs:
          for module_id in obs_run['dataset_module_ids']:
              for ph_file in obs_run['ori'].obs_pff_files[module_id]['ph']:
                  self.total_ph_frames += ph_file['nframes']
        # Set dataset length
        if 'max_ph_frames' in ph_dataset_config:
          self.dataset_length = min(ph_dataset_config['max_ph_frames'], self.total_ph_frames)
        else:
          self.dataset_length = self.total_ph_frames
    
    def __getitem__(self, index: int) -> torch.Tensor:
        """Get PH frame at index. Note: currently index has no effect (TODO). index is required by the PyTorch abstract Dataset class."""
        ph_img = self.get_ph_data(index)['img']
        return self.transform(ph_img)

    def __len__(self) -> int:
        return self.dataset_length

    def dataset_ph_frame_generator(self):
        """
        Returns a generator that lazily and sequentially yields PH frames in a round-robin fashion from all available ph files for this object.
        TODO: create indexing function to group ph frames.
        """
        for obs_run in self.obs_runs:
          obs_run['frame_gen_counter'] = 0
        for i in range(self.dataset_length):
          print(i)
          obs_run = self.obs_runs[i % len(self.obs_runs)]
          module_id = obs_run['dataset_module_ids'][obs_run['frame_gen_counter'] % len(obs_run['dataset_module_ids'])]
          obs_run['frame_gen_counter'] += 1
          try:
            j, ph_img = next(obs_run['ph_generators'][module_id])
          except StopIteration:
            # Reset ph_generator to beginning of file
            obs_run['ph_generators'][module_id] = PulseHeightDataset.obs_run_ph_frame_generator(obs_run['ori'], module_id)
            j, ph_img = next(obs_run['ph_generators'][module_id]) # should succeed
          ph_img_clean = self.clean_ph_img(ph_img)
          if ph_img_clean is None: # skip bad ph frames (all zeros).
              continue
          yield {'meta': j, 'img': ph_img_clean}
    
    def get_ph_data(self, index: int):
        # print(self.obs_runs)
        try:
            return next(self.ph_gen)
        except StopIteration:
            self.ph_gen = self.dataset_ph_frame_generator()
            return next(self.ph_gen)

    def reset_ph_generator(self):
      self.ph_gen = self.dataset_ph_frame_generator()
            
    def clean_ph_img(self, ph_img: np.ndarray, outlier_strategy='zero', clip_z_score=5) -> typing.Tuple[int, int]:
        """Set outlier pulse heigh pixel values to 0.
        Arguments:
            ph_img: raw pulse-height image
            outlier_strategy: how to deal with outliers
                'zero' => set outliers to 0
                'clip' => clip outlier values to a max pixel value given by clip_z_score and inlier mean and standard deviations.
        Returns: pulse-height frame after applying the specified outlier_strategy with pixels log normalized to [-1, 1] 
        """
        # Remove outliers
        assert outlier_strategy in ['zero', 'clip']
        if outlier_strategy == 'zero':
            ph_img_clean = ph_img.copy()
            ph_img_clean[ph_img >= self.OUTLIER_CUTOFF] = 0
        elif outlier_strategy == 'clip':
            inlier_mean = np.mean(ph_img[ph_img < self.OUTLIER_CUTOFF])
            inlier_std = np.std(ph_img[ph_img < self.OUTLIER_CUTOFF])
            clip_max_pixel_val = inlier_mean + clip_z_score * inlier_std
            ph_img_clean = ph_img.copy()
            ph_img_clean = np.clip(ph_img_clean, 0, clip_max_pixel_val)
        
        # PH image should not be all zeros
        if np.max(ph_img_clean) == 0:
            self.logger.warning('All PH pixels are zero')
            return None
        return self.norm(ph_img_clean)


# PH data EDA: visualize PH image and the distribution of pixel values.
def plot_ph_pixel_dist(norm_ph_img, meta, ax=None):
    """Plot pulse height pixel distribution (for EDA outlier rejection)."""
    ph_img = PulseHeightDataset.inv_norm(norm_ph_img)
    sns.histplot(ph_img.ravel(), stat='density', ax=ax)
    if ax:
        ax.set_title(f"Distribution of PH pixels from Q{meta['quabo_num']} at \n{meta['unix_timestamp']}")

def plot_ph_img(norm_ph_img, meta, ax=None):
    if ax is None:
        f = plt.figure()
        ax = plt.gca()
    ph_img = PulseHeightDataset.inv_norm(norm_ph_img)
    img_plt = ax.imshow(ph_img, cmap='rocket')
    plt.colorbar(img_plt, fraction=0.045)


