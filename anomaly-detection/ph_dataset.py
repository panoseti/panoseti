import sys, os
import logging, typing

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from tqdm import tqdm

import torch

sys.path.append('../util')
import panoseti_file_interfaces as pfi
import pff
from vae_model import *
from functools import cache



class PulseHeightDataset(torch.utils.data.Dataset):
    """Interface for retrieving pulse-height images from a specific observing run."""
    MAX_PH_PIXEL_VAL = 2**16 - 1  # Max PH pixel value. PH pixels are typically represented as uint16 values.
    BASELINE_PERCENTILE = 0.05 # percentile of extreme values (~MAX_PH_PIXEL_VAL) used to compute the ph_baseline
  
    img_cwh = (1, 16, 16) # ph256 image dimensions: 1 channel, 16x16 image.

    @classmethod
    def baseline_shift(cls, ph_img: np.ndarray, module_meta: typing.Dict) -> np.ndarray:
      """
      Returns pulse height image after fixing underflowed pixels.
      Arguments:
          ph_img: raw pulse-height image
          module_meta: metadata for module that recorded this image
      """
      # ph_baseline = module_meta['ph_baseline']
      # ph_outlier_cutoff = module_meta['ph_outlier_cutoff']
      
      pixels_to_correct = ph_img[ph_img > max(cls.MAX_PH_PIXEL_VAL - 1000, 0)]
      # Value defining pixel outlier status:
      if len(pixels_to_correct) > 0:
        ph_outlier_cutoff = int(np.round(np.quantile(pixels_to_correct, q=cls.BASELINE_PERCENTILE)))
        # ph_outlier_cutoff = int(np.round(np.min(pixels_to_correct)))
      else:
        ph_outlier_cutoff = cls.MAX_PH_PIXEL_VAL
      
      ph_baseline = cls.MAX_PH_PIXEL_VAL - ph_outlier_cutoff
      ph_outlier_cutoff = min(ph_outlier_cutoff, cls.MAX_PH_PIXEL_VAL - 1000)

      ph_img_clean = ph_img + ph_baseline # Undo baseline subtraction
      ph_img_clean[ph_img_clean >= ph_outlier_cutoff] = 0 # Zero any remaining outliers
      assert ph_img_clean.dtype == np.uint16
      return ph_img_clean

    @classmethod
    def obs_run_ph_frame_generator(cls, ori: pfi.ObservingRunInterface, module_id: int):
      """Sequentially yields PH frames from the module with given module_id from the obs_run ori."""
      if module_id not in ori.obs_pff_files:
          raise ValueError(
              f'No module with ID "{module_id}"\n'
              f'Available module_ids:\n\t{list(ori.obs_pff_files.keys())}')
      elif len(ori.obs_pff_files[module_id]["ph"]) == 0:
          raise FileNotFoundError(f'Missing PH files for module "{module_id}" in {ori.run_dir}!')
      # self.logger.info(f'Detected the following PH files for module_id: {module_id}: {ori.obs_pff_files[module_id]["ph"]}')
      for ph_file in ori.obs_pff_files[module_id]["ph"]:
          fpath = "{0}/{1}/{2}".format(ori.data_dir, ori.run_dir, ph_file["fname"])
          with open(fpath, 'rb') as fp:
              frame_iterator = ori.pulse_height_frame_iterator(fp, 1)
              for j, img in frame_iterator:
                  j['wr_timestamp (s)'] = pff.wr_to_unix_decimal(j['pkt_tai'], j['pkt_nsec'], j['tv_sec'])
                  j['unix_timestamp'] = pd.to_datetime(float(j['wr_timestamp (s)']), unit = 's', utc=True)
                  yield j, img.astype(np.uint16)

    @classmethod
    def init_obs_run(cls, obs_run_config: typing.Dict, compute_ph_stats=True, max_stats_sample_size=10_000):
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
      module_meta = {}
      for module_id in dataset_module_ids:
        module_meta[module_id] = {}
        module_meta[module_id]['ph_frame_generator'] = None

      # Compute PH Baseline
      if compute_ph_stats:
        print(f'Computing PH baselines for {ori.run_dir}')
        
        for module_id in dataset_module_ids:
          nframes = 0
          for ph_file in ori.obs_pff_files[module_id]['ph']:
              nframes += ph_file['nframes']
          # Subsample all ph frames (could be millions)
          nsamples = min(max_stats_sample_size, nframes)
          # sample_idxs = np.random.choice(np.arange(nframes), size=nsamples, replace=False) CORRECT
          sample_idxs = np.arange(nsamples) # INCORRECT, but faster
          sampled_imgs = np.zeros((nsamples, 16, 16), dtype=np.uint16)
          
          frame_gen = cls.obs_run_ph_frame_generator(ori, module_id)
          img_idx = 0
          for i, (j, img) in tqdm(enumerate(frame_gen), unit="frames", total=nsamples - 1):
            if i in sample_idxs:
              sampled_imgs[img_idx] = img
              img_idx += 1
            if img_idx == nsamples:
              break
          pixels_to_correct = sampled_imgs[sampled_imgs > max(cls.MAX_PH_PIXEL_VAL - 1000, 0)]
          if len(pixels_to_correct) > 0:
            ph_outlier_cutoff = int(np.round(np.quantile(pixels_to_correct, q=cls.BASELINE_PERCENTILE))) # Value defining pixel outlier status:
          else:
            ph_outlier_cutoff = 0
          module_meta[module_id]['ph_baseline'] = cls.MAX_PH_PIXEL_VAL - ph_outlier_cutoff # Value defining amount to increase all pixel values by to account for baseline subtraction during data acquisition:
          module_meta[module_id]['ph_outlier_cutoff'] = min(ph_outlier_cutoff, cls.MAX_PH_PIXEL_VAL - 1000)
          shifted_sampled_imgs = cls.baseline_shift(sampled_imgs, module_meta)
          module_meta[module_id]['ph_median'] = np.median(shifted_sampled_imgs)
          module_meta[module_id]['ph_std'] = np.std(shifted_sampled_imgs)
          # im_plt = plt.imshow(module_meta[module_id]['ph_median_img'])
          # plt.colorbar(im_plt)
          # plt.show()

      obs_run = {
        'ori': ori,
        'frame_gen_counter': 0,
        'dataset_module_ids': dataset_module_ids,
        'module_meta': module_meta,
      }
      return obs_run

    @classmethod
    def init_logger(cls, log_level):
      """Initialize a logger instance"""
      logger = logging.getLogger("PulseHeightDataset_Logger")
      logger.setLevel(log_level)
      console_handler = logging.StreamHandler()
      console_handler.setLevel(logging.INFO)
      formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s')
      console_handler.setFormatter(formatter)
      logger.addHandler(console_handler)
      return logger

    def __init__(self, ph_dataset_config: typing.Dict, transform=None, target_transform=None, log_level=logging.INFO):
        assert 'observing_runs' in ph_dataset_config, "Could not find 'observing_runs' in ph_dataset_config."
        super().__init__()
        self.transform = transform
        self.obs_runs = []
        self.EFFECTIVE_MAX_VAL = self.MAX_PH_PIXEL_VAL / 16 # most ph images are less than 4k intensity.
        self.NORM_CONST = np.log(self.EFFECTIVE_MAX_VAL)
        
        self.logger = PulseHeightDataset.init_logger(log_level)
        
        # Parse dataset configs and initialize observing file interfaces.
        for obs_run_config in ph_dataset_config['observing_runs']:
          obs_run = PulseHeightDataset.init_obs_run(obs_run_config)
          self.obs_runs.append(obs_run)

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

        # Initialize PH frame generator
        self.ph_gen = self.dataset_ph_frame_generator()
        self.compute_ph_stats()
        self.reset_ph_generator()

    # @cache
    def __getitem__(self, index: int) -> torch.Tensor:
        """Get PH frame at index. Note: currently index has no effect (TODO). index is required by the PyTorch abstract Dataset class."""
        ph_img = self.get_ph_data(index)['img']
        ph_img_norm = self.norm(ph_img)
        return self.transform(ph_img_norm)

    def __len__(self) -> int:
        return self.dataset_length

    def compute_ph_stats(self, n_samples=10_000):
      print('Computing PH image statistics')
      sample_idxs = np.random.choice(np.arange(self.dataset_length), size=min(n_samples, self.dataset_length), replace=False)
      sampled_ph_imgs = torch.zeros((len(sample_idxs), 1, 16, 16))
      img_idx = 0
      for i in tqdm(range(self.dataset_length), unit="ph_frames"):
        ph_img = self.get_ph_data(i)['img']
        if i in sample_idxs:
          sampled_ph_imgs[img_idx] = self.transform(ph_img)
          img_idx += 1
      self.stats = {
        'mean': torch.mean(sampled_ph_imgs).numpy(),
        'median': torch.median(sampled_ph_imgs).numpy(),
        'std': torch.std(sampled_ph_imgs).numpy(),
        'min': torch.min(sampled_ph_imgs).numpy(),
        'max': torch.max(sampled_ph_imgs).numpy(),
        'q9995':torch.quantile(sampled_ph_imgs, 0.9995),
      }
      print(self.stats)
      self.reset_ph_generator()
      
    def norm(self, ph_img):
      """Log-normalize a given PH image with uint16 pixels into the range [-1, 1]."""
      # norm_ph_img = np.clip(ph_img - self.ph_mean, 0, self.MAX_PH_PIXEL_VAL)
      # norm_ph_img = 2 * (np.log(ph_img + 1) / np.log(self.MAX_PH_PIXEL_VAL + 1)) - 1 # [-1, 1]
      # norm_ph_img = np.log(ph_img + 1) / np.log(cls.MAX_PH_PIXEL_VAL + 1) # [0, 1]
      # norm_ph_img = 2 * (ph_img / cls.MAX_PH_PIXEL_VAL) - 1
      # norm_ph_img = 2 * (np.log(ph_img / self.MAX_PH_PIXEL_VAL + 2) / np.log(3)) - 1 # [-1, 1]
      # norm_ph_img = 2 * ph_img / self.MAX_PH_PIXEL_VAL
      # norm_ph_img = np.clip(norm_ph_img, 0, self.MAX_PH_PIXEL_VAL)
      norm_ph_img = ph_img - self.stats['min']
      norm_ph_img = np.clip(norm_ph_img, 0, self.EFFECTIVE_MAX_VAL)
      norm_ph_img = np.log(norm_ph_img + 1) / self.NORM_CONST # [0, ~1]
      norm_ph_img = 2 * (norm_ph_img - 0.25) # [-1, ~1]
      # print(np.min(norm_ph_img), np.max(norm_ph_img))
      # assert -1.0 <= np.min(norm_ph_img) and np.max(norm_ph_img) <= 1.0, "np.min(norm_ph_img)={0}, np.max(norm_ph_img)={1}".format(np.min(norm_ph_img), np.max(norm_ph_img))
      return norm_ph_img
  
    def inv_norm(self, norm_ph_img):
      """Invert the log-normalization performed by norm."""
      
      # ph_img = np.exp(norm_ph_img * np.log(cls.MAX_PH_PIXEL_VAL)) - 1
      # ph_img = (norm_ph_img + 1) * cls.MAX_PH_PIXEL_VAL / 2
      # ph_img = (norm_ph_img * self.MAX_PH_PIXEL_VAL) + self.ph_mean
      # ph_img = (np.exp((norm_ph_img + 1) * np.log(3) / 2) - 2) * self.MAX_PH_PIXEL_VAL # [-1, 1]
      # ph_img = self.MAX_PH_PIXEL_VAL * norm_ph_img / 2
      # ph_img = np.clip(ph_img, 0, self.MAX_PH_PIXEL_VAL)
      # ph_img[ph_img >= self.OUTLIER_CUTOFF] = 0
      # assert 0.0 <= np.min(ph_img) and np.max(ph_img) <= self.MAX_PH_PIXEL_VAL, "np.min(ph_img)={0}, np.max(ph_img)={1}".format(np.min(ph_img), np.max(ph_img))
      # norm_ph_img = (norm_ph_img + 1) / 2
      ph_img = norm_ph_img / 2 + 0.25
      ph_img = np.exp(ph_img * self.NORM_CONST) - 1
      ph_img = ph_img + self.stats['min']
      return ph_img

    def clean_ph_img(self, ph_img: np.ndarray, module_meta: typing.Dict) -> np.ndarray:
      """
      Returns pulse-height image after applying the specified outlier_strategy with pixels log normalized to [-1, 1] 

      Arguments:
          ph_img: raw pulse-height image
          module_meta: metadata for module that recorded this image
      Returns: pulse-height frame after applying the specified outlier_strategy with pixels log normalized to [-1, 1] 
      """
      assert ph_img.dtype == np.uint16
      # image should not be all zeros
      if np.max(ph_img) == 0:
          # self.logger.info(f'{np.sum(ph_img_clean == 0)} PH pixels are zero')
          return None
      ph_img_clean = self.baseline_shift(ph_img, module_meta)
      ph_img_clean = (ph_img_clean - module_meta['ph_median']) / module_meta['ph_std']
      return ph_img_clean

    def dataset_ph_frame_generator(self):
        """
        Returns a generator that lazily and sequentially yields PH frames in a round-robin fashion from all available ph files for this object.
        TODO: create indexing function to group ph frames.
        """
        self.logger.info('Resetting dataset-global PH frame generator') 
        for obs_run in self.obs_runs:
          obs_run['frame_gen_counter'] = 0
          for module_id in obs_run['dataset_module_ids']:
            obs_run['module_meta'][module_id]['ph_generator'] = self.obs_run_ph_frame_generator(obs_run['ori'], module_id)
        for i in range(self.dataset_length):
          obs_run = self.obs_runs[i % len(self.obs_runs)]
          module_id = obs_run['dataset_module_ids'][obs_run['frame_gen_counter'] % len(obs_run['dataset_module_ids'])]
          module_meta = obs_run['module_meta'][module_id]
          obs_run['frame_gen_counter'] += 1
          try:
            j, ph_img = next(module_meta['ph_generator'])
          except StopIteration:
            # Reset ph_generator to beginning of file
            self.logger.info(f"resetting ph_generator for moddule {module_id} in in {obs_run['ori'].run_dir}")
            module_meta['ph_generator'] = self.obs_run_ph_frame_generator(obs_run['ori'], module_id)
            try:
              j, ph_img = next(module_meta['ph_generator']) # should succeed
            except StopIteration: # only has StopIteration error if ph file is all zero data
              self.logger.error(f"Unexpected StopIteration error when accessing ph frames from module {module_id} in {obs_run['ori'].run_dir}")
              continue
          ph_img_clean = self.clean_ph_img(ph_img, module_meta)
          if ph_img_clean is None: # skip bad ph frames (all zeros).
              self.logger.warn(f"run: {obs_run['ori'].run_dir:<58} module {module_id:<4} has a frame with all zeros.")
              continue
          yield {'meta': j, 'img': ph_img_clean}
    
    @cache
    def get_ph_data(self, index: int):
        try:
            return next(self.ph_gen)
        except StopIteration:
            self.reset_ph_generator()
            return next(self.ph_gen)

    def reset_ph_generator(self):
      self.ph_gen = self.dataset_ph_frame_generator()


    # PH data EDA: visualize PH image and the distribution of pixel values.
    def plot_ph_pixel_dist(self, ph_img, meta, ax=None, upper=None, limit=10_000):
        """Plot pulse height pixel distribution (for EDA outlier rejection)."""
        # plt.xscale('log')
        pixels = self.norm(ph_img).ravel()
        title = ""
        if upper == True:
          pixels = pixels[pixels > limit]
          log_scale = False
          title = f"Distribution of pixels > {limit}"
        elif upper == False:
          pixels = pixels[pixels <= limit]
          log_scale = False
          title = f"Distribution of pixels <= {limit}"
        elif upper is None:
          log_scale = False
          title = f"Distribution of all PH pixels"
        sns.histplot(pixels, stat='density', ax=ax, log_scale=log_scale)
        if ax is not None:
            ax.set_title(title)
    
    def plot_ph_img(self, ph_img, meta, ax=None, log_cbar=False):
        if ax is None:
            f = plt.figure()
            ax = plt.gca()
        if log_cbar:
          img_plt = ax.imshow(self.norm(ph_img), cmap='icefire')
          cbar = plt.colorbar(img_plt, fraction=0.045)
          # cbar.ax.locator_params(nbins=10)
          cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, t: self.colorbar_formatter(v, t)))
        else:
          img_plt = ax.imshow(self.norm(ph_img), cmap='icefire')
          cbar = plt.colorbar(img_plt, fraction=0.045)
          # cbar.ax.locator_params(nbins=10)
          # cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, t: self.colorbar_formatter(v, t)))

    def colorbar_formatter(self, value, tick_position):
        val_in_original_units = self.inv_norm(np.array([value]))[0]
        return int(val_in_original_units)
