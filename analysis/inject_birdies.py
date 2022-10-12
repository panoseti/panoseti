"""
The program uses models of the modules in an observatory and the
celestial sphere to generate birdies and simulate image mode data for a single image file.

TODO:
    - File IO:
        - Add utility methods to import image mode files, read their metadata and image arrays, and write RAW + birdie frames.
        - Most important metadata:
            - Module ID, module orientation (alt-az + observatory GPS), integration time, start time, and end time.
    - Setup procedure:
        - Create or update birdie log file.
        - Open a file object for the image file. Done
    - Main loop
        - Check if we’ve reached EOF in any of the image mode files. Done
        - Simulate module image mode output. Done
        - Update image frames (if applicable). Done

"""
import math
import time
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


import analysis_util
from BirdieSource import BaseBirdieSource
from ModuleView import ModuleView
import birdie_injection_utils as birdie_utils
import sky_band

sys.path.append('../util')
import pff
import config_file
sys.path.append('../control')

np.random.seed(300)

# File IO


def get_birdie_config(birdie_config_path):
    """Loads a birdie injection config file with the form:
    {
        "integration_time": 20e-1,
        "num_birdies": 30,
        "array_resolution": 50,
        "psf_sigma": 6,
        "param_ranges": {
            "duty_cycle": [0.1, 1],
            "period": [1, 10],
            "intensity": [100, 400]
        }
    }
    array_resolution: number of pixels per degree RA and degree DEC.
    psf_sigma: value of sigma used in the simulated (gaussian) point-spread function.
    param_ranges: possible values for BirdieSource objects.
    """
    config_file.check_config_file(birdie_config_path)
    with open(birdie_config_path, 'r+') as f:
        birdie_config = json.loads(f.read())
        return birdie_config


def update_birdie_log(birdie_log_path):
    """Updates the birdie_log.json file (creating it if necessary) with metadata about each birdie."""
    new_log_data = {
    }
    if os.path.exists(birdie_log_path):
        with open(birdie_log_path, 'r+') as f:
            s = f.read()
            c = json.loads(s)
            # Add new birdie log data
            # Example below
            new_log_data["backup_number"] = c["backups"][-1]["backup_number"] + 1
            c["backups"].append(new_log_data)

            json_obj = json.dumps(c, indent=4)
            f.seek(0)
            f.write(json_obj)
    else:
        c = {
            'Stuff'
        }
        with open(birdie_log_path, 'w+') as f:
            json_obj = json.dumps(c, indent=4)
            f.write(json_obj)


def get_obs_config(data_dir, run):
    pass


def get_next_frame(file_obj):
    """Returns the next image frame from file_obj."""
    j, img = None, None
    start_timestamp = None
    try:
        j = pff.read_json(file_obj)
        j = json.loads(j.encode())
        # For ph files: img size = 16 x 16 and bytes per pixel = 2.
        img = pff.read_image(file_obj, 32, 2)
    except Exception as e:
        # Deal with EOF issue in pff.read_json
        if repr(e)[:26] == "Exception('bad type code',":
            return None
    if not j or not img:
        return None
    return img, j


# Simulation set up


def init_module(start_utc):
    m1 = ModuleView('test', start_utc, 10.3, 44.2, 234, 77, 77, 77)
    return m1


def init_sky_array(array_resolution):
    return birdie_utils.get_sky_image_array(array_resolution, 1.25, verbose=True)


def get_birdie_config_vector(param_ranges):
    """Generates a tuple of BirdieSource initialization parameters with uniform distribution
    on the ranges of possible values, provided by param_ranges."""
    unif = np.random.uniform
    config_vector = []
    param_order = ['ra', 'dec', 'start_utc', 'end_utc', 'duty_cycle', 'period', 'intensity']
    for param in param_order:
        config_vector.append(unif(*(param_ranges[param])))
    return config_vector


def init_birdies(num, param_ranges):
    """Initialize BirdieSource objects with randomly selected parameter values
     and store them in a hashmap indexed by RA."""
    birdie_sources = {d: [] for d in range(360)}
    for x in range(num):
        config_vector = get_birdie_config_vector(param_ranges)
        b = BaseBirdieSource(*config_vector)
        birdie_sources[int(b.ra)].append(b)
    return birdie_sources


def init_birdie_param_ranges(start_utc, end_utc, bounding_box, param_ranges):
    """Param_ranges specifies the range of possible values for each BirdieSource parameter."""
    l_ra, r_ra = birdie_utils.ra_range
    l_dec, r_dec = birdie_utils.dec_range
    if r_ra < l_ra:
        r_ra += 360
    param_ranges['ra'] = (l_ra, r_ra)
    param_ranges['dec'] = (l_dec, r_dec)
    param_ranges['start_utc'] = (start_utc, start_utc)
    param_ranges['end_utc'] = (end_utc, end_utc)
    return param_ranges

def do_setup(start_utc, end_utc, birdie_config):
    """Initialize objects and arrays for birdie injection."""

    # Init ModuleView object
    mod = init_module(start_utc)
    bounding_box = birdie_utils.get_coord_bounding_box(mod.center_ra, mod.center_dec)
    birdie_utils.init_ra_dec_ranges(start_utc, end_utc, bounding_box)

    # Init array modeling the sky
    sky_array = init_sky_array(birdie_config['array_resolution'])


    # Init birdies and convolution kernel.
    param_ranges = init_birdie_param_ranges(start_utc, end_utc, bounding_box, birdie_config['param_ranges'])
    birdie_sources = init_birdies(birdie_config['num_birdies'], param_ranges)

    return mod, sky_array, birdie_sources


# Birdie simulation routine


def update_birdies(frame_utc, bounding_box, sky_array, birdie_sources):
    """Call the generate_birdie method on every BirdieSource object with an RA
    that may be visible by the given module."""
    birdies_in_view = False
    left = int(bounding_box[0][0] % 360)
    right = int(bounding_box[0][1] % 360) - 1
    if right < left:
        right += 360
    i = left
    #input(f'left={left}, right={right}')
    while i < right:
        for b in birdie_sources[i % 360]:
            point_added = b.generate_birdie(frame_utc, sky_array, bounding_box)
            birdies_in_view = birdies_in_view or point_added
        i += 1
    return birdies_in_view


def apply_psf(sky_array, sigma):
    """Apply a 2d gaussian filter to simulate optical distortion."""
    return gaussian_filter(sky_array, sigma=sigma)


def do_simulation(start_utc,
                  end_utc,
                  module,
                  sky_array,
                  birdie_sources,
                  birdie_config,
                  nframes,
                  integration_time,
                  fin,
                  fout,
                  noise_mean=0,
                  num_updates=10,
                  plot_images=False,
                  draw_sky_band=False):
    time_step = (end_utc - start_utc) / nframes
    print(time_step)
    frame_num = 0
    print(f'Start simulation of {round((end_utc - start_utc) / 60, 2)} minute file ({nframes} frames)'
          f'\n\tEstimated time to completion: {round(0.07 * nframes // 60)} min {round(0.07 * nframes % 60)} s')
    total_time = max_counter = 0
    s = time.time()
    t = start_utc
    while t < end_utc:
        noisy_img = np.random.poisson(noise_mean, 1024)
        raw_img, j = get_next_frame(fin)
        #print(j)
        #input(f"\t calculated timestamp={start_utc + frame_num * integration_time}, actual tv_sec={j['tv_sec']}")
        #input(raw_img)
        birdie_utils.show_progress(frame_num, raw_img, module, nframes, num_updates, plot_images)

        # Update module on-sky position.
        module.update_center_ra_dec_coords(t)
        bounding_box = birdie_utils.get_coord_bounding_box(module.center_ra, module.center_dec)

        # Update birdie signal points.
        sky_array.fill(0)
        center_ra = module.center_ra
        birdies_in_view = update_birdies(t, bounding_box, sky_array, birdie_sources)

        # Simulate image mode data
        blurred_sky_array = apply_psf(sky_array, sigma=birdie_config['psf_sigma'])
        module.simulate_all_pixel_fovs(blurred_sky_array, bounding_box, birdies_in_view, draw_sky_band)

        #def write_image_1D(f, img, img_size, bytes_per_pixel):
        #pff.write_image_1D(fout, module.add_birdies_to_image_array(noisy_img), 32, 2)
        max_counter = max(max_counter, max(module.simulated_img_arr))
        t += time_step
        frame_num += 1
    e = time.time()
    total_time += e - s
    avg_time = total_time / nframes
    print(f'\nNum sims = {nframes}, avg sim time = {round(avg_time, 5)}s, total sim time = {round(total_time, 4)}s')
    print(f'Max image counter value = {max_counter}')
    if draw_sky_band:
        # Plot a heatmap of the sky covered during the simulation.
        module.plot_sky_band()


def do_test_simulation(start_utc,
                  end_utc,
                  module,
                  sky_array,
                  birdie_sources,
                  birdie_config,
                  nframes,
                  noise_mean=0,
                  num_updates=10,
                  plot_images=False,
                  draw_sky_band=False):
    time_step = (end_utc - start_utc) / nframes
    frame_num = 0
    print(f'Start simulation of {round((end_utc - start_utc) / 60, 2)} minute file ({nframes} frames)'
          f'\n\tEstimated time to completion: {round(0.07 * nframes // 60)} min {round(0.07 * nframes % 60)} s')
    total_time = max_counter = 0
    s = time.time()
    t = start_utc
    while t < end_utc:
        noisy_img = np.random.poisson(noise_mean, 1024)
        birdie_utils.show_progress(frame_num, noisy_img, module, nframes, num_updates, plot_images)

        # Update module on-sky position.
        module.update_center_ra_dec_coords(t)
        bounding_box = birdie_utils.get_coord_bounding_box(module.center_ra, module.center_dec)

        # Update birdie signal points.
        sky_array.fill(0)
        center_ra = module.center_ra
        birdies_in_view = update_birdies(t, bounding_box, sky_array, birdie_sources)

        # Simulate image mode data
        blurred_sky_array = apply_psf(sky_array, sigma=birdie_config['psf_sigma'])
        module.simulate_all_pixel_fovs(blurred_sky_array, bounding_box, birdies_in_view, draw_sky_band)

        max_counter = max(max_counter, max(module.simulated_img_arr))
        t += time_step
        frame_num += 1
    e = time.time()
    total_time += e - s
    avg_time = total_time / nframes
    print(f'\nNum sims = {nframes}, avg sim time = {round(avg_time, 5)}s, total sim time = {round(total_time, 4)}s')
    print(f'Max image counter value = {max_counter}')
    if draw_sky_band:
        # Plot a heatmap of the sky covered during the simulation.
        module.plot_sky_band()


def test_simulation():
    start_utc = 1685417643
    end_utc = start_utc + 3600
    integration_time = 20e-1
    birdie_config = get_birdie_config('birdie_config.json')

    # Initialize objects
    module, sky_array, birdie_sources = do_setup(start_utc, end_utc, birdie_config)
    nframes = math.ceil((end_utc - start_utc) / integration_time)
    do_test_simulation(
        start_utc, end_utc,
        module, sky_array, birdie_sources,
        birdie_config,
        nframes,
        noise_mean=0,
        num_updates=20,
        plot_images=1,
        draw_sky_band=0
    )


def do_file(data_dir, run, analysis_dir, fin_name, params):
    #input(run)
    print('processing file ', fin_name)
    file_attrs = pff.parse_name(fin_name)
    module_id = file_attrs['module']
    module_dir = analysis_util.make_dir('%s/module_%s_with_birdies'%(analysis_dir, module_id))
    birdie_config = get_birdie_config('birdie_config.json')

    # Get start and end utc timestamps
    start_iso = pff.parse_name(run)['start']  # use the timestamp from the run directory name.
    run_complete_path = f'{data_dir}/{run}/run_complete'
    with open(run_complete_path) as f:
        end_iso = f.readline()  # use the timestamp in "data/$run/run_complete"
    start_utc = birdie_utils.iso_to_utc(start_iso)
    end_utc = birdie_utils.iso_to_utc(end_iso)
    integration_time = birdie_utils.get_integration_time(data_dir, run)
    print(f'start_utc={start_utc}, end_utc={end_utc}, integration_time={integration_time}us')

    # Get the number of image frames
    nframes = 1e6 * (end_utc - start_utc) / integration_time

    # Initialize objects
    module, sky_array, birdie_sources = do_setup(start_utc, end_utc, birdie_config)

    # Do simulation
    #input(f'{module_dir}/birdie-injection.{fin_name}')
    with open(f'{data_dir}/{run}/{fin_name}', 'rb') as fin:
        with open(f'{module_dir}/birdie-injection.{fin_name}', 'w+b') as fout:
            do_simulation(
                start_utc, end_utc,
                module, sky_array, birdie_sources,
                birdie_config,
                nframes,
                integration_time,
                fin,
                fout,
                noise_mean=0,
                num_updates=20,
                plot_images=True,
                draw_sky_band=True
            )


def do_run(data_dir, run, params, username):
    analysis_dir = analysis_util.make_dir('birdie_injection_test')#analysis_util.make_analysis_dir('birdie_injection', run)
    print('processing run', run)
    for f in os.listdir(f'{data_dir}/{run}'):
        if not pff.is_pff_file(f):
            continue
        if pff.pff_file_type(f) != 'img16':
            continue
        #input(pff.parse_name(run))
        do_file(data_dir, run, analysis_dir, f, params)
    analysis_util.write_summary(analysis_dir, params, username)




DATA_DIR = '/Users/nico/Downloads/test_data/data'
fname = 'start_2022-05-11T23/39/15Z.dp_1.bpp_2.dome_0.module_1.seqno_1.pff'


def main():
    #analysis_dir = analysis_util.make_analysis_dir('birdie_injection', run)
    params = {
        'seconds': 1
    }
    do_run(DATA_DIR, 'obs_Lick.start_2022-05-11T23:38:29Z.runtype_eng.pffd', params, 'nico')
    #test_simulation()


if __name__ == '__main__':
    print("RUNNING")
    main()
    print("DONE")
