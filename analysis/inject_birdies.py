"""
The program uses models of the modules in an observatory and the
celestial sphere to generate birdies and simulate image mode data for a single image file.

The output is a pff file of simulated image mode data containing birdies.

TODO:
    - File IO:
        - Add utility methods to import image mode files, read their metadata and image arrays, and write RAW + birdie frames.
        - Most important metadata:
            - Module ID, module orientation (alt-az + observatory GPS), integration time, start time, and end time.
    - Setup procedure:
        - Create or update birdie log file.
        - Create a file object for the simulated image file (birdies only).
    - Main loop
        - Check if we’ve reached EOF in any of the image mode files.

"""
import math
import time
import sys
import os
import numpy as np
import cProfile

import analysis_util
import birdie_utils
import birdie_simulation as birdie_sim

sys.path.append('../util')
import pff
import config_file
sys.path.append('../control')

np.random.seed(301)


def test_simulation():
    start_utc = 1685417643
    end_utc = start_utc + 5
    # integration time in usec
    integration_time = 20
    birdie_config = birdie_utils.get_birdie_config('birdie_config.json')

    f = '''birdie_sim.do_simulation(
        start_utc, end_utc,
        birdie_config,
        integration_time,
        noise_mean=0,
        num_updates=30,
        plot_images=False,
    )'''
    cProfile.runctx(f, globals(), locals(), sort='tottime')


def do_file(data_dir, run, analysis_dir, fin_name, params):
    print('processing file ', fin_name)
    file_attrs = pff.parse_name(fin_name)
    module_id = file_attrs['module']
    module_dir = analysis_util.make_dir('%s/module_%s_with_birdies'%(analysis_dir, module_id))
    birdie_config = birdie_utils.get_birdie_config('birdie_config.json')

    # Get start and end utc timestamps
    start_iso = pff.parse_name(run)['start']  # use the timestamp from the run directory name.
    run_complete_path = f'{data_dir}/{run}/run_complete'
    with open(run_complete_path) as f:
        end_iso = f.readline()  # use the timestamp in "data/$run/run_complete"
    # Get timing info
    start_utc = birdie_utils.iso_to_utc(start_iso)
    end_utc = birdie_utils.iso_to_utc(end_iso)
    integration_time = birdie_utils.get_integration_time(data_dir, run)
    print(f'start_utc={start_utc}, end_utc={end_utc}, integration_time={integration_time} us')

    # Do simulation
    #input(f'{module_dir}/birdie-injection.{fin_name}')
    with open(f'{data_dir}/{run}/{fin_name}', 'rb') as fin:
        with open(f'{module_dir}/birdie-injection.{fin_name}', 'w+b') as fout:
            birdie_sim.do_simulation(
                start_utc, end_utc,
                birdie_config,
                integration_time,
                fin=fin,
                fout=fout,
                noise_mean=0,
                num_updates=20,
                plot_images=True,
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
    #do_run(DATA_DIR, 'obs_Lick.start_2022-05-11T23:38:29Z.runtype_eng.pffd', params, 'nico')
    #exec('test_simulation()')
    test_simulation()

if __name__ == '__main__':
    print("RUNNING")
    main()
    #cProfile.runctx('main()', globals(), locals(), sort='tottime')
    print("DONE")
