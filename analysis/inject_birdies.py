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
import shutil
import numpy as np
import cProfile

import birdie_utils
import birdie_simulation as birdie_sim
import analysis_util

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


def do_file(data_dir, run, birdie_run, fin_name, sequence_num, params):
    """Inject birdies into a single file."""
    file_attrs = pff.parse_name(fin_name)
    birdie_config = birdie_utils.get_birdie_config('birdie_config.json')

    # Get start and end utc timestamps
    start_iso = pff.parse_name(run)['start']  # use the timestamp from the run directory name.
    with open(f'{data_dir}/{run}/run_complete') as f:
        end_iso = f.readline()  # use the timestamp in "data/$run/run_complete"

    # Get timing info
    start_utc = birdie_utils.iso_to_utc(start_iso)
    end_utc = birdie_utils.iso_to_utc(end_iso)
    integration_time = birdie_utils.get_integration_time(data_dir, run)
    print(f'start_utc={start_utc}, end_utc={end_utc}, integration_time={integration_time} us')
    dt = end_utc-start_utc
    print(f'recording_time = {int(dt // 60)}:{int(dt%60)}')

    fout_name = fin_name.replace('.pff', '') + f'.birdie_{sequence_num}.pff'
    fout_path = f'{data_dir}/{birdie_run}/{fout_name}'
    fin_path = os.path.abspath(f'{data_dir}/{run}/{fin_name}')

    #print(f"fin= {fin_path},\nfout={fout_path}")
    # Do simulation
    with open(fin_path, 'rb') as fin:
        with open(fout_path, 'w+b') as fout:
            # Create a copy of the file fin_name for birdie injection.
            print(f'Copying:\n\tFrom:\t{fin_name}\n\tTo:\t\t{fout_name}')
            shutil.copy(fin_path, fout_path)

            return
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


def do_run(data_dir, run, fin_name, params):
    """Run birdie injection on a real observing run."""
    if not pff.is_pff_dir(run):
        print(f'"{run}" is not a pff directory')
        return
    print('Injecting birdies into run ', run)
    # Get sequence number
    run_name_without_pffd = run.replace('.pffd', '')
    sequence_num = 0
    for f in os.listdir(f'{data_dir}'):
        if run_name_without_pffd in f.replace('.pffd', ''):
            run_attrs = pff.parse_name(f)
            if 'birdie' in run_attrs:
                sequence_num += 1
    #print(f'sequence_num = {sequence_num}')
    # Create directory for run + birdie data
    birdie_run = run_name_without_pffd + f'.birdie_{sequence_num}.pffd'
    analysis_util.make_dir(f'{data_dir}/{birdie_run}')
    for f in os.listdir(f'{data_dir}/{run}'):
        if not pff.is_pff_file(f):
            continue
        if pff.pff_file_type(f) not in ('img16', 'img8'):
            continue
        do_file(data_dir, run, birdie_run, fin_name, sequence_num, params)
    print(f'Finished injecting birdies.')



DATA_DIR = '/Users/nico/Downloads/test_data'
RUN = 'obs_Lick.start_2022-10-13T00:08:12Z.runtype_eng.pffd'
fname = 'start_2022-10-13T00:08:28Z.dp_img8.bpp_1.module_1.seqno_0.pff'


def main():
    params = {
        'seconds': 1
    }
    do_run(DATA_DIR, RUN, fname, params)
    #exec('test_simulation()')
    #test_simulation()


if __name__ == '__main__':
    print("RUNNING")
    main()
    #cProfile.runctx('main()', globals(), locals(), sort='tottime')
    print("DONE")
