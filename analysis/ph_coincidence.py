#! /usr/bin/env python3

"""

"""
import getpass
import itertools

from search_ph import *
from analysis_util import *

# Control


def get_file_path(file_name):
    """Return the file path to file_name."""
    return f'{DATA_IN_DIR}/{file_name}'




#a_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff'
#b_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff'
#a_fname = 'start_2022-06-28T19_06_14Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff'

# July 19
#a_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
#a_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
#b_fname = 'start_2022-07-20T06_44_48Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome

# July 20
a_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_1.seqno_0.pff' # astrograph 1
#b_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_254.seqno_0.pff' # astrograph 2
b_fname = 'start_2022-07-21T06_03_03Z.dp_ph16.bpp_2.dome_0.module_3.seqno_0.pff' # nexdome


def do_run(vol, run, params, username):
    analysis_dir = None# make_analysis_dir(ANALYSIS_TYPE_PULSE_HEIGHT_COINCIDENCE, vol, run)

    ph_files = dict()
    for f in os.listdir('%s/data/%s' % (vol, run)):
        if not pff.is_pff_file(f): continue
        t = pff.pff_file_type(f)
        if t == 'ph16':
            bytes_per_pixel = 2
        elif t == 'ph8':
            bytes_per_pixel = 1
        else:
            continue

        file_path = '%s/data/%s/%s' % (vol, run, f)
        if os.path.getsize(file_path) != 0:
            file_attrs = pff.parse_name(f)
            module = file_attrs['module']
            if module in ph_files:
                raise Warning(f'Expected exactly one ph file for module {module} but found more than one.')
            if params['modules'] == 'all_modules':
                ph_files[module] = f
            else:
                if module in params['modules']:
                    ph_files[module] = f
    module_pairs = list(itertools.combinations(ph_files.keys(), 2))
    for module_a, module_b in module_pairs:
        a_fname, b_fname = ph_files[module_a], ph_files[module_b]
        fpair_dir = make_dir(f'{analysis_dir}/module_{module_a}.module_{module_b}')
        do_coincidence_search(
            fpair_dir,
            a_fname,
            b_fname,
            bytes_per_pixel,
            params['max_time_diff'],
            params['threshold_max_adc'],
            params['max_group_time_diff'],
            verbose=params['verbose'],
            save_fig=params['save_fig']
        )
    write_summary(analysis_dir, params, username)


def main():
    params = {
        'modules': [1, 3],
        'max_time_diff': 500,
        'threshold_max_adc': 0,
        'max_group_time_diff': 300,
        'verbose': True,
        'save_fig': False,
    }
    run = None
    vol = None
    username = None
    argv = sys.argv
    i = 1
    while i < len(argv):
        option = argv[i].replace('--', '', 1)
        if option in params:
            i += 1
            params[option] = argv[i]
        elif option == 'run':
            i += 1
            run = argv[i]
        elif option == 'vol':
            i += 1
            vol = argv[i]
        elif option == 'username':
            i += 1
            username = argv[i]
        else:
            raise Warning(f'unrecognized input: "--{option} {argv[i]}". Options have the form: "--[option]".')
        i += 1

    if not run:
        raise Exception('no run specified')
    if not vol:
        raise Exception('no volume specified')
    if not username:
        username = getpass.getuser()

    do_run(vol, run, params, username)


if __name__ == '__main__':
    main()
