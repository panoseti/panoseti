#! /usr/bin/env python3

"""
Generates cumulative pulse height distributions.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../util')
import pff

# Quabo (4) x Pixels (256) x Raw adc value range (2**12)
shape = (4, 256, 2**12)
threshold_pe = 0
mod_num = None
counts = np.zeros(shape, dtype='uint64')
DATA_OUT_DIR = '.ph_cdist_data'

#np.random.seed(seed=10)

def style_fig(fig):
    # Clean the plot
    fig.suptitle(f'Log-scaled cumulative pulse height distributions for module {mod_num} (>= {threshold_pe} pe)')


def style_ax(fig, ax, quabo):
    ax.grid(True)
    ax.set_xlabel('Pulse Height (Raw ADC)')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_title('Quabo {0}'.format(quabo))
    ax.axvline(x=threshold_pe, color='g', label=f'Threshold={threshold_pe}')
    #ax.legend(loc='upper right')#, bbox_to_anchor=(1, 0.5))


def add_hover_tooltip(fig, ax, plot):
    """
    Adds a tooltip to show pixel labels on a mouse down event.
    See https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-to-a-plot
    """
    annot = ax.annotate('', xy=(0,0), xytext=(10,10), textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='w'),
                        arrowprops=dict(arrowstyle='->'))
    annot.set_visible(False)
    
    def update_annot(click_event):
        mouse_x, mouse_y = click_event.xdata, click_event.ydata
        annot.xy = mouse_x, mouse_y
        text = plot.get_label()
        #print(text[6:], end='__')
        annot.set_text(text)
        annot.set_y(annot.get_position()[0] + plot.get_gid() / shape[1] * 10)
        annot.get_bbox_patch().set_alpha(0.4)

    def on_click(click_event):
        vis = annot.get_visible()
        if click_event.inaxes == ax:
            cont = plot.contains(click_event)
            if cont[0]:
                update_annot(click_event)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
                    
    def on_key_press(press_event):
        if press_event.key == ' ' and annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()
            
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('button_press_event', on_click)


def create_pixel_plt(fig, ax, data, pixel):
    """
    Plots a log-scaled cumulative distribution of the number of pulse height
    events against raw adc values in the range (0..2^12-1)
    for one pixel.
    """
    if np.count_nonzero(data) > 0:
        n_bins = np.arange(threshold_pe, shape[2] + 1)
        sc = ax.stairs(data, n_bins, fill=False, color='grey', label=f'Pixel {pixel}', gid=pixel)
        add_hover_tooltip(fig, ax, sc)
        

def create_quabo_plt(fig, ax, quabo):
    """Plots the distribution for each pixel in a quabo."""
    style_ax(fig, ax, quabo)
    for pixel in range(shape[1]):
        data = counts[quabo][pixel]
        # Only use data greater than or equal to threshold_pe
        data = data[threshold_pe:]
        # Create cumulative distribution
        data = data[::-1].cumsum()[::-1]
        create_pixel_plt(fig, ax, data, pixel)
    

def draw_plt():
    """Draws a subplot for each quabo."""
    fig, axs = plt.subplots(2,2, figsize=(8, 4))
    for quabo in range(shape[0]):
        print('Graphing Quabo {0}... '.format(quabo), end='')
        ax = axs[quabo % 2][quabo // 2]
        create_quabo_plt(fig, ax, quabo)
        print('Done!')
    style_fig(fig)
    fig.tight_layout()
    plt.show()
    

def update_counts(quabo, img_array):
    """
    Identifies the pixels in img_array that have a pe value above threshold_pe and,
    in the counts array, increments the tally for that pe value in the corresponding
    pixel.
    """
    pixels_above_threshold = np.nonzero(img_array >= threshold_pe)[0]
    for pixel in pixels_above_threshold:
        pe = img_array[pixel]
        if pe >= shape[2]:
            pe = shape[2] - 1
        counts[quabo][pixel][pe] += 1


def do_save_data(fname):
    """Save the counts array to a binary file."""
    os.system(f'mkdir -p {DATA_OUT_DIR}')
    filepath = f'{DATA_OUT_DIR}/{fname}.npy'
    print(f'Saving data to {filepath}')
    np.save(filepath, counts)


def do_load_data(filepath):
    """Returns an array loaded from a binary file."""
    if filepath[-4:] != '.npy':
        raise Warning(f'"{filepath}" is not a numpy binary file (.npy)')
    try:
        global counts
        counts = np.load(filepath)
    except FileNotFoundError as ferr:
        print(f'** {filepath} is not a valid path.')
        raise
    if counts.shape != shape:
        msg = f'The shape of the array at {filepath} is {counts.shape},\n'
        msg += f' which is different from the expected shape: {shape}.'
        raise Warning(msg)


def plot_data_from_npy(in_path):
    do_load_data(in_path)
    draw_plt()


def process_file(fname, img_size, bytes_per_pixel):
    with open(fname, 'rb') as f:
        i = 0
        while True:
            j = None
            # Deal with EOF bug in pff.read_json
            try:
                j = pff.read_json(f)
            except Exception as e:
                if repr(e)[:26] == "Exception('bad type code',":
                    print('\nreached EOF')
                    return
                else:
                    print("ERROR BRANCH")
                    print(f'"{repr(e)}"')
                    raise
            if not j:
                print('\nreached EOF')
                return
            print(f'Processed up to frame {i}.', end='\r')
            # show_pff.print_json(j.encode(), is_ph, verbose)
            img = pff.read_image(f, img_size, bytes_per_pixel)
            j = json.loads(j.encode())
            quabo_num = j['quabo_num']
            img_arr = np.array(img)
            update_counts(quabo_num, img_arr)
            i += 1


def do_test(num_images=10**3, save_data=True, data_gen=True, fname='test_ph_cum_dist'):
    """Generate test data and plots."""
    print('**TEST**')
    print(f'Array size: {counts.size:,}')
    #np.set_printoptions(threshold=sys.maxsize)
    if data_gen:
        for quabo in range(shape[0]):
            print(f'Generating test data for Quabo {quabo}... ', end='')
            for x in range(num_images):
                #test_data = np.ones(shape[1]).astype('int')*2
                test_data = np.random.geometric(0.005, size=shape[1])
                #test_data = np.random.poisson(lam=1000, size=shape[1])
                #test_data = np.random.normal(132, 700, size=shape[1])
                #test_data = np.random.randint(low=0, high=2**12, size=shape[1])
                update_counts(quabo, test_data)
            print('Done!')
        if save_data:
            do_save_data(fname)
    else:
        do_load_data(fname)
    draw_plt()


def usage():
    msg = "usage: ph_cdist.py process <options> file \t\t\tprocess ph data from a .pff file"
    msg += "\n   or: ph_cdist.py load <options> [--show-data] file \t\tplot processed ph data from a .npy file"
    msg += "\n   or: ph_cdist.py test \t\t\t\t\tgenerate test data and plots"
    msg += "\n\noptions:"
    msg += "\n\t--set-threshold <integer 0..4095>" + '\t' * 3 + 'set the minimum pe threshold'
    msg += "\n\t--no-show-plot" + '\t' * 6 + 'hide plots'
    print(msg)

def main():
    i = 1
    global mod_num
    global threshold_pe
    cmds = ['test', 'process', 'load']
    cmd = None
    ops = {
        '--set-threshold': None,
        '--no-show-plot': False,
        '--show-data': False,
    }
    fname = None
    # Process CLI commands and options
    argv = sys.argv
    while i < len(argv):
        if argv[i] in cmds:
            if cmd:
                'more than one command given'
                usage()
                return
            cmd = argv[i]
        elif argv[i] in ops:
            if argv[i] == '--set-threshold':
                i += 1
                if i >= len(argv):
                    print('must supply a number')
                    usage()
                    return
                ops['--set-threshold'] = argv[i]
            else:
                ops[argv[i]] = True
        elif i == len(argv) - 1:
            fname = argv[i]
            if not os.path.isfile(fname):
                print(f'{fname} may not be a valid file, or has a bad path')
                usage()
                return
        else:
            print(f'bad input: "{argv[i]}"')
            usage()
            return
        i += 1
    # Dispatch commands and options
    if val := ops['--set-threshold']:
        if val is not None and val.isnumeric():
            threshold_pe = int(val)
        else:
            print(f'"{val}" is not a valid integer')
            usage()
            return
    if cmd is None:
        print('bad command')
        usage()
        return
    elif cmd == 'test':
        do_test()
        return

    if fname is None:
        if cmd == 'load' and ops['--show-data']:
            for f in sorted(os.listdir(DATA_OUT_DIR)):
                print(f'{DATA_OUT_DIR}/{f}')
            return
        usage()
        return
    else:
        parsed = pff.parse_name(fname)
        mod_num = parsed['module']
    if cmd == 'load':
        do_load_data(fname)
    elif cmd == 'process':
        # Get data mode
        if fname == 'img':
            dp = 'img16'
        elif fname == 'ph':
            dp = 'ph16'
        else:
            dp = parsed['dp']
        # Get file metadata
        if dp == 'img16' or dp == '1':
            image_size = 32
            bytes_per_pixel = 2
            is_ph = False
        elif dp == 'ph16' or dp == '3':
            image_size = 16
            bytes_per_pixel = 2
            is_ph = True
        else:
            raise Exception("bad data product %s" % dp)
        # Process the data if fname is a ph file.
        if is_ph:
            process_file(fname, image_size, bytes_per_pixel)
            do_save_data(fname[:-4])
        else:
            raise Warning(f'{fname} is not a ph packet')
    # Draw plots if passed the option --show-plot
    if not ops['--no-show-plot']:
        draw_plt()

if __name__ == '__main__':
    main()
