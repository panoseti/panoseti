#! /usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

# Quabo (4) x Pixels (256) x Raw adc value range (2**12)
shape = (4, 256, 2**12)
threshold_pe = 0
counts = np.zeros(shape, dtype='uint64')
print(f'Array size: {counts.size:,}')

#np.random.seed(seed=10)


def style_fig(fig):
    # Clean the plot
    fig.suptitle('Log-scaled cumulative pulse height distributions')


def style_ax(fig, ax, quabo):
    ax.grid(True)
    ax.set_xlabel('PE')
    ax.set_ylabel('Count')
    #ax.set_yscale('log')
    ax.set_title('Quabo {0}'.format(quabo))
    ax.axvline(x=threshold_pe, color='g', label=f'Threshold pe = {threshold_pe}')
    #ax.legend(loc='upper right')#, bbox_to_anchor=(1, 0.5))


def add_hover_tooltip(fig, ax, plot):
    """
    Adds a tooltip to show the names of graphs on a mouse down event.
    See https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-to-a-plot
    """
    annot = ax.annotate('aksjdfhasjd', xy=(0,0), xytext=(10,10), textcoords='offset points',
                    bbox=dict(boxstyle='round', fc='w'),
                    arrowprops=dict(arrowstyle='->'))
    annot.set_visible(False)
    
    def update_annot(click_event):
        mouse_x, mouse_y = click_event.xdata, click_event.ydata
        annot.xy = mouse_x, mouse_y
        annot.set_y(annot.get_position()[0] + plot.get_gid() / shape[1] * 10)
        text = plot.get_label()
        #print(text[6:], end='__')
        annot.set_text(text)
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


def do_save_data(filepath):
    """Save the counts array to a binary file."""
    np.save(filepath, counts)


def do_load_data(filepath):
    """Returns an array loaded from a binary file."""
    return np.load(filepath)


def test(num_images, save_data=True, data_gen=True, filepath='./ph_distrib_counts.npy'):
    """Generate test data and plots."""
    print('**TEST**')
    np.set_printoptions(threshold=sys.maxsize)
    if data_gen:
        for quabo in range(shape[0]):
                print(f'Populating data array for Quabo {quabo}... ', end='')
                for x in range(num_images):
                    #test_data = np.ones(shape[1]).astype('int')*2
                    #test_data = np.random.geometric(0.005, size=shape[1])
                    #test_data = np.random.poisson(lam=10, size=shape[1])
                    #test_data = np.random.normal(132, 700, size=shape[1])
                    test_data = np.random.randint(low=0, high=2**12, size=shape[1])
                    update_counts(quabo, test_data)
                print('Done!')     
    else:
        global counts
        try:
            counts = do_load_data(filepath)
        except FileNotFoundError as ferr:
            print(f'** {filepath} is not a valid path.') 
            raise
        if counts.shape != shape:
            msg = f'The shape of the array at {filepath} is {counts.shape},\n'
            msg += f' which is different from the expected shape: {shape}.'
            raise Warning(msg)
    if save_data:
        try:
            do_save_data(filepath)
        except FileNotFoundError as ferr:
            print(f'** {filepath} is not a valid path.') 
            raise
    draw_plt()


test(10**3, data_gen=True)
