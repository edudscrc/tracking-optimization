import matplotlib.pyplot as plt
import numpy as np
import re


def m_imshow(data, title, xlabel, ylabel, xticks, yticks, filename, extent, show):
    vmax = np.percentile(abs(data), 99.9)
    vmin = -vmax

    plt.rcParams.update({'font.size': 5})

    plt.figure(layout="constrained", dpi=300)
    plt.imshow(abs(data), aspect='auto', origin='lower', vmax=vmax, vmin=vmin, extent=extent)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.savefig(f'./plots_19000/{filename}', dpi=300)
    
    if show:
        plt.show()

    plt.close('all')


def m_plot(x_arr, data, title, xlabel, ylabel, xticks, filename, show):
    plt.rcParams.update({'font.size': 5})

    plt.figure(layout="constrained", dpi=300)
    plt.plot(x_arr, data, linewidth=0.5)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(xticks)
    plt.grid(True)
    plt.savefig(f'./plots_19000/{filename}', dpi=300)

    if show:
        plt.show()
        
    plt.close('all')


def load_data(filename):
    p = re.compile(r'(?:\w+)_ds(?:\d+)_(\d+)m_(\d+)m_tds(?:\d+)_(\d+)s_(\d+)s_fs(?:\d+)Hz')
    m = re.search(p, filename)

    spatial_position_start = int(m.group(1))
    spatial_position_end = int(m.group(2))
    time_start = int(m.group(3))
    time_end = int(m.group(4))

    spatial_position_end -= spatial_position_start
    spatial_position_start -= spatial_position_start

    time_end -= time_start
    time_start -= time_start

    data = np.load(f'./{filename}.npy')

    return data, spatial_position_start, spatial_position_end, time_start, time_end
