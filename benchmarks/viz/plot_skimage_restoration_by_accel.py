import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np


font_size = 18
rcParams['font.size'] = font_size
rcParams['axes.titleweight'] = 'bold'


labels = [
    'wavelet denoising\n(BayesShrink)',
    'TV Denoising\n(Bregman)',
    'TV Denoising\n(Chambolle)',
    'Bilteral Filtering',
    'Non-local Means',
    ]

greg_cpu_means = [  # Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
    0.783,  # 'wavelet denoising\n(BayesShrink)',
    2.33,   # 'TV Denoising\n(Bregman)',
    12.1,   # 'TV Denoising\n(Chambolle)',
    3.79,   # 'Bilteral Filtering',
    8.37,   # 'Non-local Means',
]

greg_dask_cpu_means_v18 = [  # Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
    0.453,  # 'wavelet denoising\n(BayesShrink)',
    3.200,  # 'TV Denoising\n(Bregman)',  #
    3.98,   # 'TV Denoising\n(Chambolle)',
    5.23,   # 'Bilteral Filtering',
    1.250,   # 'Non-local Means',
]

greg_dask_cpu_means_main = [  # Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
    0.453,  # 'wavelet denoising\n(BayesShrink)',
    0.529,  # 'TV Denoising\n(Bregman)',  #
    3.98,   # 'TV Denoising\n(Chambolle)',
    0.811,  # 'Bilteral Filtering',
    1.250,  # 'Non-local Means',
]


x = np.arange(len(labels))  # the label locations
log_scale = False

n_gpus = 0  # number of GPU models to include in the plot. Must be between 1 and 3

ones = np.ones_like(greg_cpu_means)
accels_dask = np.array(greg_cpu_means) / np.array(greg_dask_cpu_means_main)
accels_daskv18 = np.array(greg_cpu_means) / np.array(greg_dask_cpu_means_v18)
# accels_a100 = np.array(greg_cpu_means) / np.array(gpu_means_A100)
# [print(f"{name}: acceleration = {acc}") for name, acc in zip(labels, accels_a100)]

if n_gpus == 0:
    # Plot Dask results vs. SciPy
    width = 0.43  # the width of the bars
    figsize = [15, 5.58]
    fig, ax = plt.subplots(figsize=figsize)

    rects1 = ax.bar(x - width / 2, ones, width, label='without apply_parallel', color='#FFB043')
    rects2 = ax.bar(x + width / 2, accels_dask, width, label='with apply_parallel', color='#6A16F8')

else:
    raise NotImplementedError("only 3 GPU models have results stored in this script")


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance Gain', fontdict=dict(fontweight='bold', fontsize=font_size))
ax.set_title('Using skimage.util.apply_parallel with Denoising Algorithms')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontdict=dict(fontweight='bold', fontsize=font_size))
ax.set_ylim([0, 5.0])
ax.legend()

if log_scale:
    ax.set_yscale('log')
    #ax.set_ylim([0, 900.0])
    ax.set_ylim([0, 50.0])
    ax.set_yticks([0.1, 1.0, 10.0, 100.0, 1000.0])
    ax.set_yticklabels(['0.1x', '1x', '10x', '100x', '1000x'])
    max_label_y = np.inf
else:
    max_label_y = 15
    ax.set_ylim([0, 10.0])


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 1e3:
            v  = '{:0.4g}x'.format(height)
        else:
            v  = '{:0.3g}x'.format(height)
        ax.annotate(v,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
if n_gpus >= 0:
    autolabel(rects2)
if n_gpus >= 1:
    autolabel(rects3)
if n_gpus >= 2:
    autolabel(rects4)

fig.tight_layout()

plt.show()
