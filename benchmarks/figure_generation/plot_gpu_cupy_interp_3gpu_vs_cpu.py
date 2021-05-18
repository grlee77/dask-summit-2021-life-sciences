import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['font.size'] = 16
rcParams['axes.titleweight'] = 'bold'


labels = [
    'affine transform\n(3D, order=0)',
    'affine transform\n(3D, order=1)',
    'affine transform\n(3D, order=3)',
    'affine transform\n(3D, order=5)',
    # 'map coordinates\n(3D, order=1)',
    ]

greg_cpu_means = [  # Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
    0.0601,  # 'affine transform (3D, order=0)',
    0.1725,  # 'affine transform (3D, order=1)',
    1.4531,  # 'affine transform (3D, order=3)',
    4.0232,  # 'affine transform (3D, order=5)',
]

greg_dask_cpu_means = [  # Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz
    0.0590,  # 'affine transform (3D, order=0)',
    0.0841,  # 'affine transform (3D, order=1)',
    0.3200,  # 'affine transform (3D, order=3)',
    0.7143,  # 'affine transform (3D, order=5)',
]

gpu_means_1080ti = [
    0.0008766,  # 'affine transform (3D, order=0)',
    0.001235,   # 'affine transform (3D, order=1)',
    0.01149,    # 'affine transform (3D, order=3)',
    0.02505,    # 'affine transform (3D, order=5)',
]


gpu_means_V100 =  [  # V100 results
    0.0002528,   # 'affine transform (3D, order=0)',
    0.0002610,   # 'affine transform (3D, order=1)',
    0.003318,    # 'affine transform (3D, order=3)',
    0.007357,    # 'affine transform (3D, order=5)',
   #  0.0004062,   # 'map coordinates (3D, order=1)'
]

gpu_means_A100 =  [  # A100 results
    0.0002947,   # 'affine transform (3D, order=0)',
    0.0003364,   # 'affine transform (3D, order=1)',
    0.002134,    # 'affine transform (3D, order=3)',
    0.003948,    # 'affine transform (3D, order=5)',
    # 0.0003164,   # 'map coordinates (3D, order=1)'
]


x = np.arange(len(labels))  # the label locations
log_scale = True

n_gpus = 2  # number of GPU models to include in the plot. Must be between 1 and 3

if n_gpus == 0:
    # Plot A100 results vs. CPU only
    width = 0.42  # the width of the bars
    figsize = [15.42, 6.58]
    fig, ax = plt.subplots(figsize=figsize)

    rects1 = ax.bar(x - width / 2, greg_cpu_means, width, label='SciPy (CPU): Intel Core i9-7900X', color='#FFB043')
    rects2 = ax.bar(x + width / 2, greg_dask_cpu_means, width, label='CuPy (GPU): NVIDIA A100', color='#6A16F8')

elif n_gpus == 1:
    # A100 and V100 results vs. CPU
    figsize = [20, 7.54]
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.28  # the width of the bars
    rects1 = ax.bar(x - width, greg_cpu_means, width, label='SciPy (CPU): Intel Core i9-7900X', color='#FFB043')
    rects2 = ax.bar(x, greg_dask_cpu_means, width, label='dask-image (CPU): Intel Core i9-7900X', color='#2CC9D9')
    rects3 = ax.bar(x + width, gpu_means_1080ti, width, label='CuPy (GPU): NVIDIA GTX 1080-Ti', color='#6A16F8')

elif n_gpus == 2:
    # A100, V100 and GTX-1080 Ti results vs. CPU
    figsize = [28.87, 9.36]
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.225  # the width of the bars
    rects1 = ax.bar(x - 1.5 * width, greg_cpu_means, width, label='SciPy (CPU): Intel Core i9-7900X', color='#FFB043')
    rects2 = ax.bar(x - 0.5 * width, greg_dask_cpu_means, width, label='dask-image (CPU): Intel Core i9-7900X', color='#D028C8')
    rects3 = ax.bar(x + 0.5 * width, gpu_means_1080ti, width, label='CuPy (GPU): NVIDIA GTX 1080-Ti', color='#2CC9D9')
    rects4 = ax.bar(x + 1.5 * width, gpu_means_A100, width, label='CuPy (GPU): NVIDIA A100', color='#6A16F8')
else:
    raise NotImplementedError("only 3 GPU models have results stored in this script")


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Duration (s)', fontdict=dict(fontweight='bold', fontsize=16))
ax.set_title('Duration (SciPy vs. CuPy)')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontdict=dict(fontweight='bold', fontsize=16))
ax.set_ylim([0, 5.0])
ax.legend()

if log_scale:
    ax.set_yscale('log')
    #ax.set_ylim([0, 900.0])
    ax.set_ylim([0.0001, 30.0])
    max_label_y = np.inf
else:
    max_label_y = 20
    ax.set_ylim([0, 30.0])


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if True:
            if height < 0.001:
                v  = '{:0.2e}'.format(height)
            else:
                v  = '{:0.3g}'.format(height)
        else:
            v  = '{}'.format(height)
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

accels = np.array(greg_cpu_means) / np.array(gpu_means_A100)
[print(f"{name}: acceleration = {acc}") for name, acc in zip(labels, accels)]
