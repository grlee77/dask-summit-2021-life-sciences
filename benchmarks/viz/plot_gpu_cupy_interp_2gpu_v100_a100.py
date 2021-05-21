import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['font.size'] = 16
rcParams['axes.titleweight'] = 'bold'

# colors.rgb2gray
# filters.gaussian_filter
# filters.median_filter
# morphology.binary_erosion
# transform.resize
# feature.canny
# measure.label
# restoration.denoise_tv_chambolle
# segmentation.random_walker
# registration.phase_cross_correlation
# restoration.richardson_lucy
# metrics.structural_similarity

labels = [
    'affine transform\n(3D, order=0)',
    'affine transform\n(3D, order=1)',
    'affine transform\n(3D, order=3)',
    'affine transform\n(3D, order=5)',
    'map coordinates\n(3D, order=1)',
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

# ben_cpu_means = [  # Intel(R) Xeon(R) Platinum 8168 CPU @ 2.70GHz
#    0.2466,     # rgb2hed
#    2.3688,     # median (5, 5)
#    0.2444,     # gaussian_filter (sigma=4)
#    0.1237,   # binary_erosion (connectivity=2)
#    2.147,    # canny
# #   1.031,     # resize (order=3)
#    0.6798,     # structure_tensor
#    0.0785,    # label (connectivity=2)
# ]

gpu_means_1080ti = [
    0.0008766,  # 'affine transform (3D, order=0)',
    0.001235,   # 'affine transform (3D, order=1)',
    0.01149,    # 'affine transform (3D, order=3)',
    0.02505,    # 'affine transform (3D, order=5)',
]


gpu_means_V100 =  [
    0.0002528,  # 'affine transform (3D, order=0)',
    0.0002610,   # 'affine transform (3D, order=1)',
    0.003318,    # 'affine transform (3D, order=3)',
    0.007357,    # 'affine transform (3D, order=5)',
]


gpu_means_A100 =  [
    0.0002947,   # 'affine transform (3D, order=0)',
    0.0003364,   # 'affine transform (3D, order=1)',
    0.002134,    # 'affine transform (3D, order=3)',
    0.003948,    # 'affine transform (3D, order=5)',
]


x = np.arange(len(labels))  # the label locations
width = 0.28  # the width of the bars
log_scale = True
if log_scale:
    fig, ax = plt.subplots(figsize=[20,  6.54])  # [27.31,  5.87])
else:
    fig, ax = plt.subplots(figsize=(20, 5.8))
rects1 = ax.bar(x - width, greg_cpu_means, width, label='SciPy ndimage (CPU): Intel Core i9-7900X', color='#FFB043')
rects2 = ax.bar(x, gpu_means_V100, width, label='CuPy (GPU): NVIDIA V100', color='#2CC9D9')
rects3 = ax.bar(x + width, gpu_means_A100, width, label='CuPy (GPU): NVIDIA A100', color='#6A16F8')

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
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()

accels = np.array(greg_cpu_means) / np.array(gpu_means_A100)
[print(f"{name}: {acc}") for name, acc in zip(labels, accels)]
 #array([659.29487179, 423.52941176, 232.44397012, 491.44144144,
 #       55.18823529,  96.50943396, 179.36708861,  33.85869565])

if False:
    def autolabel_accels(rects1, accels):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect, acc in zip(rects1, accels):
            height = 20
            ax.annotate('R = {:0.3g}'.format(acc),
                        xy=(rect.get_x() + rect.get_width(), height),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='center',
                        fontweight='bold')

    autolabel_accels(rects1, accels)

