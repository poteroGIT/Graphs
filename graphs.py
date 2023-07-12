import matplotlib.pyplot as plt
import numpy as np


plt.style.use('Solarize_Light2')
import matplotlib as mpl
params = {'xtick.labelsize': 20, 'ytick.labelsize': 20}
mpl.rcParams.update(params)

font1 = {'family': 'sans serif', 'color': 'darkred', 'size': 22}
font2 = {'family': 'sans-serif', 'color': 'saddlebrown', 'size': 26}
labels = ['CTGAN', 'CopulaGAN', 'TVAE']
labels = ['CTGAN', 'CopulaGAN', 'TVAE', 'Promedio']
colors = ['chocolate', 'sienna', 'maroon']
colors = ['chocolate', 'sienna', 'maroon', 'tomato']

cs_ct = np.array([0.9998, 0.99985, 0.99968, 0.99981, 0.99983])
cs_co = np.array([0.9989, 0.9996, 0.99972, 0.99965, 0.9984])
cs_tv = np.array([0.99999, 0.99998, 0.99996, 0.99998, 0.999965])

ks_ct = np.array([0.9, 0.83, 0.843, 0.846, 0.859])
ks_co = np.array([0.907, 0.877, 0.885, 0.859, 0.906])
ks_tv = np.array([0.944, 0.93, 0.916, 0.91, 0.932])

ld_ct = np.array([0.485, 0.5, 0.662, 0.693, 0.48])
ld_co = np.array([0.41, 0.59, 0.47, 0.6, 0.44])
ld_tv = np.array([0.22, 0.24, 0.49, 0.25, 0.306])

sd_ct = np.array([0.89, 0.912, 0.915, 0.892, 0.89])
sd_co = np.array([0.915, 0.916, 0.925, 0.907, 0.911])
sd_tv = np.array([0.749, 0.795, 0.81, 0.748, 0.79])

ml_ct = np.array([0.84, 0.87, 0.873, 0.865, 0.868])
ml_co = np.array([0.81, 0.815, 0.83, 0.845, 0.65])
ml_tv = np.array([0.89, 0.888, 0.91, 0.885, 0.925])


av_ct = []
av_co = []
av_tv = []
av_av = []

for j in range(len(ml_ct)):
    ct = (cs_ct[j] + ks_ct[j] + ld_ct[j] + sd_ct[j] + ml_ct[j])/len(ml_ct)
    co = (cs_co[j] + ks_co[j] + ld_co[j] + sd_co[j] + ml_co[j])/len(ml_ct)
    tv = (cs_tv[j] + ks_tv[j] + ld_tv[j] + sd_tv[j] + ml_tv[j])/len(ml_ct)
    av = (ct + co + tv)/3

    av_ct.append(ct)
    av_co.append(co)
    av_tv.append(tv)
    av_av.append(av)


ypoints = av_ct
ypoints2 = av_co
ypoints3 = av_tv

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

# Plot 1 - Line plot
ax1.plot(ypoints, color=colors[0], marker = 'o', ms = 13, ls = '--', linewidth = '4', label = labels[0])
ax1.plot(ypoints2, color=colors[1], marker = 'D', ms = 13, ls = '--', linewidth = '4', label = labels[1])
ax1.plot(ypoints3, color=colors[2], marker = 's', ms = 13, ls = '--', linewidth = '4', label = labels[2])
ax1.plot(av_av, color='tomato', marker = 'P', ms = 14, ls = '--', linewidth = '4', label = labels[3])

#plt.title("Nombre Métrica\n", fontdict=font1)
x_labels = ['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4', 'Model 5']
ax1.set_xticks(range(len(x_labels)))
ax1.set_xticklabels(x_labels)
ax1.set_xlabel('Modelos entrenados', fontdict = font2)
ax1.set_ylabel('Resultado Métrica\n', fontdict=font2)
ax1.grid()
ax1.grid(linewidth=2)
ax1.legend(fontsize=13, loc='best')

# Plot 2 - Boxplot
# bplot = ax2.boxplot([ypoints, ypoints2, ypoints3], patch_artist=True, labels=labels)
bplot = ax2.boxplot([ypoints, ypoints2, ypoints3, av_av], patch_artist=True, labels=labels)

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

colors = ['sienna', 'maroon', 'chocolate', 'brown']
# colors = ['sienna', 'maroon', 'chocolate']
for median, color in zip(bplot['medians'], colors):
    median.set(color =color, linewidth=3)

ax2.set_xlabel('Modelos de generación', fontdict = font2)

plt.show()

# plt.savefig('CStest.png', format="png")
# plt.savefig('KStest.png', format="png")
# plt.savefig('LogDet.png', format="png")
# plt.savefig('SVCDet.png', format="png")
# plt.savefig('mle.png', format="png")
# plt.savefig('av.png', format="png")

