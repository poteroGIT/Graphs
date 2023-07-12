import matplotlib.pyplot as plt
import numpy as np

# Aplicar el tema Solarized_light2
plt.style.use('Solarize_Light2')
import matplotlib as mpl
params = {'xtick.labelsize': 20, 'ytick.labelsize': 20}
mpl.rcParams.update(params)

font2 = {'family': 'sans-serif', 'color': 'saddlebrown', 'size': 26}

# Datos de ejemplo
clasificadores = ['Logistic Regression', 'SV Machines', 'Decission Trees', 'Random Forest', 'Naive Bayes', 'K-Nearest Neighbor']
accuracy = [0.81, 0.8, 0.86, 0.94, 0.77, 0.9]
precision = [0.81, 0.81, 0.85, 0.94, 0.67, 0.95]
recall = [0.84, 0.83, 0.89, 0.955, 0.92, 0.88]

accuracy = [0.98, 0.982, 0.96, 0.986, 0.97, 0.984]
precision = [0., 0., 0., 0., 0.065, 0.]
recall = [0., 0., 0., 0., 0.06, 0.]

# Configuración del gráfico
indice = np.arange(len(clasificadores))
ancho_barras = 0.2

# Crear el gráfico de barras
fig, ax = plt.subplots()
bar1 = ax.bar(indice, accuracy, ancho_barras, label='Accuracy', color='chocolate')
bar2 = ax.bar(indice + ancho_barras, precision, ancho_barras, label='Precision', color='sienna')
bar3 = ax.bar(indice + 2 * ancho_barras, recall, ancho_barras, label='Recall', color='peru')

# Configurar el eje x
#ax.set_xlabel('Clasificadores')
#ax.set_ylabel('Puntuación')
ax.set_xticks(indice + ancho_barras)
ax.set_xticklabels(clasificadores, rotation=15, fontdict=font2)
ax.set_ylim(0, 1)

# Mover la leyenda y ajustar el espacio
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=3, fontsize=23)

# Ajustar los márgenes del gráfico
plt.subplots_adjust(top=0.85)

# Mostrar el gráfico de barras
plt.show()
