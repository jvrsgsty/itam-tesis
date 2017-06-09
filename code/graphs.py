#!/bin/python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_context('talk')
sns.set(font_scale = 1.3)
plt.rc('font', family='serif')
font_size = 20

data = pd.read_csv('./graph_data/ceng_cols_rens.csv', header=0, index_col=0)
data.plot()
plt.ylabel(r'Tiempo de ejecución [horas]', fontsize=font_size)
plt.xlabel(r'Número de variables categóricas', fontsize=font_size)
plt.tight_layout()
plt.savefig('ceng_cols_rens.eps', format='eps', dpi=1000)
plt.show()

data = pd.read_csv('./graph_data/ceng_cols_rens_1.csv', header=0, index_col=0)
data.plot()
plt.ylabel(r'Tiempo de ejecución [horas]', fontsize=font_size)
plt.xlabel(r'Número de variables categóricas', fontsize=font_size)
plt.tight_layout()
plt.savefig('ceng_cols_rens_1.eps', format='eps', dpi=1000)
plt.show()

data = pd.read_csv('./graph_data/ceng_cols_rens_2.csv', header=0, index_col=0)
data.plot()
plt.ylabel(r'Tiempo de ejecución [horas]', fontsize=font_size)
plt.xlabel(r'Número de variables categóricas', fontsize=font_size)
plt.tight_layout()
plt.savefig('ceng_cols_rens_2.eps', format='eps', dpi=1000)
plt.show()

data = pd.read_csv('./graph_data/ceng_epochs.csv', header=0, index_col=0)
data.plot()
plt.ylabel(r'Función adecuación', fontsize=font_size)
plt.xlabel(r'Número de épocas', fontsize=font_size)
plt.tight_layout()
plt.savefig('ceng_epochs.eps', format='eps', dpi=1000)
plt.show()

data = pd.read_csv('./graph_data/ceng_generations.csv', header=0, index_col=0)
data.plot()
plt.ylabel(r'Función adecuación escalada', fontsize=font_size)
plt.xlabel(r'Número de generaciones', fontsize=font_size)
plt.tight_layout()
plt.savefig('ceng_generations.eps', format='eps', dpi=1000)
plt.show()

data = pd.read_csv('./graph_data/ceng_instancias.csv', header=0, index_col=0)
ax = data.transpose().plot(secondary_y='Tiempo')
#ax2.set_yscale('log')
ax.set_ylabel(r'Función adecuación', fontsize=font_size)
ax.right_ax.set_ylabel(r'Tiempo de ejecución [horas]', fontsize=font_size)
ax.set_xlabel(r'Número de instancias categóricas', fontsize=font_size)
plt.tight_layout()
plt.savefig('ceng_instancias.eps', format='eps', dpi=1000)
plt.show()

data = pd.read_csv('./graph_data/parallel_ceng.csv', header=0, index_col=0)
data.plot.bar(rot=0)
plt.ylabel(r'Tiempo de ejecución [horas]', fontsize=font_size)
plt.tight_layout()
plt.savefig('parallel_ceng.eps', format='eps', dpi=1000)
plt.show()
