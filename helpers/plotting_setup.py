from IPython.display import display
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from cycler import cycler
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
import matplotlib.patheffects as mpatheffects   
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.axes import Axes
import scienceplots
plt.style.use(['science'])
rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'size': 10})
# rc('figure.constrained_layout', use=True)
rc('text', usetex=True)
rc('lines', linewidth=2)
rc('axes.formatter', use_mathtext=True)
plt.rcParams.update({'figure.dpi': '100'})
prop_cycle = plt.rcParams['axes.prop_cycle']
COLORS = prop_cycle.by_key()['color']
COLORS.append('tab:pink')
plt.rcParams['axes.prop_cycle'] =  cycler(color=COLORS)
comp_cycler = (cycler(color=COLORS[:4]) + cycler(lw=[2, 2, 2, 2]) + cycler(linestyle=['-', '--', '-.', ':']))