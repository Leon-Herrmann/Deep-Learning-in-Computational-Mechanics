import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.font_manager
matplotlib.rcParams["figure.dpi"] = 300
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman'],
             'size' : 10})
rc('text', usetex=True)

import Utilities
import torch

x = torch.linspace(0,1,200)
c = Utilities.bump(x,0.5,0.2,0.25)


# Set up plot
fig, ax = plt.subplots(figsize=(4,3))
ax.set_xlabel("$x$")
ax.set_ylabel("$v_{p}(x)$")
ax.plot(x.detach().numpy(),c,color='k',linewidth=2)
plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)




plt.show()
fig.tight_layout()
plt.savefig("Bump.eps")