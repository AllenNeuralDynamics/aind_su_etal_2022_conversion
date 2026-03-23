# %%
import random
from pathlib import Path

import numpy as np
from myterial import orange
from rich import print

from brainrender import Scene, settings
from brainrender.actors import Points, Cylinder, Point

import pandas as pd

settings.SHOW_AXES = False
settings.WHOLE_SCREEN = False


# %%

TitleLabel = r"CCF Coordinates for PL Photometry Fiber Implantation"

#import vedo
#vedo.settings.default_backend= 'vtk'


scene = Scene(title=f"{TitleLabel}")

# Get a numpy array with (fake) coordinates of some labelled cells
# ACB = scene.add_brain_region("ACB", alpha=0.20, color=(0.2,0.6,0.2))
# VTA = scene.add_brain_region("VTA", alpha=0.20, color=(0.6,0.2,0.2))
PL = scene.add_brain_region("PL", alpha=0.20, color=(0.2,0.2,0.6))

# load ccf coordinates from file
ccf_file = Path(r"C:\Users\zhixi\OneDrive - Allen Institute\LCpaper\probe_tracking\photometry\PL_ccf_coordinates_pir.csv")
ccf_locations = pd.read_csv(ccf_file)
PL_coordinates = ccf_locations[['x', 'y', 'z']].values * 25  # convert to microns

bregma = np.array([216, 18, 228]).reshape(1, 3) * 25  # in microns

origin = np.array([0, 0, 0]).reshape(1, 3)

PL_coordinates_flip = PL_coordinates.copy()
PL_coordinates_flip[:, 2] = 2*bregma[:, 2] - PL_coordinates[:, 2]  # flip z-axis
#VTA = scene.add_brain_region("VTA", alpha=0.50)
#VTA_coordinates = get_n_random_points_in_region(VTA, 100)
#SNc = scene.add_brain_region("SNc", alpha=0.50)
#SNc_coordinates = get_n_random_points_in_region(SNc, 100)
#CP = scene.add_brain_region("CP", alpha=0.15)
#PL = scene.add_brain_region("PL", alpha=0.15)
#BLA = scene.add_brain_region("BLA", alpha=0.15)
#CeA = scene.add_brain_region("CEA", alpha=0.15)
#VISp = scene.add_brain_region("VISp", alpha=0.50) 
#LH = scene.add_brain_region("LH", alpha=0.15) 
#LGv = scene.add_brain_region("LGv", alpha=0.15)
#PVT = scene.add_brain_region("PVT", alpha=0.15)
#VMH = scene.add_brain_region("VMH", alpha=0.15)
#LHA = scene.add_brain_region("LHA", alpha=0.15)
#CA1sr = scene.add_brain_region("CA1sr", alpha=0.15)


# Add to scene
# scene.add(Points(ACB_coordinates, name="CELLS", colors="steelblue",radius=80))
# scene.add(Points(VTA_coordinates, name="CELLS", colors="red",radius=80))
#scene.add(Points(SNc_coordinates, name="CELLS", colors="orange"))
scene.add(Points(PL_coordinates_flip, name="CELLS", colors="purple",radius=80))
# scene.add(Points(origin, colors="black", radius=50))
# scene.add(Points(bregma, colors="green", radius=50))

# create and add a cylinder actor ()
#actor = Cylinder(pos=[4035, 5870, 6934], root=scene.root, alpha=0.5, radius=100)
#scene.add(Points(np.array((4035, 5870, 6934)), name="CELLS", colors="red",radius=80))
#actor2 = Cylinder(pos=[8225, 4873, 6174], root=scene.root, alpha=0.5, radius=100)
#scene.add(actor)
#scene.add(actor2)

# render
scene.content
scene.render()
 


