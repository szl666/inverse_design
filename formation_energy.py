import os
import re
import ase
import json
from ase import io
from ase.io import cif
import numpy as np
from ase import Atoms, Atom
import torch
import pandas as pd
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.ticker as ticker
from pymatgen.ext.crystalsai import CrystalAIRester


def predict_formation_en(struct):
    megnets = CrystalAIRester()
    return megnets.predict_structure('formation_energy', struct)


slab_info = np.load('slab_info_test.npy', allow_pickle=True).item()

location = 0
location_end = -1
while True:
    try:
        for key, value in slab_info.items():
            location = key
            if key <= 1:
                print(key)
                for info in value.values():
                    info['formation_ene'] = predict_formation_en(
                        info['struct'])
                np.save('slab_info_test.npy', slab_info)
    except:
        print(f'end------------------------------{location}')
        location_end = location
    if location == 99:
        break
