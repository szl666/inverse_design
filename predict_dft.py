import os
import ase
import json
from ase import io
from ase.io import cif
import numpy as np
from ase import Atoms, Atom
import pytorch_lightning
import torch
import pandas as pd
import collections
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab
from pymatgen.io.ase import AseAtomsAdaptor
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.models.gemnet_oc import gemnet_oc
from ocpmodels.models import dimenet_plus_plus
from torch_geometric.data import Batch
from ase.constraints import dict2constraint
from ase.calculators.singlepoint import SinglePointCalculator
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def get_dir(path):
    return [
        dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))
    ]


def mae(x, y):
    sumu = 0
    for i, j in zip(x, y):
        sumu += abs(i - j)
    return sumu / len(x)


def get_system_output(command):
    temp = os.popen(command)
    return float(temp.readlines()[0])


def get_energy(path, root):
    os.chdir(path)
    value = get_system_output(
        "grep 'TOTEN' OUTCAR | tail -1 | awk '{print $5}'")
    os.chdir(root)
    return value


def get_struct_ene(cal_path):
    struct_dict = {}
    ene_dict = {}
    root = os.getcwd()
    for slabs_path in get_dir(cal_path):
        if 'slab' not in slabs_path:
            for slab_path in get_dir(f'{cal_path}/{slabs_path}'):
                struct_name = f'{slabs_path}/{slab_path}'
                try:
                    ene_dict[struct_name] = get_energy(
                        f'{cal_path}/{slabs_path}/{slab_path}', root)
                    if slab_path != 'slab':
                        struct_dict[struct_name] = Structure.from_file(
                            f'{cal_path}/{slabs_path}/{slab_path}/CONTCAR')
                except:
                    pass
    slab_ene_dict = {}
    for key in ene_dict.keys():
        if key[-4:] == 'slab':
            slab_ene_dict[key] = ene_dict[key]
    for key in ene_dict.keys():
        if key.split('+')[0] in slab_ene_dict.keys():
            ene_dict[key] = ene_dict[key] - slab_ene_dict[key.split('+')
                                                          [0]] + 14.409911
    ene_dict = {
        key: value
        for key, value in ene_dict.items() if key[-4:] != 'slab'
    }
    return ene_dict, struct_dict


def generate_batch(struct_dict):
    atoms = [
        AseAtomsAdaptor.get_atoms(struct) for struct in struct_dict.values()
    ]
    atg = AtomsToGraphs()
    data = atg.convert_all(atoms)
    for index, dataobject in enumerate(data):
        dataobject.sid = index
    databatch = Batch.from_data_list(data)
    return databatch


def load_dimenet(model_path):
    DimeNet = dimenet_plus_plus.DimeNetPlusPlusWrap(hidden_channels=256,
                                                    out_emb_channels=192,
                                                    num_blocks=3,
                                                    cutoff=6.0,
                                                    num_radial=6,
                                                    num_spherical=7,
                                                    num_before_skip=1,
                                                    num_after_skip=2,
                                                    num_output_layers=3,
                                                    regress_forces=False,
                                                    use_pbc=True,
                                                    num_targets=1)
    model_CKPT = torch.load(model_path)
    state_dicts = collections.OrderedDict()
    for key, value in model_CKPT['state_dict'].items():
        state_dicts[key[14:]] = value
    DimeNet.load_state_dict(state_dicts)
    return DimeNet


def get_pre_result(ene_dict, predicts):
    real_ene = list(ene_dict.values())
    return r2_score(real_ene, predicts), mae(real_ene, predicts)


def draw_fitting(energy_real, energy_pre):
    energy_real = list(energy_real.values())
    plt.figure(figsize=(18, 18))
    plt.rcParams['font.size'] = 48
    plt.rcParams['font.sans-serif'] = ['Liberation Sans']
    plt.scatter(energy_real,
                energy_pre,
                marker='o',
                linewidth=2.0,
                alpha=0.5,
                s=200)
    plt.plot([-4, 1.5], [-4, 1.5], linewidth=3.0, linestyle='--', c='grey')
    bwith = 2
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.tick_params(which='major', width=3, length=10)
    plt.ylabel('Predicted absoption energies(eV)')
    plt.xlabel("Actual absoption energies(eV)")
    plt.xlim((-4, 1.5))
    plt.ylim((-4, 1.5))
    plt.savefig('test_dft.pdf', bbox_inches='tight', dpi=1200)


print('Loading model')
model_path = 'checkpoint.pt'
DimeNet = load_dimenet('checkpoint.pt')

print('Analyze calculation')
cal_path = '/home/zlsong/slab_opt_CO_absorption/'
ene_dict, struct_dict = get_struct_ene(cal_path)
# ene_dict = np.load('ene_dict.npy', allow_pickle=True).item()
# struct_dict = np.load('struct_dict.npy', allow_pickle=True).item()
databatch = generate_batch(struct_dict)

np.save('struct_dict_new.npy', struct_dict)
np.save('ene_dict_new.npy', ene_dict)

print('Perform prediction')
with torch.no_grad():
    predicts = DimeNet(databatch) * 0.7121978259612424 - 0.7213243512666709
predicts = predicts.detach().numpy().reshape(len(predicts))
np.save('predicts_new.npy', predicts)

R2, MAE = get_pre_result(ene_dict, predicts)
print(R2)
print(MAE)
draw_fitting(ene_dict, predicts)
