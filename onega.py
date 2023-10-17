import os
import time
import yaml
import argparse
import torch
import collections
import warnings

import numpy as np
from bird_swarm_opt import Bird_swarm_opt
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from torch.nn import functional as F
from types import SimpleNamespace
from torch_geometric.data import Batch
from mendeleev import element
from eval_utils import load_model
from pymatgen.core import Lattice, Structure, Molecule
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.models import dimenet_plus_plus
from catalyst import build_surface_with_absorbate
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar
from pyxtal.lattice import Lattice
from pyxtal import pyxtal

warnings.filterwarnings('ignore')


def read_yaml(yaml_name):
    with open(yaml_name, 'r') as f:
        data = yaml.safe_load(f)
    return data


def optimization(settings, model, ld_kwargs, data_loader):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:settings['num_starting_points']].detach().clone()
    else:
        z = torch.randn(settings['num_starting_points'],
                        model.hparams.hidden_dim,
                        device=model.device)
    model.freeze()
    bsa = Bird_swarm_opt(opt_matrix=z,
                         cdvae_model=model,
                         ld_kwargs=ld_kwargs,
                         property_list=settings['property'],
                         property_list_range=settings['property_range'],
                         poscar_path=settings['struct_path'],
                         interval=settings['interval'],
                         min_value=settings['min_z_value'],
                         max_value=settings['max_z_value'],
                         improve_symmetry=settings['improve_symmetry'],
                         exponent=settings['exponent'])
    if settings['conti']:
        conti_iter = int(torch.load("index.pt"))
        fMin, bestIndex, bestX, b2 = bsa.search(M=settings['num_steps'],
                                                conti=settings['conti'],
                                                conti_iteration=conti_iter)
    else:
        fMin, bestIndex, bestX, b2 = bsa.search(M=settings['num_steps'])
    all_crystals = bsa.all_crystals
    return {
        k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0)
        for k in
        ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']
    }


def run_onega(settings):
    settings = read_yaml('input.yaml')
    if settings['model_path']:
        model_path = Path(settings['model_path'])
    else:
        model_path = Path('/fs0/home/liqiang/onega/cdvae_model')
    model, test_loader, cfg = load_model(
        model_path, load_data=(settings['start_from'] == 'data'))
    ld_kwargs = SimpleNamespace(n_step_each=settings['n_step_each'],
                                step_lr=float(settings['step_lr']),
                                min_sigma=float(settings['min_sigma']),
                                save_traj=settings['save_traj'],
                                disable_bar=settings['disable_bar'])
    if torch.cuda.is_available():
        model.to('cuda')
    print('Performing bird swarm optimization')
    if settings['start_from'] == 'data':
        loader = test_loader
    else:
        loader = None
    optimized_crystals = optimization(settings, model, ld_kwargs, loader)

    if settings['label'] == '':
        gen_out_name = 'eval_opt.pt'
    else:
        label = settings['label']
        gen_out_name = f'eval_opt_{label}.pt'
    torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    settings = read_yaml('input.yaml')
    run_onega(settings)
