import torch
import collections
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from pymatgen.core import Lattice, Structure, Molecule
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.models import dimenet_plus_plus
from catalyst import build_surface_with_absorbate
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar

warnings.filterwarnings('ignore')
build_surface_with_absorbate = build_surface_with_absorbate()
atg = AtomsToGraphs()


def load_dimenet(model_path, hcs, oec):
    DimeNet = dimenet_plus_plus.DimeNetPlusPlusWrap(hidden_channels=hcs,
                                                    out_emb_channels=oec,
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
        state_dicts[key[7:]] = value
    DimeNet.load_state_dict(state_dicts)
    return DimeNet


def predict_ene(structure, DimeNet, mean, std):
    atoms = [AseAtomsAdaptor.get_atoms(structure)]
    data = atg.convert_all(atoms, disable_tqdm=True)
    for index, dataobject in enumerate(data):
        dataobject.sid = index
    databatch = Batch.from_data_list(data)
    with torch.no_grad():
        pre_ene_ori = DimeNet(databatch)[0]
    pre_ene = pre_ene_ori * std + mean
    return pre_ene_ori, pre_ene


@torch.no_grad()
def predict_ene_batch(structure, DimeNet, mean, std):
    atoms = [AseAtomsAdaptor.get_atoms(struct) for struct in structure]
    data = atg.convert_all(atoms, disable_tqdm=True)
    for index, dataobject in enumerate(data):
        dataobject.sid = index
    databatch = Batch.from_data_list(data)
    with torch.no_grad():
        pre_ene_ori = DimeNet(databatch)
    pre_ene = pre_ene_ori * std + mean
    return pre_ene_ori, pre_ene


def get_structures(frac_coords, atom_types, num_atoms, lengths, angles):
    location = 0
    structures = {}
    for index, num_atom in enumerate(num_atoms):
        num_atom = int(num_atom)
        atom_type = atom_types.detach().cpu().numpy()[location:location +
                                                      num_atom]
        frac_coord = frac_coords.detach().cpu().numpy()[location:location +
                                                        num_atom]
        length = lengths.detach().cpu().numpy()[index]
        angle = angles.detach().cpu().numpy()[index]
        lattice = Lattice.from_parameters(length[0], length[1], length[2],
                                          angle[0], angle[1], angle[2])
        structure = Structure(lattice=lattice,
                              species=atom_type,
                              coords=frac_coord,
                              coords_are_cartesian=False)
        try:
            structure = structure.get_reduced_structure()
        except:
            print(structure)
        structures[index] = structure
        location += num_atom
    return structures
