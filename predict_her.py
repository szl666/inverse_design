import os
import re
import torch
import collections
import warnings
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from pymatgen.core import Lattice, Structure, Molecule
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.models import dimenet_plus_plus
from ocpmodels.models.painn import painn
from catalyst import build_surface_with_absorbate
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Poscar

warnings.filterwarnings('ignore')
build_surface_with_absorbate = build_surface_with_absorbate()
atg = AtomsToGraphs()


class predict_H:
    def __init__(self, model_path):
        self.model_path = model_path
        self.pre_model = self.load_dimenet(self.model_path)

    def load_dimenet(self, model_path):
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
                                                        num_targets=1,
                                                        otf_graph=False)
        model_CKPT = torch.load(model_path)
        state_dicts = collections.OrderedDict()
        for key, value in model_CKPT['state_dict'].items():
            state_dicts[key[14:]] = value
            # state_dicts[key] = value
        DimeNet.load_state_dict(state_dicts)
        return DimeNet

    # def load_painn(self, model_path):
    #     PaiNN = painn.PaiNN(
    #         hidden_channels=1024,
    #         num_layers=6,
    #         num_rbf=128,
    #         cutoff=12.0,
    #         max_neighbors=50,
    #         scale_file='/home/zlsong/cdvae-main/painn_nb6_scaling_factors.pt',
    #         regress_forces=False,
    #         use_pbc=True)
    #     model_CKPT = torch.load(model_path)
    #     state_dicts = collections.OrderedDict()
    #     for key, value in model_CKPT['state_dict'].items():
    #         state_dicts[key[7:]] = value
    #     PaiNN.load_state_dict(state_dicts)
    #     return PaiNN

    def predict_ene(self,
                    structure,
                    write_poscar=False,
                    step=None,
                    file_path=None):
        absorbates = ['H']
        try:
            slabs = build_surface_with_absorbate.create_slabs_with_absorbates(
                absorbates=absorbates,
                miller_index=[0, 0, 1],
                struct_type='direct',
                direct_struct=structure,
                write_input=False,
                struct_is_supercell=True,
                show_absorption_sites=False,
                adsorption_structures_num=6)
            atoms = [
                AseAtomsAdaptor.get_atoms(i['slab+H']) for i in slabs.values()
            ]
            data = atg.convert_all(atoms, disable_tqdm=True)
            for index, dataobject in enumerate(data):
                dataobject.sid = index
            databatch = Batch.from_data_list(data)
            pre_ene = self.pre_model(
                databatch) * 0.5105034148103647 - 0.0406671312847654
            min_pre_ene = min(pre_ene)
            ene_str = str(round(float(min_pre_ene), 4)).replace('.', '_')
            if write_poscar:
                poscar = Poscar(structure=structure)
                poscar.write_file(f'{file_path}/POSCAR_step{step}({ene_str})')
            return min_pre_ene
        except:
            poscar = Poscar(structure=structure)
            poscar.write_file('POSCAR_bugs')
            return 10.0

    def get_slabs_info(self, poscars_path):
        torch.no_grad()
        slab_info = collections.defaultdict(dict)
        poscars = os.listdir(poscars_path)
        dict_index = {}
        for poscar in poscars:
            ene_row = re.findall(r"(\-?\d+\_\d+)", poscar)[0]
            energy_co = eval(ene_row.replace('_', '.'))
            step = int(re.findall(r"step\-?\d+", poscar)[0].strip('step'))
            dict_index[step] = dict_index.get(step, 0) + 1
            struct = Structure.from_file(f'{poscars_path}/{poscar}')
            with torch.no_grad():
                energy_h = self.predict_ene(struct)
            formula = struct.formula.replace(' ', '')
            single_info = {
                'energy_co': energy_co,
                'energy_h': energy_h,
                'formula': formula,
                'struct': struct
            }
            slab_info[step][dict_index[step]] = single_info
        return slab_info


# H_predict_model = predict_H('checkpoint_h.pt')
H_predict_model = predict_H('best_checkpoint_h.pt')
slab_info = H_predict_model.get_slabs_info('opt_poscars')
slab_info = dict(sorted(slab_info.items(), key=lambda d: d[0]))
np.save('slab_info11.npy', slab_info)
