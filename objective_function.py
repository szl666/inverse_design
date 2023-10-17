import numpy as np
from predict_energy import *
import warnings
import torch
import yaml
from mendeleev import element
from scipy.stats.mstats import gmean
from pymatgen.io.vasp.inputs import Poscar
from ocpmodels.models.dimenet_plus_plus import DimeNetPlusPlusWrap
from ocpmodels.models.painn.painn import PaiNN

warnings.filterwarnings('ignore')


class objective_func:
    def __init__(self, property_list, poscar_path):
        self.property_list = property_list
        self.model_info_list = self.get_model_info()
        self.model_list = self.get_mode_list()
        self.poscar_path = poscar_path

    def make_predictions(self, structs, model, mean, std):
        return predict_ene_batch(structs, model, mean, std)

    def get_model_info(self):
        with open('model_info.yaml', 'r') as f:
            model_info = yaml.safe_load(f)
        model_info_list = [model_info[prop] for prop in self.property_list]
        return model_info_list

    def get_mode_list(self):
        model_list = []
        for prop, model_info in zip(self.property_list, self.model_info_list):
            prop = prop.replace(' ', '_')
            model_list.append(
                load_model(DimeNetPlusPlusWrap, 'models/dpp_Tc_supercon.yml',
                           f'models/model_{prop}.pt'))
        return model_list

    @torch.no_grad()
    def get_objective_func(self,
                           structs,
                           weights=None,
                           write_poscar=True,
                           index=None):
        property_values_norm_list = []
        property_values_list = []
        for model, model_info in zip(self.model_list, self.model_info_list):
            property_values_norm, property_values = self.make_predictions(
                structs, model, model_info[2], model_info[3])
            property_values_norm_list.append(
                property_values_norm.flatten().numpy())
            property_values_list.append(property_values.flatten().numpy())
        if weights:
            property_values_norm_list = [
                prop_norm * weight for prop_norm, weight in zip(
                    property_values_norm_list, weights)
            ]
            property_values_list = [
                prop_norm * weight for prop_norm, weight in zip(
                    property_values_norm_list, weights)
            ]
        if write_poscar:
            self.write_poscar(index, structs, property_values_list)
        return self.limit_value(property_values_norm_list), self.limit_value(
            property_values_list)

    def limit_value(self, list_origin):
        list_output = np.array(list_origin)
        list_output[list_output > 1000] = 1000
        list_output[list_output < 0] = 0
        return list_output

    def write_poscar(self, index, structs, property_values_list):
        property_values_list = np.array(property_values_list)
        for num, struct in enumerate(structs):
            poscar = Poscar(structure=struct)
            prop_list = [
                str(round(float(i), 3)).replace('.', '_')
                for i in property_values_list[:, num]
            ]
            poscar_prop = ''.join([f'({i})' for i in prop_list])
            poscar.write_file(
                f'{self.poscar_path}/POSCAR_step{index}{poscar_prop})')
