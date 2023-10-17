import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import N
# from adjustText import adjust_text
from pymatgen.analysis.adsorption import *
from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.core.surface import Slab, SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
# from pymatgen.core.structure import Structure
# from pymatgen.ext.matproj import MPRester
from matplotlib import pyplot as plt
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.sets import *
from pymatgen.io.cif import CifWriter


class build_surface_with_absorbate:
    def __init__(self,
                 structures_file_path=None,
                 molecules_npy_path='molecules.npy'):
        self.structures_file_path = structures_file_path
        self.molecules_npy = np.load(molecules_npy_path,
                                     allow_pickle=True).item()
        self.molecules = self.build_molecules()

    def build_molecules(self):
        molecules = {}
        for molecules_str in list(self.molecules_npy.keys()):
            molecules[molecules_str] = Molecule(
                self.molecules_npy[molecules_str][0],
                self.molecules_npy[molecules_str][1])
        return molecules

    def define_molecule(self,
                        molecules_str,
                        molecules_coor,
                        molecules_num=None):
        if molecules_num:
            self.molecules_npy[molecules_str] = Molecule(
                molecules_num, molecules_coor)
        else:
            self.molecules_npy[molecules_str] = Molecule(
                molecules_str, molecules_coor)
        np.save('molecules_user_defined.npy', self.molecules_npy)
        self.molecules = self.build_molecules()

    def get_slab(self,
                 struct_type,
                 mp_id,
                 miller_index,
                 direct_struct=None,
                 struct_path=None,
                 min_slab_size=10.0,
                 min_vacuum_size=15.0,
                 center_slab=True,
                 lll_reduce=True,
                 slab_shift=None,
                 struct_is_supercell=False,
                 supercell=[[2, 0, 0], [0, 2, 0], [0, 0, 1]]):
        if struct_type == 'direct':
            slabs = direct_struct
            slabs = self.center_struct(slabs)
            struct = slabs
        elif struct_type == 'slab':
            slabs = Structure.from_file(struct_path)
            slabs = self.center_struct(slabs)
            struct = slabs
        # else:
        #     if mp_id:
        #         # mpr = MPRester()
        #         # struct = mpr.get_structure_by_material_id(mp_id)
        #     elif struct_type == 'bulk':
        #         struct = Structure.from_file(struct_path)
        #     struct = SpacegroupAnalyzer(
        #         struct).get_conventional_standard_structure()
        #     slab = SlabGenerator(struct,
        #                          min_slab_size=min_slab_size,
        #                          min_vacuum_size=min_vacuum_size,
        #                          center_slab=center_slab,
        #                          lll_reduce=lll_reduce,
        #                          miller_index=miller_index)
        #     if slab_shift:
        #         slabs = slab.get_slab(shift=slab_shift)
        #     else:
        #         slabs = slab.get_slabs()[0]
        # if struct_type != 'slab' or struct_is_supercell is not True:
        #     slabs.make_supercell(supercell)
        return slabs, struct

    def create_slabs_with_absorbates(self,
                                     absorbates,
                                     miller_index,
                                     struct_path=None,
                                     direct_struct=None,
                                     min_slab_size=10.0,
                                     min_vacuum_size=15.0,
                                     center_slab=True,
                                     lll_reduce=True,
                                     slab_shift=None,
                                     struct_type=None,
                                     mp_id=None,
                                     absorption_sites=None,
                                     absorption_distance=2.0,
                                     struct_is_supercell=False,
                                     supercell=[[2, 0, 0], [0, 2, 0],
                                                [0, 0, 1]],
                                     selective_dynamics=None,
                                     height=None,
                                     site_type=['ontop', 'bridge', 'hollow'],
                                     adsorption_structures_index=[0],
                                     adsorption_structures_num=None,
                                     generate_all_structures=False,
                                     return_adsorption_site=False,
                                     show_absorption_sites=False,
                                     vasp_input_set=MPRelaxSet,
                                     write_input=True):

        slabs, struct = self.get_slab(struct_type, mp_id, miller_index,
                                      direct_struct, struct_path,
                                      min_slab_size, min_vacuum_size,
                                      center_slab, lll_reduce, slab_shift,
                                      struct_is_supercell, supercell)
        if selective_dynamics:
            asf_slab = AdsorbateSiteFinder(
                slabs, selective_dynamics=selective_dynamics, height=height)
        else:
            asf_slab = AdsorbateSiteFinder(slabs)
        if absorption_sites:
            struc_slab = AdsorbateSiteFinder(slabs)
            for absorbate in absorbates:
                locals()['ads_structs_{}'.format(absorbate)] = []
                for absorption_site in absorption_sites:
                    print(self.molecules[absorbate])
                    locals()['ads_structs_{}'.format(absorbate)].append(
                        struc_slab.add_adsorbate(self.molecules[absorbate],
                                                 absorption_site))
        else:
            absorption_sites = asf_slab.find_adsorption_sites(
                distance=2.0, positions=site_type)
            for absorbate in absorbates:
                locals()['ads_structs_{}'.format(
                    absorbate)] = asf_slab.generate_adsorption_structures(
                        self.molecules[absorbate],
                        find_args={
                            'distance': absorption_distance,
                            'positions': site_type,
                            'symm_reduce': 0.3,
                            'near_reduce': 0.01
                        })
            if generate_all_structures:
                adsorption_structures_index = [
                    i for i in range(
                        len(locals()['ads_structs_{}'.format(absorbate)]))
                ]
            if adsorption_structures_num:
                adsorption_structures_index = np.linspace(
                    0,
                    len(locals()['ads_structs_{}'.format(absorbate)]) - 1,
                    adsorption_structures_num)
                adsorption_structures_index = [
                    int(i) for i in adsorption_structures_index
                ]
            if len(adsorption_structures_index) > len(
                    locals()['ads_structs_{}'.format(absorbate)]):
                print(
                    'The number of adsorption_structures required is greater than\
                        the number of possible structures! ')
                adsorption_structures_index = adsorption_structures_index[:len(
                    locals()['ads_structs_{}'.format(absorbate)])]
            absorption_sites = [
                absorption_sites['all'][index]
                for index in adsorption_structures_index
            ]
        if show_absorption_sites:
            self.show_absorption_sites(
                struct=struct,
                asf=asf_slab,
                miller_index=miller_index,
                absorption_sites=absorption_sites,
                adsorption_structures_index=adsorption_structures_index)
        if write_input:
            relax_set = vasp_input_set(structure=slabs)
            relax_set.write_input(output_dir='slab', potcar_spec=False)
        all_structs = {}
        for index in adsorption_structures_index:
            structs = {}
            for absorbate in absorbates:
                structs['slab+' + absorbate] = locals()[
                    'ads_structs_{}'.format(absorbate)][index]
            all_structs[index] = structs
            if write_input:
                for key, struct in structs.items():
                    relax_set = vasp_input_set(structure=struct)
                    if self.structures_file_path:
                        output_dir = self.structures_file_path + "./{}_{}".format(
                            key, index)
                    else:
                        output_dir = "./{}_{}".format(key, index)
                    relax_set.write_input(output_dir=output_dir,
                                          potcar_spec=False)
        if return_adsorption_site:
            return all_structs, absorption_sites
        else:
            return all_structs

    @staticmethod
    def center_struct(struct):
        i = 0
        while np.min(abs(struct.frac_coords[:, 2])) <= 0.1 or np.max(
                abs(struct.frac_coords[:, 2])) >= 0.8:
            if i > 20:
                print(i)
                poscar = Poscar(structure=struct)
                poscar.write_file('POSCAR_bug')
                break
            for site in struct:
                site.coords -= np.array([0, 0, 0.1 * struct.lattice.c])
                site.coords[2] %= struct.lattice.c
                # if site.frac_coords[2] < 0:
                #     deci, peri = math.modf(site.frac_coords[2])
                #     site.frac_coords[2] = deci
                #     site.frac_coords[2] = 1 + site.frac_coords[2]
                #     site.coords[2] = (peri +1) * struct.lattice.c + site.coords[2]
            i += 1
        cifs = CifWriter(struct)
        cifs.write_file('temp.cif')
        struct = Structure.from_file('temp.cif')
        return struct

    def show_absorption_sites(self,
                              miller_index,
                              struct=None,
                              asf=None,
                              find_absorption_sites=False,
                              struct_path=None,
                              struct_type=None,
                              mp_id=None,
                              min_slab_size=None,
                              min_vacuum_size=None,
                              center_slab=None,
                              lll_reduce=None,
                              slab_shift=None,
                              struct_is_supercell=None,
                              supercell=None,
                              slab=None,
                              absorption_sites=None,
                              site_type=['ontop', 'bridge', 'hollow'],
                              adsorption_structures_index=None):
        if find_absorption_sites:
            slab, struct = self.get_slab(struct_path, struct_type, mp_id,
                                         miller_index, min_slab_size,
                                         min_vacuum_size, center_slab,
                                         lll_reduce, slab_shift,
                                         struct_is_supercell, supercell)
        if slab:
            asf = AdsorbateSiteFinder(slab)
        fig = plt.figure(figsize=(8, 8))
        axes = fig.add_subplot(1, 1, 1)
        # plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.sans-serif'] = ['Liberation Sans']
        if absorption_sites is None:
            absorption_sites = asf.find_adsorption_sites(
                positions=site_type)['all']
        plot_slab(asf.slab, axes, adsorption_sites=False)
        mi_string = "".join([str(i) for i in miller_index])
        sop = get_rot(asf.slab)
        ads_sites = [
            sop.operate(ads_site)[:2].tolist() for ads_site in absorption_sites
        ]
        axes.plot(*zip(*ads_sites),
                  color='k',
                  marker='x',
                  markersize=12,
                  mew=2,
                  linestyle='',
                  zorder=10000)
        texts = []
        for index, ads_site in zip(adsorption_structures_index, ads_sites):
            texts.append(
                axes.text(ads_site[0] + 0.1,
                          ads_site[1] + 0.1,
                          'Site{}'.format(index),
                          fontsize=15,
                          zorder=10000))
        adjust_text(texts)
        axes.set_title("{}({})".format(struct.composition.reduced_formula,
                                       mi_string),
                       fontsize=18)
        axes.set_xticks([])
        axes.set_yticks([])
        plt.savefig("{}({}).pdf".format(struct.composition.reduced_formula,
                                        mi_string),
                    format='pdf',
                    dpi=1200,
                    bbox_inches='tight')
