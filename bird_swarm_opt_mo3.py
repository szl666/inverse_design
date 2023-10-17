from numpy import *
from predict_energy1 import *
import warnings
import torch
from mendeleev import element
from scipy.stats.mstats import gmean
from pyxtal.lattice import Lattice as pyxtal_lattice
from pyxtal import pyxtal
from pymatgen.core import Lattice, Structure
from torch.nn import functional as F
from cdvae.common.data_utils import (EPSILON, cart_to_frac_coords, mard,
                                     lengths_angles_to_volume,
                                     frac_to_cart_coords, min_distance_sqr_pbc)
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

warnings.filterwarnings('ignore')


class Bird_swarm_opt:
    def __init__(self,
                 opt_matrix,
                 cdvae_model,
                 ld_kwargs,
                 ck_path_ehull,
                 ck_path_gap,
                 ck_path_pbx,
                 min_value=-8,
                 max_value=8,
                 save_traj=False,
                 traj_path='',
                 interval=100):
        self.opt_matrix = opt_matrix
        self.cdvae_model = cdvae_model
        self.ld_kwargs = ld_kwargs
        self.save_traj = save_traj
        self.traj_path = traj_path
        self.pBounds = {}
        self.kwargs = {}
        self.step = 0
        self.min_value = min_value
        self.max_value = max_value
        self.dim, self.lb, self.ub = self.get_lb_ub_dim()
        self.ehull_model = load_dimenet(ck_path_ehull, 512, 256)
        self.gap_model = load_dimenet(ck_path_gap, 512, 256)
        self.pbx_model = load_dimenet(ck_path_pbx, 256, 192)
        self.interval = interval
        self.all_crystals = []

    def get_lb_ub_dim(self):
        dim = self.opt_matrix.size(1)
        lb = [self.min_value] * dim
        ub = [self.max_value] * dim
        lb = expand_dims(lb, axis=0)
        ub = expand_dims(ub, axis=0)
        return torch.tensor(dim), torch.tensor(lb), torch.tensor(ub)

    @staticmethod
    def cal_mulliken(ele):
        elem = element(ele)
        ea = elem.electron_affinity
        ea = ea if ea else 0
        ip = elem.ionenergies.get(1, None)
        ip = ip if ip else 0
        return (ea + ip) / 2

    def cal_objective_function(self, structure, band_gap, ehull, ehull_S, Gpbx,
                               Gpbx_S):
        species_en_mulliken = [
            self.cal_mulliken(str(i)) for i in structure.species
        ]
        if band_gap <= 0:
            band_gap = 0
        if ehull <= 0:
            ehull_S = -1.7750069410012883 / 1.3928968647223894
        if Gpbx <= 0:
            Gpbx_S = -1.4296479801859339 / 1.7844662928101613
        if ehull >= 20:
            ehull_S = 2
        if Gpbx >= 20:
            Gpbx_S = 2
        Xs = gmean(species_en_mulliken)
        Ecb = Xs - 4.44 - 0.5 * band_gap
        Evb = Xs - 4.44 + 0.5 * band_gap
        # if 1.3 <= band_gap <= 3.3 and Ecb <= 0 and Evb >= 1.23 and len(
        #         set(structure.species)) <= 4:
        if 1.3 <= band_gap <= 3.3 and Ecb <= 0 and Evb >= 1.23:
            return ehull_S + Gpbx_S
        # elif len(species_en_mulliken) > 4:
        #     return 100
        else:
            return 100

    def get_struct_from_z(self, model, z, ld_kwargs):
        num_atoms, _, lengths, angles, composition_per_atom = model.decode_stats(
            z)
        # space_groups = model.predict_space_groups(z)
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        cur_atom_types = model.sample_composition(composition_per_atom,
                                                  num_atoms)
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)
        # cur_frac_coords_o, cur_lengths_o, cur_angles_o, cur_atom_types_o = self.get_sym_coords(
        #     cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
        # annealed langevin dynamics.
        for sigma in tqdm(model.sigmas,
                          total=model.sigmas.size(0),
                          disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / model.sigmas[-1])**2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(cur_frac_coords) * torch.sqrt(
                    step_size * 2)
                pred_cart_coord_diff, pred_atom_types = model.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths,
                    angles)
                cur_cart_coords = frac_to_cart_coords(cur_frac_coords, lengths,
                                                      angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(cur_cart_coords, lengths,
                                                      angles, num_atoms)
                cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1
        cur_frac_coords_o, cur_lengths_o, cur_angles_o, cur_atom_types_o = self.get_sym_coords(
            cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
        # output_dict = {
        #     'num_atoms': num_atoms,
        #     'lengths': lengths,
        #     'angles': angles,
        #     'frac_coords': cur_frac_coords,
        #     'atom_types': cur_atom_types,
        #     'is_traj': False
        # }
        output_dict = {
            'num_atoms': num_atoms,
            'lengths': cur_lengths_o,
            'angles': cur_angles_o,
            'frac_coords': cur_frac_coords_o,
            'atom_types': cur_atom_types_o,
            'is_traj': False
        }
        return output_dict

    def get_sym_coords(self, sym_frac_coords, atom_types, num_atoms, lengths,
                       angles):
        location = 0
        for index, num_atom in enumerate(num_atoms):
            print(index)
            num_atom = int(num_atom)
            frac_coord = sym_frac_coords.detach().cpu().numpy(
            )[location:location + num_atom]
            atom_type = atom_types.detach().cpu().numpy()[location:location +
                                                          num_atom]
            length = lengths.detach().cpu().numpy()[index]
            angle = angles.detach().cpu().numpy()[index]
            com_dict = {}
            for ato in atom_type:
                ato_symbol = element(int(ato)).symbol
                com_dict[ato_symbol] = com_dict.get(ato_symbol, 0) + 1
            elements = list(com_dict.keys())
            composition = list(com_dict.values())
            # cell = pyxtal_lattice.from_para(length[0], length[1], length[2],
            #                                 angle[0], angle[1], angle[2])
            lattice = Lattice.from_parameters(length[0], length[1], length[2],
                                              angle[0], angle[1], angle[2])
            struct_ori = Structure(lattice=lattice,
                                   species=atom_type,
                                   coords=frac_coord,
                                   coords_are_cartesian=False)
            analyzer = SpacegroupAnalyzer(struct_ori, symprec=0.1)
            try:
                spgg = analyzer.get_space_group_number()
            except:
                spgg = 1
            if spgg == 1:
                structs_sym = []
                for spg in range(2, 231):
                    try:
                        sp_crystal = pyxtal()
                        sp_crystal.from_random(3, spg, elements, composition)
                        # sp_crystal.from_random(3,
                        #                        spg,
                        #                        elements,
                        #                        composition,
                        #                        lattice=cell)
                        structs_sym.append(sp_crystal)
                    except:
                        continue
                if structs_sym:
                    struct = random.choice(structs_sym).to_pymatgen()
                    sym_frac_coords[location:location +
                                    num_atom] = torch.tensor(
                                        struct.frac_coords)
                    lengths[index] = torch.tensor(struct.lattice.lengths)
                    angles[index] = torch.tensor(struct.lattice.angles)
                atom_types[location:location +
                           num_atom] = torch.tensor(atom_type)
        return sym_frac_coords, lengths, angles, atom_types

    # def get_sym_coords(self,
    #                    sym_frac_coords,
    #                    space_groups,
    #                    atom_types,
    #                    num_atoms,
    #                    lengths,
    #                    angles,
    #                    z_device,
    #                    afterLD=False,
    #                    fail_index=None,
    #                    cur_atom_types=None,
    #                    cur_frac_coords=None,
    #                    lengthsLD=None,
    #                    anglesLD=None):
    #     location = 0
    #     if afterLD:
    #         for index, num_atom in enumerate(num_atoms):
    #             print(index)
    #             num_atom = int(num_atom)
    #             space_group = argmax(
    #                 space_groups.detach().cpu().numpy()[index]) + 1
    #             if index in fail_index:
    #                 atom_type = cur_atom_types.detach().cpu().numpy(
    #                 )[location:location + num_atom]
    #                 try:
    #                     sp_crystal = pyxtal()
    #                     com_dict = {}
    #                     for ato in atom_type:
    #                         ato_symbol = element(int(ato)).symbol
    #                         com_dict[ato_symbol] = com_dict.get(ato_symbol,
    #                                                             0) + 1
    #                     elements = list(com_dict.keys())
    #                     composition = list(com_dict.values())
    #                     sp_crystal.from_random(3, space_group, elements,
    #                                            composition)
    #                     struct = sp_crystal.to_pymatgen()
    #                     sym_frac_coords[location:location +
    #                                     num_atom] = torch.tensor(
    #                                         struct.frac_coords)
    #                     lengths[index] = torch.tensor(struct.lattice.lengths)
    #                     angles[index] = torch.tensor(struct.lattice.angles)
    #                     atom_types[location:location +
    #                                num_atom] = torch.tensor(atom_type)
    #                 except:
    #                     pass
    #             else:
    #                 sym_frac_coords[location:location +
    #                                 num_atom] = cur_frac_coords[
    #                                     location:location + num_atom]
    #                 lengths[index] = lengthsLD[index]
    #                 angles[index] = anglesLD[index]
    #                 atom_types[location:location +
    #                            num_atom] = cur_atom_types[location:location +
    #                                                       num_atom]
    #             location += num_atom
    #     else:
    #         fail_index = []
    #         for index, num_atom in enumerate(num_atoms):
    #             num_atom = int(num_atom)
    #             space_group = argmax(
    #                 space_groups.detach().cpu().numpy()[index]) + 1
    #             atom_type = atom_types.detach().cpu().numpy(
    #             )[location:location + num_atom]
    #             try:
    #                 sp_crystal = pyxtal()
    #                 com_dict = {}
    #                 for ato in atom_type:
    #                     ato_symbol = element(int(ato)).symbol
    #                     com_dict[ato_symbol] = com_dict.get(ato_symbol, 0) + 1
    #                 elements = list(com_dict.keys())
    #                 composition = list(com_dict.values())
    #                 sp_crystal.from_random(3, space_group, elements,
    #                                        composition)
    #                 struct = sp_crystal.to_pymatgen()
    #                 sym_frac_coords[location:location +
    #                                 num_atom] = torch.tensor(
    #                                     struct.frac_coords)
    #                 lengths[index] = torch.tensor(struct.lattice.lengths)
    #                 angles[index] = torch.tensor(struct.lattice.angles)
    #             except:
    #                 fail_index.append(index)
    #             location += num_atom
    #     return sym_frac_coords, lengths, angles, atom_types, fail_index

    @torch.no_grad()
    def get_objective_function(self, z, index, M):
        # print(z)
        # print('11111111111111111111111111111')
        samples = self.cdvae_model.langevin_dynamics(z, self.ld_kwargs)
        # samples = self.get_struct_from_z(self.cdvae_model, z, self.ld_kwargs)
        if index % self.interval == 0 or index == (M - 1):
            self.all_crystals.append(samples)
        # print('222222222222222222222')
        structures = get_structures(samples['frac_coords'],
                                    samples['atom_types'],
                                    samples['num_atoms'], samples['lengths'],
                                    samples['angles'])
        objectives = []
        structs = list(structures.values())
        # print('333333333333333333333')
        _, band_gaps = predict_ene_batch(structs,
                                         self.gap_model,
                                         mean=1.497423233546115,
                                         std=2.3198268552027064)
        ehull_Ss, ehulls = predict_ene_batch(structs,
                                             self.ehull_model,
                                             mean=1.7750069410012883,
                                             std=1.3928968647223894)
        Gpbx_Ss, Gpbxs = predict_ene_batch(structs,
                                           self.pbx_model,
                                           mean=1.4296479801859339,
                                           std=1.7844662928101613)
        # print('444444444444444444444')
        for struct, band_gap, ehull_S, ehull, Gpbx_S, Gpbx in zip(
                structs, band_gaps, ehull_Ss, ehulls, Gpbx_Ss, Gpbxs):
            # print('5555555555555555555555')
            objectives.append(
                self.cal_objective_function(struct, band_gap, ehull, ehull_S,
                                            Gpbx, Gpbx_S))
            gap_str = str(round(float(band_gap), 3)).replace('.', '_')
            ehull_str = str(round(float(ehull), 3)).replace('.', '_')
            Gpbx_str = str(round(float(Gpbx), 3)).replace('.', '_')
            poscar = Poscar(structure=struct)
            file_path = 'opt_poscars_mp30'
            poscar.write_file(
                f'{file_path}/POSCAR_step{index}({gap_str})({ehull_str})({Gpbx_str})'
            )
        objectives = torch.tensor(objectives)
        return torch.tensor(array(objectives).reshape(len(objectives),
                                                      1)).cuda()

    def randiTabu(self, minm, maxm, tabu, dim):
        value = ones((dim, 1)) * maxm * 2
        num = 1
        while (num <= dim):
            temp = random.randint(minm, maxm)
            findi = [
                index for (index, values) in enumerate(value) if values != temp
            ]
            if (len(findi) == dim and temp != tabu):
                value[0][num - 1] = temp
                num += 1
        return value

    def Bounds(self, s, lb, ub):
        temp = nan_to_num(s.cpu(), nan=0, posinf=10, neginf=-10)
        try:
            I = [
                index for (index, values) in enumerate(temp)
                if values < lb[0][index]
            ]
            for indexlb in I:
                temp[indexlb] = lb[0][indexlb]
            J = [
                index for (index, values) in enumerate(temp)
                if values > ub[0][index]
            ]
            for indexub in J:
                temp[indexub] = ub[0][indexub]
        except:
            print(temp)
        return torch.tensor(temp).cuda()

    def search(self,
               M=100,
               FQ=10,
               c1=0.15,
               c2=0.15,
               a1=1,
               a2=1,
               conti=False,
               conti_iteration=None):
        #############################################################################
        #     Initialization
        if conti:
            x = torch.load("x9.pt")
            pop = x.size(0)
            fit = torch.load("fit9.pt")
            pFit = torch.load("pFit9.pt")
            pX = torch.load("pX9.pt")
            fMin = torch.load("fMin9.pt")
            bestIndex = torch.load("bestIndex9.pt")
            bestX = torch.load("bestX9.pt")
            b2 = torch.load("b29.pt")
        else:
            x = self.opt_matrix
            x.cuda()
            pop = self.opt_matrix.size(0)
            fit = self.get_objective_function(x, -1, M)
            fit.cuda()
            pFit = fit.clone()
            pFit.cuda()
            pX = x.clone()
            pX.cuda()
            fMin = float(min(pFit))
            print(fMin)
            bestIndex = torch.argmin(pFit)
            bestX = pX[bestIndex, :]
            # print(bestX)
            b2 = torch.zeros([M, 1])
            b2[0] = fMin
        #     Start the iteration.
        if conti_iteration:
            inter_start = conti_iteration
        else:
            inter_start = 0
        for index in range(inter_start, M):
            print(f'Iteration {index}')
            prob = torch.rand(pop, 1) * 0.2 + 0.8
            if (mod(index, FQ) != 0):
                ###############################################################################
                #     Birds forage for food or keep vigilance
                sumPfit = pFit.sum().cuda()
                meanP = torch.mean(pX).cuda()
                realmin = torch.tensor(finfo(float).tiny).cuda()
                for i in range(0, pop):
                    if torch.rand(1).cuda() < float(prob[i]):
                        x[i, :] = x[i, :] + c1 * torch.rand(1).cuda() * (
                            bestX - x[i, :]) + c2 * torch.rand(1).cuda() * (
                                pX[i, :] - x[i, :])
                    else:
                        person = int(self.randiTabu(1, pop, i, 1)[0])
                        x[i, :] = x[i, :] + torch.rand(1).cuda() * (
                            meanP - x[i, :]) * a1 * torch.exp(
                                -pFit[i] / (sumPfit + realmin) * pop) + a2 * (
                                    torch.rand(1).cuda() * 2 - 1
                                ) * (pX[person, :] - x[i, :]) * torch.exp(
                                    -(pFit[person] - pFit[i]) /
                                    (abs(pFit[person] - pFit[i]) + realmin) *
                                    pFit[person] / (sumPfit + realmin) * pop)
    #                 print(x[i,:])
                    x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    #                 print(x[i,:])
                fit = self.get_objective_function(x, index, M)
    ###############################################################################
            else:
                FL = torch.rand(pop, 1) * 0.4 + 0.5
                ###############################################################################
                #     Divide the bird swarm into two parts: producers and scroungers
                minIndex = torch.argmin(pFit)
                maxIndex = torch.argmax(pFit)
                choose = 0
                if (minIndex < 0.5 * pop and maxIndex < 0.5 * pop):
                    choose = 1
                elif (minIndex > 0.5 * pop and maxIndex < 0.5 * pop):
                    choose = 2
                elif (minIndex < 0.5 * pop and maxIndex > 0.5 * pop):
                    choose = 3
                elif (minIndex > 0.5 * pop and maxIndex > 0.5 * pop):
                    choose = 4
    ###############################################################################
                if choose < 3:
                    for i in range(int(pop / 2 + 1) - 1, pop):
                        x[i, :] = x[i, :] * (1 + torch.randn(1).cuda())
                        #                     print(x[i,:])
                        x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    fit[int(pop / 2 + 1) -
                        1:pop] = self.get_objective_function(
                            x[int(pop / 2 + 1) - 1:pop, :], index, M)
                    if choose == 1:
                        x[minIndex, :] = x[minIndex, :] * (
                            1 + torch.randn(1).cuda())
                        x[minIndex, :] = self.Bounds(x[minIndex, :], self.lb,
                                                     self.ub)
                        fit[minIndex] = self.get_objective_function(
                            x[minIndex - 1:minIndex, :], index, M)
                    for i in range(0, int(0.5 * pop)):
                        if choose == 2 or minIndex != i:
                            # print(type(pop))
                            person = random.randint(0.5 * pop + 1, pop)
                            # print(person)
                            x[i, :] = x[i, :] + (pX[person, :] -
                                                 x[i, :]) * FL[i].cuda()
                            x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    fit[0:int(0.5 * pop)] = self.get_objective_function(
                        x[0:int(0.5 * pop), :], index, M)
                else:
                    for i in range(0, int(0.5 * pop)):
                        x[i, :] = x[i, :] * (1 + torch.randn(1).cuda())
                        x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    fit[minIndex] = self.get_objective_function(
                        x[minIndex - 1:minIndex, :], index, M)
                    if choose == 4:
                        x[minIndex, :] = x[minIndex, :] * (
                            1 + torch.randn(1).cuda())
                        x[minIndex, :] = self.Bounds(x[minIndex, :], self.lb,
                                                     self.ub)
                        fit[minIndex] = self.get_objective_function(
                            x[minIndex - 1:minIndex, :], index, M)
                    for i in range(int(0.5 * pop), pop):
                        if choose == 3 or minIndex != i:
                            # print(type(pop))
                            person = random.randint(1, 0.5 * pop + 1)
                            # print(person)
                            x[i, :] = x[i, :] + (pX[person, :] -
                                                 x[i, :]) * FL[i].cuda()
                            x[i, :] = self.Bounds(x[i, :], self.lb, self.ub)
                    fit[int(0.5 * pop):pop] = self.get_objective_function(
                        x[int(0.5 * pop):pop, :], index, M)

    ###############################################################################
    #     Update the individual's best fitness value and the global best one
            for i in range(0, pop):
                if (fit[i] < pFit[i]):
                    pFit[i] = fit[i]
                    pX[i, :] = x[i, :]
                if (pFit[i] < fMin):
                    fMin = pFit[i]
            fMin = float(min(pFit))
            bestIndex = torch.argmin(pFit)
            bestX = pX[bestIndex, :]
            b2[index] = fMin
            torch.save(x, "x9.pt")
            torch.save(fit, "fit9.pt")
            torch.save(pFit, "pFit9.pt")
            torch.save(pX, "pX9.pt")
            torch.save(fMin, "fMin9.pt")
            torch.save(bestIndex, "bestIndex9.pt")
            torch.save(bestX, "bestX9.pt")
            torch.save(b2, "b29.pt")
            torch.save(index, "index.pt")
        # print(fMin)
        return fMin, bestIndex, bestX, b2
