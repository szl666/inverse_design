import os
import time
import argparse
import torch
import collections
import warnings

import numpy as np
# from bird_swarm_opt_mo import Bird_swarm_opt
# from bird_swarm_opt_mo1 import Bird_swarm_opt
from bird_swarm_opt_mo2 import Bird_swarm_opt
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


def reconstruction(loader,
                   model,
                   ld_kwargs,
                   num_evals,
                   force_num_atoms=False,
                   force_atom_types=False,
                   down_sample_traj_step=1):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(loader)}')
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stoichaticity in langevin dynamics
        _, _, z = model.encode(batch)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None
            outputs = model.langevin_dynamics(z, ld_kwargs, gt_num_atoms,
                                              gt_atom_types)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lengths.append(outputs['lengths'].detach().cpu())
            batch_angles.append(outputs['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    outputs['all_frac_coords']
                    [::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    outputs['all_atom_types']
                    [::down_sample_traj_step].detach().cpu())
        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(
                torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(
                torch.stack(batch_all_atom_types, dim=0))
        # Save the ground truth structure
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack, input_data_batch)


def get_sym_coords(space_groups, atom_types, num_atoms, lengths, angles):
    location = 0
    for index, num_atom in enumerate(num_atoms):
        num_atom = int(num_atom)
        space_group = np.argmax(space_groups.detach().cpu().numpy()[index]) + 1
        atom_type = atom_types.detach().cpu().numpy()[location:location +
                                                      num_atom]
        length = lengths.detach().cpu().numpy()[index]
        angle = angles.detach().cpu().numpy()[index]
        try:
            sp_crystal = pyxtal()
            cell = Lattice.from_para(length[0], length[1], length[2], angle[0],
                                     angle[1], angle[2])
            com_dict = {}
            for ato in atom_type:
                ato_symbol = element(int(ato)).symbol
                com_dict[ato_symbol] = com_dict.get(ato_symbol, 0) + 1
            elements = list(com_dict.keys())
            composition = list(com_dict.values())
            print(length)
            print(angle)
            print(elements)
            print(composition)
            print(space_group)
            # sp_crystal.from_random(3,
            #                        space_group,
            #                        elements,
            #                        composition,
            #                        lattice=cell)
            sp_crystal.from_random(3, space_group, elements, composition)
            struct = sp_crystal.to_pymatgen()
            poscar = Poscar(structure=struct)
            file_path = 'gen_poscars1'
            poscar.write_file(f'{file_path}/POSCAR_{index}')
            if index == 0:
                sym_frac_coords = struct.frac_coords
            else:
                sym_frac_coords = np.concatenate(
                    (sym_frac_coords, struct.frac_coords))
        except:
            print(index)
        location += num_atom
    return sym_frac_coords


def generation(model,
               ld_kwargs,
               num_batches_to_sample,
               num_samples_per_z,
               batch_size=512,
               down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        # batch_all_frac_coords = []
        # batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size,
                        model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            # samples = model.langevin_dynamics(z, ld_kwargs)
            num_atoms, _, lengths, angles, composition_per_atom = model.decode_stats(
                z)
            space_groups = model.predict_space_groups(z)
            composition_per_atom = F.softmax(composition_per_atom, dim=-1)
            atom_types = model.sample_composition(composition_per_atom,
                                                  num_atoms)
            sym_frac_coords = get_sym_coords(space_groups, atom_types,
                                             num_atoms, lengths, angles)
            # collect sampled crystals in this batch.
            batch_frac_coords.append(sym_frac_coords)
            batch_num_atoms.append(num_atoms)
            batch_atom_types.append(atom_types)
            batch_lengths.append(lengths)
            batch_angles.append(angles)
            # if ld_kwargs.save_traj:
            #     batch_all_frac_coords.append(
            #         samples['all_frac_coords']
            #         [::down_sample_traj_step].detach().cpu())
            #     batch_all_atom_types.append(
            #         samples['all_atom_types']
            #         [::down_sample_traj_step].detach().cpu())

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        # if ld_kwargs.save_traj:
        #     all_frac_coords_stack.append(
        #         torch.stack(batch_all_frac_coords, dim=0))
        #     all_atom_types_stack.append(
        #         torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    # if ld_kwargs.save_traj:
    #     all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
    #     all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


def optimization(model,
                 ld_kwargs,
                 data_loader,
                 num_starting_points=100,
                 num_steps=500,
                 interval=10,
                 lr=1e-3):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
    else:
        z = torch.randn(num_starting_points,
                        model.hparams.hidden_dim,
                        device=model.device)
    model.freeze()
    bsa = Bird_swarm_opt(opt_matrix=z,
                         cdvae_model=model,
                         ld_kwargs=ld_kwargs,
                         ck_path_ehull='model_ehull.pt',
                         ck_path_gap='model_gap.pt',
                         ck_path_pbx='model_pbx.pt',
                         interval=interval)
    # fMin, bestIndex, bestX, b2 = bsa.search(M=num_steps)
    conti_iter = int(torch.load("index.pt"))
    fMin, bestIndex, bestX, b2 = bsa.search(M=num_steps,
                                            conti=True,
                                            conti_iteration=conti_iter)
    # np.save('fMin.npy', fMin)
    # np.save('bestIndex.npy', bestIndex)
    # np.save('bestX.npy', bestX)
    # np.save('b2.npy', b2)
    all_crystals = bsa.all_crystals
    return {
        k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0)
        for k in
        ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']
    }


# def check_folder_update(folder_path, time_delta_seconds):
#     current_time = time.time()
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             last_modified_time = os.path.getmtime(file_path)
#             if current_time - last_modified_time <= time_delta_seconds:
#                 return True
#     return False

# if check_folder_update("/path/to/folder", 0.5 * 60 * 60):
#     print("The folder has been updated in the last 24 hours.")
# else:
#     os.system('scancel --state=RUNNING')
#     os.system('sbatch sub_gpu')


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path,
        load_data=('recon' in args.tasks)
        or ('opt' in args.tasks and args.start_from == 'data'))
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack,
         input_data_batch) = reconstruction(test_loader, model, ld_kwargs,
                                            args.num_evals,
                                            args.force_num_atoms,
                                            args.force_atom_types,
                                            args.down_sample_traj_step)

        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        torch.save(
            {
                'eval_setting': args,
                'input_data_batch': input_data_batch,
                'frac_coords': frac_coords,
                'num_atoms': num_atoms,
                'atom_types': atom_types,
                'lengths': lengths,
                'angles': angles,
                'all_frac_coords_stack': all_frac_coords_stack,
                'all_atom_types_stack': all_atom_types_stack,
                'time': time.time() - start_time
            }, model_path / recon_out_name)

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
             model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
             args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save(
            {
                'eval_setting': args,
                'frac_coords': frac_coords,
                'num_atoms': num_atoms,
                'atom_types': atom_types,
                'lengths': lengths,
                'angles': angles,
                'all_frac_coords_stack': all_frac_coords_stack,
                'all_atom_types_stack': all_atom_types_stack,
                'time': time.time() - start_time
            }, model_path / gen_out_name)

    if 'opt' in args.tasks:
        print('Evaluate model on the property optimization task.')
        start_time = time.time()
        if args.start_from == 'data':
            loader = test_loader
        else:
            loader = None
        optimized_crystals = optimization(model, ld_kwargs, loader)
        optimized_crystals.update({
            'eval_setting': args,
            'time': time.time() - start_time
        })

        if args.label == '':
            gen_out_name = 'eval_opt.pt'
        else:
            gen_out_name = f'eval_opt_{args.label}.pt'
        torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='no', type=str)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()

    main(args)
