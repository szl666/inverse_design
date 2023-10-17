import numpy as np
from predict_energy1 import *
import warnings
import torch
from mendeleev import element
from scipy.stats.mstats import gmean


def cal_mulliken(ele):
    elem = element(ele)
    ea = elem.electron_affinity
    ea = ea if ea else 0
    ip = elem.ionenergies.get(1, None)
    ip = ip if ip else 0
    return (ea + ip) / 2


def cal_gap_vb_cb(structure, band_gap, ehull, Gpbx):
    species_en_mulliken = [cal_mulliken(str(i)) for i in structure.species]
    if band_gap <= 0:
        band_gap = 0
    if ehull <= 0:
        ehull = 0
    if Gpbx <= 0:
        Gpbx = 0
    if ehull >= 20:
        ehull = 2
    if Gpbx >= 20:
        Gpbx = 2
    Xs = gmean(species_en_mulliken)
    Ecb = Xs - 4.44 - 0.5 * band_gap
    Evb = Xs - 4.44 + 0.5 * band_gap
    return Ecb, Evb, band_gap, Gpbx, ehull


ck_path_ehull = 'model_ehull.pt'
ck_path_gap = 'model_gap.pt'
ck_path_pbx = 'model_pbx.pt'
ehull_model = load_dimenet(ck_path_ehull, 512, 256)
gap_model = load_dimenet(ck_path_gap, 512, 256)
pbx_model = load_dimenet(ck_path_pbx, 256, 192)
structs = np.load('stable_structs.npy', allow_pickle=True)
_, band_gaps = predict_ene_batch(structs,
                                 gap_model,
                                 mean=1.497423233546115,
                                 std=2.3198268552027064)
_, ehulls = predict_ene_batch(structs,
                              ehull_model,
                              mean=1.7750069410012883,
                              std=1.3928968647223894)
_, Gpbxs = predict_ene_batch(structs,
                             pbx_model,
                             mean=1.4296479801859339,
                             std=1.7844662928101613)
objectives = []
for struct, band_gap, ehull, Gpbx in zip(structs, band_gaps, ehulls, Gpbxs):
    objectives.append(cal_gap_vb_cb(struct, band_gap, ehull, Gpbx))
np.save('objectives.npy', objectives)