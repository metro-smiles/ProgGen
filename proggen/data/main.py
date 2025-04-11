#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp, sys
import h5py
import json
import numpy as np
from dotmap import DotMap

from .perception_hard import percept_hard, decode_hdf5_to_frames

def get_dataset_name(name, split):
    if name == 'parabola':
        if split == 'train':
            return 'parabola_30K'
        elif split == 'test':
            return 'parabola_eval'
        else:
            raise ValueError(f'Unknown split: {split} for dataset: {name}')
    elif name == 'uniform_motion':
        if split == 'train':
            return 'uniform_motion_30K'
        elif split == 'test':
            return 'uniform_motion_eval'
        else:
            raise ValueError(f'Unknown split: {split} for dataset: {name}')
    elif name == 'collision':
        if split == 'train':
            return 'collision_30K'
        elif split == 'test':
            return 'collision_eval'
        else:
            raise ValueError(f'Unknown split: {split} for dataset: {name}')
    else:
        raise ValueError(f'Unknown dataset: {name}')
class Dataset:
    def __init__(self, name, split, seed=0, novideo=False,):
        curdir = osp.dirname(os.path.abspath(__file__))
        name = get_dataset_name(name, split)
        self.data_fn = osp.join(curdir, 'downloaded_datasets', f'{name}.hdf5')
        assert osp.exists(self.data_fn)
        self.percepted_data_fn = osp.join(curdir, 'percepted_hard_datasets', f'{name}_percepted')
        os.makedirs(self.percepted_data_fn, exist_ok=True)

        data_f = h5py.File(self.data_fn, 'r')
        split_indexes = data_f['video_streams'].keys()
        indexes = [(si, ti) for si in split_indexes for ti in range(len(data_f['video_streams'][si]))]
        rng = np.random.RandomState(seed)
        self.indexes = [indexes[i] for i in rng.permutation(len(indexes))]
        data_f.close()
        self.novideo = novideo
    def __len__(self):
        return len(self.indexes)
    def __getitem__(self, i):
        si, ti = self.indexes[i]
        if osp.exists(self.percepted_data_fn + f'_{si}_{ti}.json'):
            try:
                with open(self.percepted_data_fn + f'_{si}_{ti}.json', 'r') as f:
                    percepted_data = json.load(f)
            except:
                os.remove(self.percepted_data_fn + f'_{si}_{ti}.json')
                percept_hard(osp.basename(self.data_fn), si, ti,)
                with open(self.percepted_data_fn + f'_{si}_{ti}.json', 'r') as f:
                    percepted_data = json.load(f)
        else:
            percept_hard(osp.basename(self.data_fn), si, ti,)
            with open(self.percepted_data_fn + f'_{si}_{ti}.json', 'r') as f:
                percepted_data = json.load(f)

        obj_list = {c for sps in percepted_data['shapes'] for c in sps}
        obj2name = {obj: f'shape{i}' for i, obj in enumerate(sorted(obj_list))}
        trajs = []
        for sps in percepted_data['shapes'][:percepted_data['bad_index_start']]:
            state = {obj2name[obj]: {
                'position': s['position'],
                'angle': 0,
                'shape': 'circle',
                'radius': s['radius'],
                'velocity': None,
                'angular_velocity': None,
            } for obj, s in sps.items()}
            trajs.append(DotMap(state, _dynamic=False))

        gt_positions = percepted_data['gt_feats'][:percepted_data['bad_index_start']]
        assert len(gt_positions) == len(trajs), f'{len(gt_positions)} != {len(trajs)}'

        gt_inits = percepted_data['gt_init']
        if 'collision' in self.data_fn:
            split = split_indices_collision(gt_inits)
        elif 'parabola' in self.data_fn:
            split = split_indices_parabola(gt_inits)
        elif 'uniform_motion' in self.data_fn:
            split = split_indices_uniform(gt_inits)
        else:
            raise ValueError(f'Unknown dataset: {self.data_fn}')


        if self.novideo:
            return {
                'trajs': trajs,
                'gt_positions': gt_positions,
                'gt_inits': gt_inits,
                'split': split,
            }

        frames, fps = decode_hdf5_to_frames(self.data_fn, si, ti)
        frames = frames[:percepted_data['bad_index_start']]
        assert len(frames) == len(trajs), f'{len(frames)} != {len(trajs)}'

        return {
            'trajs': trajs,
            'gt_positions': gt_positions,
            'gt_inits': gt_inits,
            'split': split,
            'frames': frames,
            'fps': fps,
        }
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

# Copied from Phyworld
# https://raw.githubusercontent.com/phyworld/phyworld/e92934249ecd5a9ffbb587017dbd5be33593f4da/id_ood_data/evaluate.py
def split_indices_uniform(init):
    MIN_V = 1.0
    MAX_V = 4.0
    MIN_R = 0.7
    MAX_R = 1.4
    if init[0] < 0.6: # filter out dispearing balls
        return 'invalid'
    if init[1] == 0: #! (r, v)
        return 'zero'
    elif MIN_R <= init[0] <= MAX_R and MIN_V <= init[1] <= MAX_V:
        return 'in'
    elif not (MIN_R <= init[0] <= MAX_R) and MIN_V <= init[1] <= MAX_V:
        return 'r_out'
    elif (MIN_R <= init[0] <= MAX_R) and not (MIN_V <= init[1] <= MAX_V):
        return 'v_out'
    elif not (MIN_R <= init[0] <= MAX_R) and not (MIN_V <= init[1] <= MAX_V):
        return 'rv_out'
    else:
        raise ValueError('Unexpected case')
def split_indices_parabola(init):
    MIN_V = 1.0
    MAX_V = 4.0
    MIN_R = 0.7
    MAX_R = 1.4
    if init[0] < 0.6: # filter out dispearing balls
        return 'invalid'
    if init[1] == 0:
        return 'zero'
    elif MIN_R <= init[0] <= MAX_R and MIN_V <= init[1] <= MAX_V:
        return 'in'
    elif not (MIN_R <= init[0] <= MAX_R) and MIN_V <= init[1] <= MAX_V:
        return 'r_out'
    elif (MIN_R <= init[0] <= MAX_R) and not (MIN_V <= init[1] <= MAX_V):
        return 'v_out'
    elif not (MIN_R <= init[0] <= MAX_R) and not (MIN_V <= init[1] <= MAX_V):
        return 'rv_out'
    else:
        raise ValueError('Unexpected case')
def split_indices_collision(init):
    MIN_V = 1.0
    MAX_V = 4.0
    MIN_R = 0.5
    MAX_R = 1.5
    if init[0] <= 0.5 or init[1] <=0.5: # filter out dispearing balls
        return 'invalid'
    # r1, r2, v1, v2
    if init[2] == 0 or init[3] == 0:
        return 'zero'
    elif (MIN_R <= init[0] <= MAX_R and MIN_R <= init[1] <= MAX_R) and \
        (MIN_V <= init[2] <= MAX_V and MIN_V <= init[3] <= MAX_V):
        return 'in'
    elif not (MIN_R <= init[0] <= MAX_R and MIN_R <= init[1] <= MAX_R) and \
        (MIN_V <= init[2] <= MAX_V and MIN_V <= init[3] <= MAX_V):
        return 'r_out'
    elif (MIN_R <= init[0] <= MAX_R and MIN_R <= init[1] <= MAX_R) and \
        not (MIN_V <= init[2] <= MAX_V and MIN_V <= init[3] <= MAX_V):
        return 'v_out'
    elif not (MIN_R <= init[0] <= MAX_R and MIN_R <= init[1] <= MAX_R) and \
        not (MIN_V <= init[2] <= MAX_V and MIN_V <= init[3] <= MAX_V):
        return 'rv_out'
    else:
        raise ValueError('Unexpected case')
