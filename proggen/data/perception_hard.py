#!/usr/bin/env python
# coding=utf-8

import random
import numpy as np
from collections import Counter
from skimage import measure
import os, os.path as osp
import h5py
import tempfile
import imageio
import json
from dotmap import DotMap

WORLD_SCALE = 10.

def decode_hdf5_to_frames(hdf, split_index, trial_index,):
    if isinstance(hdf, str):
        hdf = h5py.File(hdf, 'r')
    """Decode video frames from a byte stream stored in an HDF5 file by first writing to a temporary file."""
    byte_stream = hdf['video_streams'][split_index][trial_index]
    byte_obj = byte_stream.tobytes()
    # Use a temporary file to write the byte stream
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(byte_obj)

    # Now read the video from the temporary file
    with imageio.get_reader(temp_file_name, format='mp4') as reader:
        frames = [frame for frame in reader]
        fps = reader.get_meta_data()['fps']

    return np.array(frames), fps
def bad_index(shapes):
    if len(shapes) == 0:
        return True
    return any([s['mask'][0].any() or s['mask'][-1].any() or s['mask'][:, 0].any() or s['mask'][:, -1].any() for s in shapes.values()])
def _true_bad_index(gt_feats, gt_init):
    r, _ = gt_init
    x, y = gt_feats.T
    return np.stack([x < r, x > WORLD_SCALE - r, y < r, y > WORLD_SCALE - r]).any(axis=0)
def to_serializable(obj):
    if isinstance(obj, DotMap):
        return to_serializable(obj.toDict())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (list, tuple)):
        return [to_serializable(o) for o in obj]
    if isinstance(obj, dict):
        return {k if isinstance(k, tuple) else str(k): to_serializable(v) for k, v in obj.items()}
    return obj


def img2shapes(img, verbose=True, WORLD_SCALE=10.):
    # Assuming all circles with distinct colors such as R, G, B
    assert len(img.shape) == 3, f"Image shape is not 3D: {img.shape}"
    assert img.shape[2] == 3, f"Image shape is not 3 channels: {img.shape}"
    assert img.shape[0] == img.shape[1], f"Image shape is not square: {img.shape}"

    masks = [img[:,:,i] > 127 for i in range(3)]
    shapes = dict()
    for i in range(3):
        mask = masks[i]
        for j in range(3):
            if i == j: continue
            mask = mask & np.logical_not(masks[j])
        if not (mask.any() and mask.sum() > 0.001 * img.shape[0] * img.shape[1]):
            continue

        area = mask.sum()
        radius = np.sqrt(area / np.pi) / img.shape[0] * WORLD_SCALE
        center = np.mean(np.argwhere(mask), axis=0)[::-1] / img.shape[0] * WORLD_SCALE
        center[1] = WORLD_SCALE - center[1]
        shapes[i] = {
            'mask': mask,
            'area': area,
            'radius': radius,
            'position': center,
        }

    return shapes
def percept_hard(fn=None, split=None, trial=None,):
    assert (fn is None) == (split is None) == (trial is None), f"fn, split, and trial must be all None or all not None"
    curdir = osp.dirname(osp.abspath(__file__))
    indir = osp.join(curdir, 'downloaded_datasets')
    outdir = osp.join(curdir, 'percepted_hard_datasets')
    os.makedirs(outdir, exist_ok=True)
    fn_list = os.listdir(indir)
    random.shuffle(fn_list)
    if fn is not None: fn_list = [fn]
    for fn in fn_list:
        if fn.endswith('.hdf5'):
            print(f'Processing {fn}')
            out_fn = osp.join(outdir, fn.replace('.hdf5', '_percepted.json'))
            if osp.exists(out_fn):
                print(f'Skipping {fn}')
                continue
            in_f = h5py.File(osp.join(indir, fn), 'r')
            out = dict()
            split_name_list = list(in_f['position_streams'])
            random.shuffle(split_name_list)
            if split is not None: split_name_list = [split]
            for split_name in split_name_list:
                assert split_name in in_f['video_streams']
                assert split_name in in_f['init_streams']
                split_out_fn = out_fn.replace('.json', f'_{split_name}.json')
                if osp.exists(split_out_fn):
                    with open(split_out_fn, 'r') as f:
                        out[split_name] = json.load(f)
                    continue
                print(f'\tProcessing {fn} {split_name}')
                out[split_name] = dict()
                trial_index_list = list(range(len(in_f['position_streams'][split_name])))
                random.shuffle(trial_index_list)
                if trial is not None: trial_index_list = [trial]
                for trial_index in trial_index_list:
                    trial_out_fn = split_out_fn.replace('.json', f'_{trial_index}.json')
                    if osp.exists(trial_out_fn):
                        with open(trial_out_fn, 'r') as f:
                            out[split_name][trial_index] = json.load(f)
                        continue
                    print(f'\t\tProcessing {fn} {split_name} {trial_index}')
                    out[split_name][trial_index] = dict()
                    out[split_name][trial_index]['gt_feats'] = np.array(in_f['position_streams'][split_name][trial_index])
                    out[split_name][trial_index]['gt_init'] = np.array(in_f['init_streams'][split_name][trial_index])

                    frames, fps = decode_hdf5_to_frames(in_f, split_name, trial_index,)
                    out[split_name][trial_index]['fps'] = fps
                    out[split_name][trial_index]['shapes'] = [
                        img2shapes(fm, WORLD_SCALE=WORLD_SCALE, verbose=False,)
                        for fm in frames
                    ]

                    for i, sps in enumerate(out[split_name][trial_index]['shapes']):
                        if bad_index(sps) or len(sps) < len(out[split_name][trial_index]['shapes'][0]):
                            out[split_name][trial_index]['bad_index_start'] = i
                            break;
                    if 'bad_index_start' not in out[split_name][trial_index]:
                        out[split_name][trial_index]['bad_index_start'] = len(out[split_name][trial_index]['shapes'])

                    objects = {c for sps in out[split_name][trial_index]['shapes'][:out[split_name][trial_index]['bad_index_start']] for c in sps}
                    print(f'\t\t\tFound {len(objects)} objects: {objects}')
                    # objects = {str((252, 0, 0))}
                    pred_positions = np.array([[sps[c]['position'] for sps in out[split_name][trial_index]['shapes'][:out[split_name][trial_index]['bad_index_start']]] for c in objects])
                    gt_positions = out[split_name][trial_index]['gt_feats'][:out[split_name][trial_index]['bad_index_start']]
                    if len(gt_positions.shape) == 2:
                        assert pred_positions.shape[0] == 1
                        pred_positions = pred_positions[0]
                    else:
                        pred_positions = pred_positions.transpose(1, 0, 2)
                    assert pred_positions.shape == gt_positions.shape, f"Prediction shape is not equal to ground truth shape: {pred_positions.shape} != {gt_positions.shape}"
                    pred_positions = pred_positions.reshape(-1, pred_positions.shape[-1])
                    gt_positions = gt_positions.reshape(-1, gt_positions.shape[-1])
                    mae = np.mean(np.linalg.norm(pred_positions - gt_positions, axis=1, ord=1))
                    mse = np.mean(np.linalg.norm(pred_positions - gt_positions, axis=1, ord=2) ** 2)
                    r2 = 1 - mse / np.mean(np.linalg.norm(gt_positions - np.mean(gt_positions, axis=0), axis=1) ** 2)
                    print(f'\t\t\tMAE: {mae:.3f}; MSE: {mse:.3f}; R2: {r2:.3f}')

                    with open(trial_out_fn, 'w') as jsonf:
                        json.dump(to_serializable(out[split_name][trial_index]), jsonf)
                # with open(split_out_fn, 'w') as jsonf:
                    # json.dump(to_serializable(out[split_name]), jsonf)
                # for trial_index in trial_index_list:
                    # trial_out_fn = split_out_fn.replace('.json', f'_{trial_index}.json')
                    # os.remove(trial_out_fn)
            # with open(out_fn, 'w') as jsonf:
                # json.dump(to_serializable(out), jsonf)
            # for split_name in split_name_list:
                # split_out_fn = out_fn.replace('.json', f'_{split_name}.json')
                # os.remove(split_out_fn)
            in_f.close()

    print('Done')

if __name__ == '__main__':
    percept_hard()
