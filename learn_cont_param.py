#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import itertools
import json
from pprint import pprint
from dotmap import DotMap

import numpy as np
import matplotlib.pyplot as plt
import argparse

from proggen.prog import FixtureDef, BodyDef, JointDef, ContactableGraph
from proggen.prog import Box2DProgram, ContParams
from proggen.prog.prog import _metrics_fn
from proggen.utils.logger import set_logger
from proggen.utils.render import PygameRender
from proggen.data import Dataset

def render_trajs(trajs):
    render = PygameRender(FPS, 256, 256, 256/10.)
    for ti, s in enumerate(trajs):
        render.render(s)
    render.close()
def plot_trajs_diff(trajs, pred_trajs, gt_positions):
    assert len(trajs[0]) == 1, 'only support single object'
    H, W = 2, 2
    fig, axes = plt.subplots(H, W, figsize=(H*5, W*5))
    mae0, mae1 = 0, 0
    pred_vel_err_avg, gt_vel_err_avg = 0, 0
    pred_vel_error, gt_vel_error = 0, 0
    for oi, obj in enumerate(sorted(trajs[0])):
        gt = np.array([s[obj]['position'] for s in trajs])
        pred = np.array([s[obj]['position'] for s in pred_trajs])
        gt_pos = np.array([s for s in gt_positions])
        axes[0,0].plot(gt[:, 0], gt[:, 1], 'rx-')
        axes[0,0].plot(pred[:, 0], pred[:, 1], 'bx-')
        axes[0,0].plot(gt_pos[:, 0], gt_pos[:, 1], 'gx-')
        mae0 += np.mean(np.abs(gt[1:,0] - pred[1:,0]))
        mae1 += np.mean(np.abs(gt[1:,1] - pred[1:,1]))

        axes[0,1].plot(gt[:,0], 'rx-')
        axes[0,1].plot(pred[:,0], 'bx-')
        axes[0,1].plot(gt_pos[:,0], 'gx-')

        gt_vel = np.diff(gt, axis=0) * FPS
        pred_vel = np.diff(pred, axis=0) * FPS
        gt_pos_vel = np.diff(gt_pos, axis=0) * FPS
        axes[1,0].plot(gt_vel[:,0], 'rx-')
        axes[1,0].plot(pred_vel[:,0], 'bx-')
        axes[1,0].plot(gt_pos_vel[:,0], 'gx-')
        gt_vel_err = np.abs(gt_pos_vel[:,0] - gt_vel[:,0])
        pred_vel_err = np.abs(pred_vel[:,0] - gt_vel[:,0])
        axes[1,1].plot(gt_vel_err, 'rx-')
        axes[1,1].plot(pred_vel_err, 'bx-')
        axes[1,1].set_ylim(0, 0.5)
        pred_vel_err_avg += np.mean(np.abs(pred_vel[4:,0] - gt_pos_vel[4:,0]))
        gt_vel_err_avg += np.mean(np.abs(gt_vel[4:,0] - gt_pos_vel[4:,0]))
        pred_vel_error += np.abs(pred_vel[4:,0].mean() - gt_pos_vel[4:,0].mean())
        gt_vel_error += np.abs(gt_vel[4:,0].mean() - gt_pos_vel[4:,0].mean())
    mae0 /= len(trajs[0])
    mae1 /= len(trajs[0])
    mae = (mae0 + mae1) / 2
    axes[0,0].set_title(f'mae: {mae:.4f}; mae0: {mae0:.4f}; mae1: {mae1:.4f}')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,1].set_xlabel('t')
    axes[0,1].set_ylabel('x')
    axes[1,0].set_xlabel('t')
    axes[1,0].set_ylabel('vx')
    pred_vel_err_avg /= len(trajs[0])
    gt_vel_err_avg /= len(trajs[0])
    pred_vel_error /= len(trajs[0])
    gt_vel_error /= len(trajs[0])
    axes[1,1].set_title(f'pred_vel_err_avg: {pred_vel_err_avg:.4f}, gt_vel_err_avg: {gt_vel_err_avg:.4f}\n'
                        f'pred_vel_error: {pred_vel_error:.4f}, gt_vel_error: {gt_vel_error:.4f}')
    axes[1,1].set_xlabel('t')
    axes[1,1].set_ylabel('vx_err')
    plt.show()
def plot_trajs_diff_collision(trajs, pred_trajs, gt_positions):
    H, W = 2, 2
    fig, axes = plt.subplots(H, W, figsize=(H*5, W*5))
    mae0, mae1 = 0, 0
    pred_post_vel_err_avg, gt_post_vel_err_avg = 0, 0
    pred_post_vel_error, gt_post_vel_error = 0, 0
    for oi, obj in enumerate(sorted(trajs[0])):
        gt = np.array([s[obj]['position'] for s in trajs])
        pred = np.array([s[obj]['position'] for s in pred_trajs])
        gt_pos = np.array([s[oi] for s in gt_positions])
        axes[0,0].plot(gt[:, 0], gt[:, 1], 'rx-')
        axes[0,0].plot(pred[:, 0], pred[:, 1], 'bx-')
        axes[0,0].plot(gt_pos[:, 0], gt_pos[:, 1], 'gx-')
        mae0 += np.mean(np.abs(gt[1:,0] - pred[1:,0]))
        mae1 += np.mean(np.abs(gt[1:,1] - pred[1:,1]))

        axes[0,1].plot(gt[:,0], 'rx-')
        axes[0,1].plot(pred[:,0], 'bx-')
        axes[0,1].plot(gt_pos[:,0], 'gx-')

        gt_vel = np.diff(gt, axis=0) * FPS
        pred_vel = np.diff(pred, axis=0) * FPS
        gt_pos_vel = np.diff(gt_pos, axis=0) * FPS
        axes[1,0].plot(gt_vel[:,0], 'rx-')
        axes[1,0].plot(pred_vel[:,0], 'bx-')
        axes[1,0].plot(gt_pos_vel[:,0], 'gx-')
        gt_vel_err = np.abs(gt_pos_vel[:,0] - gt_vel[:,0])
        pred_vel_err = np.abs(pred_vel[:,0] - gt_vel[:,0])
        axes[1,1].plot(gt_vel_err, 'rx-')
        axes[1,1].plot(pred_vel_err, 'bx-')
        axes[1,1].set_ylim(0, 0.5)
        pred_post_vel_err_avg += np.mean(np.abs(pred_vel[9:,0] - gt_pos_vel[9:,0]))
        gt_post_vel_err_avg += np.mean(np.abs(gt_vel[9:,0] - gt_pos_vel[9:,0]))
        pred_post_vel_error += np.abs(pred_vel[9:,0].mean() - gt_pos_vel[9:,0].mean())
        gt_post_vel_error += np.abs(gt_vel[9:,0].mean() - gt_pos_vel[9:,0].mean())
    mae0 /= len(trajs[0])
    mae1 /= len(trajs[0])
    mae = (mae0 + mae1) / 2
    axes[0,0].set_title(f'mae: {mae:.4f}; mae0: {mae0:.4f}; mae1: {mae1:.4f}')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,1].set_xlabel('t')
    axes[0,1].set_ylabel('x')
    axes[1,0].set_xlabel('t')
    axes[1,0].set_ylabel('vx')
    pred_post_vel_err_avg /= len(trajs[0])
    gt_post_vel_err_avg /= len(trajs[0])
    pred_post_vel_error /= len(trajs[0])
    gt_post_vel_error /= len(trajs[0])
    axes[1,1].set_title(f'pred_post_vel_err_avg: {pred_post_vel_err_avg:.4f}, gt_post_vel_err_avg: {gt_post_vel_err_avg:.4f}\n'
                        f'pred_post_vel_error: {pred_post_vel_error:.4f}, gt_post_vel_error: {gt_post_vel_error:.4f}')
    axes[1,1].set_xlabel('t')
    axes[1,1].set_ylabel('vx_err')
    plt.show()
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
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj
def get_program(dataset):
    if dataset in ['uniform_motion', 'parabola']:
        fixture_names = ['shape0',]
        fixtures = {
            fn: FixtureDef(fn, 'circle',)
            for fn in fixture_names
        }
        contactable_graph = ContactableGraph({tuple(sorted([k1, k2])): True for k1, k2 in itertools.product(fixtures.keys(), repeat=2)})
        print('contactable graph:', {tuple(sorted([k1, k2])) for k1, k2 in itertools.combinations(fixtures.keys(), 2) if contactable_graph.query_category(k1) & contactable_graph.query_mask(k2) and contactable_graph.query_category(k2) & contactable_graph.query_mask(k1)})
        body_def = {'ball1': BodyDef('ball1', ['shape0'], 'dynamic'),}
        print('body:', {bn: body.body_type for bn, body in body_def.items()})
        joint_def = dict()
        print('joint:', {jn: joint.joint_type for jn, joint in joint_def.items()})
        program = Box2DProgram(fixtures, contactable_graph, body_def, joint_def)
    elif dataset in ['collision']:
        fixture_names = ['shape0', 'shape1']
        fixtures = {
            fn: FixtureDef(fn, 'circle',)
            for fn in fixture_names
        }
        contactable_graph = ContactableGraph({tuple(sorted([k1, k2])): True for k1, k2 in itertools.combinations(fixtures.keys(), 2)})
        print('contactable graph:', {tuple(sorted([k1, k2])) for k1, k2 in itertools.combinations(fixtures.keys(), 2) if contactable_graph.query_category(k1) & contactable_graph.query_mask(k2) and contactable_graph.query_category(k2) & contactable_graph.query_mask(k1)})
        body_def = {'ball1': BodyDef('ball1', ['shape0'], 'dynamic'), 'ball2': BodyDef('ball2', ['shape1'], 'dynamic')}
        print('body:', {bn: body.body_type for bn, body in body_def.items()})
        joint_def = dict()
        print('joint:', {jn: joint.joint_type for jn, joint in joint_def.items()})
        program = Box2DProgram(fixtures, contactable_graph, body_def, joint_def)
    else:
        raise ValueError(f'unknown dataset: {dataset}')
    return program
def predict(trajs_list, program, params, free_init=False, stride=1):
    pred_trajs_list = []
    for ti, trajs in enumerate(trajs_list):
        if free_init:
            pred_trajs, params, initial_state_params = program._simulate_from_partial_trajs(params, trajs[:4], FPS, len(trajs)-1, STRIDE=stride,)
        else:
            initial_state_params = program._get_initial_state_params(trajs[:4], FPS,)
            pred_trajs, params, initial_state_params = program._simulate(params, initial_state_params, trajs[0], FPS, len(trajs)-1, STRIDE=stride,)
        assert len(pred_trajs) == len(trajs)
        pred_trajs_list.append(pred_trajs)
    return pred_trajs_list, params
def evaluate(pred_trajs_list, trajs_list, gt_positions_list, split_list, verbose=False):
    if verbose:
        if len(trajs_list[0][0]) == 1:
            plot_trajs_diff(trajs_list[0], pred_trajs_list[0], gt_positions_list[0])
        elif len(trajs_list[0][0]) == 2:
            plot_trajs_diff_collision(trajs_list[0], pred_trajs_list[0], gt_positions_list[0])
        else:
            raise ValueError(f'unsupported object size: {len(trajs_list[0][0])}')
    metrics, vel_err = zip(*[_metrics_fn(tpt, ot, fps=FPS, gt_positions=ogt) for tpt, ot, ogt in zip(pred_trajs_list, trajs_list, gt_positions_list)])
    metrics = [{k: np.nanmean([v[k] for v in mm.values()]) for k in list(mm.values())[0].keys()} for mm in metrics]
    vel_err = list(vel_err)

    splits = sorted(set(split_list))
    splitted_metrics = {split: [metrics[i] for i, s in enumerate(split_list) if s == split] for split in splits}
    splitted_metrics['all'] = metrics
    if 'out' in ' '.join(splits):
        splitted_metrics['out'] = [metrics[i] for i, s in enumerate(split_list) if 'out' in s]
    splitted_vel_err = {split: [vel_err[i] for i, s in enumerate(split_list) if s == split] for split in splits}
    splitted_vel_err['all'] = vel_err
    if 'out' in ' '.join(splits):
        splitted_vel_err['out'] = [vel_err[i] for i, s in enumerate(split_list) if 'out' in s]
    pprint({split: len(metrics) for split, metrics in splitted_metrics.items()})

    metrics = {split: {k: np.nanmean([mm[k] for mm in metrics]) for k in metrics[0].keys()} for split, metrics in splitted_metrics.items()}
    vel_err = {split: {k: np.nanmean([v[k] for v in vel_err]) for k in vel_err[0].keys()} for split, vel_err in splitted_vel_err.items()}
    if verbose:
        pprint(metrics)
        pprint(vel_err)
    return {
        'metrics': metrics,
        'vel_err': vel_err,
    }

FPS = 10
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='collision')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--free-init', action='store_true', default=False)

    parser.add_argument('--loss_name', type=str, default='mae')
    parser.add_argument('--early_stop_threshold', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='Powell')
    parser.add_argument('--maxiter', type=int, default=int(1e7))
    parser.add_argument('--max_tries', type=int, default=10)

    args = parser.parse_args()

    train_dataset = Dataset(args.dataset, 'train', seed=0, novideo=True,)
    test_dataset = Dataset(args.dataset, 'test', seed=0, novideo=True,)

    optim_hyperparams = {
        'loss_name': args.loss_name,
        'early_stop_threshold': args.early_stop_threshold,
        'optimizer': args.optimizer,
        'maxiter': args.maxiter,
        'max_tries': args.max_tries,
    }

    curdir = osp.dirname(os.path.abspath(__file__))
    outdir = osp.join(curdir, 'results', osp.basename(__file__).replace('.py', ''))
    outname = osp.join(outdir, f'{args.dataset}_bs{args.batch_size}' + ('_freeinit' if args.free_init else '') + ('_stride%d' % args.stride if args.stride > 1 else '') + '.json')
    os.makedirs(outdir, exist_ok=True)

    train_data_list = [train_dataset[i] for i in range(args.batch_size)]
    trajs_list, gt_positions_list, split_list = zip(*([(data['trajs'], data['gt_positions'], data['split']) for data in train_data_list]))
    trajs_list = list(trajs_list)
    gt_positions_list = list(gt_positions_list)
    split_list = list(split_list)

    program = get_program(args.dataset)

    set_logger(outname.replace('.json', '.log'))
    if args.free_init:
        params = program.fit_all(trajs_list, FPS, verbose=True, set_params={}, hyperparams_list=[optim_hyperparams,], batched=True, STRIDE=args.stride,)
    else:
        params = program.fit(trajs_list, FPS, verbose=True, set_params={}, hyperparams_list=[optim_hyperparams,], batched=True, STRIDE=args.stride,)
    pred_trajs_list, params = predict(trajs_list, program, params, args.free_init, args.stride)
    pprint(params)
    train_metrics = evaluate(pred_trajs_list, trajs_list, gt_positions_list, split_list, verbose=False,)
    pprint(train_metrics)

    test_data_list = [test_dataset[i] for i in range(len(test_dataset))]
    print(f'Test data size: {len(test_data_list)}')
    trajs_list, gt_positions_list, split_list = zip(*([(data['trajs'], data['gt_positions'], data['split']) for data in test_data_list]))
    trajs_list = list(trajs_list)
    gt_positions_list = list(gt_positions_list)
    split_list = list(split_list)
    pred_trajs_list, _ = predict(trajs_list, program, params, args.free_init)
    test_metrics = evaluate(pred_trajs_list, trajs_list, gt_positions_list, split_list, verbose=False,)
    pprint(test_metrics)
    with open(outname, 'w') as f:
        json.dump({
            'params': params.dumps(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
        }, f, indent=2)

if __name__ == '__main__':
    main()
