#!/usr/bin/env python
# coding=utf-8

import copy
from dotmap import DotMap
import numpy as np
from scipy.optimize import minimize
from Box2D import b2World, b2Vec2, b2Transform

from .prog_def import sigmoid

RANDOM_PARAM_NUM = 20
class ContParams:
    def __init__(self, params, name2indices=None, index=None):
        self.params = params
        self.name2indices = dict() if name2indices is None else name2indices
        self.index = 0 if index is None else index
    def get_params(self, name, n=None):
        assert isinstance(name, str)
        if name in self.name2indices:
            if n is None:
                assert len(self.name2indices[name]) == 1, f"Name {name} is not unique"
                return self.params[self.name2indices[name][0]]
            else:
                assert len(self.name2indices[name]) == n, f"Name {name} is not unique"
                return [self.params[vi] for vi in self.name2indices[name]]
        if n is None:
            out = self.params[self.index]
            self.name2indices[name] = [self.index,]
            self.index += 1
        else:
            out = self.params[self.index:self.index+n]
            self.name2indices[name] = list(range(self.index, self.index+n))
            self.index += n
        return out
    def set_param(self, name, values):
        assert isinstance(name, str)
        assert name in self.name2indices, f"Name {name} not found"
        assert len(self.name2indices[name]) == len(values), f"Length mismatch: {len(self.name2indices[name])} != {len(values)}"
        for i, v in zip(self.name2indices[name], values):
            self.params[i] = v
    def set_params(self, params):
        for name, values in params.items():
            self.set_param(name, values)
    def insert_param(self, name, values):
        assert isinstance(name, str)
        assert name not in self.name2indices, f"Name {name} already exists"
        self.params[self.index:self.index+len(values)] = values
        self.name2indices[name] = list(range(self.index, self.index+len(values)))
        self.index += len(values)
    def get_indices(self, name):
        assert isinstance(name, str)
        assert name in self.name2indices, f"Name {name} not found"
        return tuple(self.name2indices[name])
    def __str__(self):
        return str({
            k: self._vis_param(k, self.params[v[0]]) if len(v) == 1 else [self.params[vi] for vi in v]
            for k, v in self.name2indices.items()
        })
    def _vis_param(self, name, param):
        # Assuming always bounded_friction_restitution
        if 'friction' in name or 'restitution' in name:
            return sigmoid(param)
        return param
    def __repr__(self):
        return str(self)
    def dumps(self,):
        return {
            name: (tuple(self.params[vi] for vi in v), tuple(v))
            for name, v in self.name2indices.items()
        }
    @staticmethod
    def loads(dump):
        param_num = max([i for _, (values, indices) in dump.items() for i in indices]) + 1
        out = ContParams(np.zeros(param_num))
        for name, (values, indices) in dump.items():
            out.name2indices[name] = list(indices)
            for i, v in zip(indices, values):
                out.params[i] = v
        return out

class Box2DProgram:
    def __init__(
        self,
        fixture_definitions,
        contactable_graph,
        body_definitions,
        joint_definitions,
        #TODO: Add action & reward related
    ):
        self.fixture_definitions = fixture_definitions
        self.contactable_graph = contactable_graph
        self.body_definitions = body_definitions
        self.joint_definitions = joint_definitions
    def _simulate(self, params, initial_state_params, initial_state, fps, max_time_steps, STRIDE=1, set_params=None, set_initial_state_params=None,):
        if not isinstance(params, ContParams):
            params = ContParams(params)
        if not isinstance(initial_state_params, ContParams):
            initial_state_params = ContParams(initial_state_params)
        if set_params is not None:
            params.set_params(set_params)
        if set_initial_state_params is not None:
            initial_state_params.set_params(set_initial_state_params)

        params = copy.deepcopy(params)
        initial_state_params = copy.deepcopy(initial_state_params)

        initial_state = copy.deepcopy(initial_state)
        for obj in sorted(initial_state):
            initial_state[obj]['position'] = tuple(initial_state_params.get_params(f'{obj}.position', n=2))
            initial_state[obj]['velocity'] = tuple(initial_state_params.get_params(f'{obj}.velocity', n=2))
            initial_state[obj]['angle'] = initial_state_params.get_params(f'{obj}.angle')
            initial_state[obj]['angular_velocity'] = initial_state_params.get_params(f'{obj}.angular_velocity')

        gravity = params.get_params('gravity', 2)
        world = b2World(gravity=b2Vec2(*gravity), doSleep=True)

        fixture_definitions = {
            fn: f.init_def(params, categoryBits=self.contactable_graph.query_category(fn), maskBits=self.contactable_graph.query_mask(fn))
            for fn, f in self.fixture_definitions.items()
        }
        bodies = {
            bn:body.init(params, world, fixture_definitions, initial_state,)
            for bn, body in self.body_definitions.items()
        }
        joints = {
            jn: joint.init(params, world, bodies)
            for jn, joint in self.joint_definitions.items()
        }

        TARGET_FPS = fps
        TIME_STEP = 1.0 / TARGET_FPS
        out_states = [initial_state]
        for t in range(max_time_steps):
            #TODO: Add action & reward related
            # world.Step(TIME_STEP, 10, 10)
            assert abs(STRIDE-int(STRIDE)) < 1e-5, STRIDE
            for _ in range(int(STRIDE)):
                world.Step(TIME_STEP/STRIDE, 15, 20)
            cur_state = copy.deepcopy(initial_state)
            for bn in bodies:
                self.body_definitions[bn].update_state(cur_state, bodies[bn],)
            out_states.append(cur_state)
            world.ClearForces()
        return out_states, params, initial_state_params
    def _simulate_from_partial_trajs(self, params, partial_trajs, fps, max_time_steps, STRIDE=1, set_params=None, set_initial_state_params=None, nofit=False, oracle_nofit=False,):
        assert len(partial_trajs) > 2, len(partial_trajs)
        nofit = nofit or oracle_nofit
        if not isinstance(params, ContParams):
            params = ContParams(params)
        if set_params is not None:
            params.set_params(set_params)
        params = copy.deepcopy(params)
        initial_state_params = self._get_initial_state_params(partial_trajs, fps, oracle_nofit=oracle_nofit,)
        if not nofit:
            initial_state_params = _fit(_loss_fn_on_initstateparams, (self, params, partial_trajs, fps, 'mae', STRIDE, set_params, set_initial_state_params,),
                                        max_tries=1, maxiter=int(100), early_stop_threshold=0.1, optimizer='Powell', initial_params=initial_state_params.params, verbose=False)
        return self._simulate(params, initial_state_params, partial_trajs[0], fps, max_time_steps, STRIDE=STRIDE, set_params=set_params, set_initial_state_params=set_initial_state_params,)
    def _get_initial_state_params(self, partial_trajs, fps, oracle_nofit=False,):
        initial_state_params = ContParams(np.random.rand(RANDOM_PARAM_NUM))
        for obj in sorted(partial_trajs[0]):
            if not oracle_nofit:
                initial_state_params.insert_param(f'{obj}.position', partial_trajs[0][obj]['position'],)
                initial_state_params.insert_param(
                    f'{obj}.velocity',
                    partial_trajs[0][obj]['velocity']
                    if 'velocity' in partial_trajs[0][obj] and partial_trajs[0][obj]['velocity'] is not None
                    else (np.array(partial_trajs[1][obj]['position']) - np.array(partial_trajs[0][obj]['position'])) * fps,
                    # else ((np.array(partial_trajs[1][obj]['position']) - np.array(partial_trajs[0][obj]['position']))[0] * fps, 0),
                )
                initial_state_params.insert_param(f'{obj}.angle', [partial_trajs[0][obj]['angle'],])
                initial_state_params.insert_param(
                    f'{obj}.angular_velocity',
                    partial_trajs[0][obj]['angular_velocity']
                    if 'angular_velocity' in partial_trajs[0][obj] and partial_trajs[0][obj]['angular_velocity'] is not None
                    else [(partial_trajs[1][obj]['angle'] - partial_trajs[0][obj]['angle']) * fps,],
                )
            else:
                initial_state_params.insert_param(f'{obj}.position', (partial_trajs[0][obj]['position'][0], np.mean([s[obj]['position'][1] for s in partial_trajs]),))
                initial_state_params.insert_param(
                    f'{obj}.velocity',
                    ((np.array(partial_trajs[2][obj]['position']) - np.array(partial_trajs[0][obj]['position']))[0] / 2 * fps, 0),
                )
                initial_state_params.insert_param(f'{obj}.angle', [partial_trajs[0][obj]['angle'],])
                initial_state_params.insert_param(
                    f'{obj}.angular_velocity',
                    partial_trajs[0][obj]['angular_velocity']
                    if 'angular_velocity' in partial_trajs[0][obj] and partial_trajs[0][obj]['angular_velocity'] is not None
                    else [(partial_trajs[1][obj]['angle'] - partial_trajs[0][obj]['angle']) * fps,],
                )

        return initial_state_params
    def fit(self, trajs, fps, hyperparams_list=({
        'max_tries': 100,
        'maxiter': int(1e5),
        'early_stop_threshold': 0.1,
        'loss_name': 'mse',
        'optimizer': 'Nelder-Mead',
    },), verbose=False, STRIDE=1, set_params=None, set_initial_state_params=None, nofit=True, oracle_nofit=False, batched=False, iter_rollout=False,):
        if not batched and not iter_rollout:
            cur_loss_fn = _loss_fn_on_params
        elif batched and not iter_rollout:
            cur_loss_fn = _loss_fn_on_params_batched
        else:
            raise NotImplementedError

        best_loss, best_params = 100, None
        for hyperparams in hyperparams_list:
            params = _fit(cur_loss_fn, (self, trajs, fps, hyperparams['loss_name'], STRIDE, set_params, set_initial_state_params, nofit, oracle_nofit), verbose=verbose, **{k: v for k, v in hyperparams.items() if k != 'loss_name'})
            loss = cur_loss_fn(params, self, trajs, fps, 'mse', nofit=nofit, oracle_nofit=oracle_nofit) # TODO: Set one unified loss function
            if loss < best_loss:
                best_loss = loss
                best_params = params
        return best_params
    def fit_all(self, trajs, fps, hyperparams_list=({
        'max_tries': 100,
        'maxiter': int(1e5),
        'early_stop_threshold': 0.1,
        'loss_name': 'mse',
        'optimizer': 'Nelder-Mead',
    },), verbose=False, STRIDE=1, set_params=None, set_initial_state_params=None, batched=False, iter_rollout=False,):
        best_loss, best_params = 100, None
        if not batched and not iter_rollout:
            cur_loss_fn = _loss_fn_on_all
        elif batched and not iter_rollout:
            cur_loss_fn = _loss_fn_on_all_batched
        else:
            raise NotImplementedError
        if not batched and not iter_rollout:
            initial_state_params = self._get_initial_state_params(trajs, fps,)
        elif batched and not iter_rollout:
            assert isinstance(trajs, list) and isinstance(trajs[0], list), trajs
            initial_state_params_list = [self._get_initial_state_params(traj, fps,) for traj in trajs]
            initial_state_params = ContParams(np.concatenate([isp.params for isp in initial_state_params_list]))
        else:
            raise NotImplementedError
        for hyperparams in hyperparams_list:
            initial_params = np.concatenate((np.random.rand(RANDOM_PARAM_NUM), initial_state_params.params))
            params = _fit(cur_loss_fn, (self, trajs, fps, hyperparams['loss_name'], STRIDE, set_params, set_initial_state_params), verbose=verbose, **{k: v for k, v in hyperparams.items() if k != 'loss_name'}, initial_params=initial_params)
            loss = cur_loss_fn(params, self, trajs, fps, 'mse') # TODO: Set one unified loss function
            if loss < best_loss:
                best_loss = loss
                best_params = params
        return best_params

def _metrics_per_array_fn(true_array, pred_array):
    assert len(true_array) == len(pred_array), (len(true_array), len(pred_array))
    if np.isnan(true_array).any() or np.isnan(pred_array).any():
        return {
            'mse': 1000,
            'mae': 1000,
            'r2': -1000,
        }
    true_array = np.array(true_array)
    pred_array = np.array(pred_array)
    mse = np.mean((true_array - pred_array) ** 2)
    mae = np.mean(np.abs(true_array - pred_array))
    r2 = 1 - np.mean((true_array - pred_array) ** 2) / (
        np.mean((true_array - np.mean(true_array)) ** 2)
        if np.mean((true_array - np.mean(true_array)) ** 2) > 1e-2
        else 1e-2
    )
    assert r2 <= 1, (true_array, pred_array, r2)
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
    }
def _metrics_fn(pred_trajs, trajs, gt_positions=None, fps=None,):
    assert len(pred_trajs) == len(trajs), (len(pred_trajs), len(trajs))

    true_array = {
        f'{fn}.position[{pi}]': [t[fn]['position'][pi] for t in trajs[1:]]
        for fn in trajs[0]
        for pi in range(2)
    }
    true_array.update({
        f'{fn}.angle': [t[fn]['angle'] for t in trajs[1:]]
        for fn in trajs[0]
    })

    pred_array = {
        f'{fn}.position[{pi}]': np.array([t[fn]['position'][pi] for t in pred_trajs[1:]])
        for fn in trajs[0]
        for pi in range(2)
    }
    pred_array.update({
        f'{fn}.angle': np.array([t[fn]['angle'] for t in pred_trajs[1:]])
        for fn in trajs[0]
    })

    metrics = {
        k: _metrics_per_array_fn(true_array[k], pred_array[k])
        for k in true_array
    }

    if gt_positions:
        from skimage.draw import disk
        def percept_position(obj):
            if obj['shape'] == 'circle':
                rr, cc = disk(np.array(obj['position'])*25.6, obj['radius']*25.6)
                return np.mean(rr)/25.6, np.mean(cc)/25.6
            else:
                raise NotImplementedError(f"Unknown shape: {obj['shape']}")
                return obj['position']
        pred_positions = np.array([[pred_trajs[t][fn]['position'] for fn in sorted(pred_trajs[0])] for t in range(0, len(pred_trajs))]).squeeze()
        percepted_positions = np.array([[trajs[t][fn]['position'] for fn in sorted(trajs[0])] for t in range(0, len(trajs))]).squeeze()
        percepted_pred_positions = np.array([[percept_position(pred_trajs[t][fn]) for fn in sorted(pred_trajs[0])] for t in range(0, len(pred_trajs))]).squeeze()
        gt_positions = np.array(gt_positions)
        assert pred_positions.shape == gt_positions.shape, (pred_positions.shape, gt_positions.shape)
        assert percepted_positions.shape == gt_positions.shape, (percepted_positions.shape, gt_positions.shape)
        assert percepted_pred_positions.shape == gt_positions.shape, (percepted_pred_positions.shape, gt_positions.shape)
        pred_vel = np.diff(pred_positions, axis=0) * fps
        percepted_vel = np.diff(percepted_positions, axis=0) * fps
        percepted_pred_vel = np.diff(percepted_pred_positions, axis=0) * fps
        gt_vel = np.diff(gt_positions, axis=0) * fps
        assert pred_vel.shape == gt_vel.shape, (pred_vel.shape, gt_vel.shape)
        assert percepted_vel.shape == gt_vel.shape, (percepted_vel.shape, gt_vel.shape)
        assert percepted_pred_vel.shape == gt_vel.shape, (percepted_pred_vel.shape, gt_vel.shape)
        vel_err_avg = np.mean(np.abs(pred_vel - gt_vel))
        percepted_vel_err_avg = np.mean(np.abs(percepted_vel - gt_vel))
        percepted_pred_vel_err_avg = np.mean(np.abs(percepted_pred_vel - gt_vel))

        gt_x_vel = gt_vel[..., 0]
        pred_x_vel = pred_vel[..., 0]
        percepted_x_vel = percepted_vel[..., 0]
        percepted_pred_x_vel = percepted_pred_vel[..., 0]

        vel_error = {
            'pred_vel_err_avg': vel_err_avg,
            'percepted_vel_err_avg': percepted_vel_err_avg,
            'percepted_pred_vel_err_avg': percepted_pred_vel_err_avg,
        }

        collision_index = len(gt_x_vel) - 1 # if no frames after collision
        if gt_positions.ndim == 2:
            vel_error.update({
                'pred_vel_error': np.abs(pred_x_vel[4:].mean() - gt_x_vel[4:].mean()),
                'percepted_vel_error': np.abs(percepted_x_vel[4:].mean() - gt_x_vel[4:].mean()),
                'percepted_pred_vel_error': np.abs(percepted_pred_x_vel[4:].mean() - gt_x_vel[4:].mean()),
            })
        elif gt_positions.ndim == 3:
            for i in range(1, len(gt_x_vel)):
                # if gt_x_pos[i, 0] - gt_x_pos[i-1, 0] <= 0 or
                # gt_x_pos[i, 1] - gt_x_pos[i-1, 1] >= 0:
                # if the value of gt_x_vel changes greatly
                if np.abs(gt_x_vel[i, 0] - gt_x_vel[i-1, 0]) > 0.1 or np.abs(gt_x_vel[i, 1] - gt_x_vel[i-1, 1]) > 0.1:
                    collision_index = i
                    break

            pred_pre_vel_error = np.abs(pred_x_vel[:collision_index].mean(-2) - gt_x_vel[:collision_index].mean(-2)).mean()
            pred_post_vel_error = np.abs(pred_x_vel[collision_index+1:].mean(-2) - gt_x_vel[collision_index+1:].mean(-2)).mean()
            pred_pre_vel_err_avg = np.mean(np.abs(pred_x_vel[:collision_index] - gt_x_vel[:collision_index]))
            pred_post_vel_err_avg = np.mean(np.abs(pred_x_vel[collision_index+1:] - gt_x_vel[collision_index+1:]))
            percepted_pre_vel_error = np.abs(percepted_x_vel[:collision_index].mean(-2) - gt_x_vel[:collision_index].mean(-2)).mean()
            percepted_post_vel_error = np.abs(percepted_x_vel[collision_index+1:].mean(-2) - gt_x_vel[collision_index+1:].mean(-2)).mean()
            percepted_pre_vel_err_avg = np.mean(np.abs(percepted_x_vel[:collision_index] - gt_x_vel[:collision_index]))
            percepted_post_vel_err_avg = np.mean(np.abs(percepted_x_vel[collision_index+1:] - gt_x_vel[collision_index+1:]))
            percepted_pred_pre_vel_error = np.abs(percepted_pred_x_vel[:collision_index].mean(-2) - gt_x_vel[:collision_index].mean(-2)).mean()
            percepted_pred_post_vel_error = np.abs(percepted_pred_x_vel[collision_index+1:].mean(-2) - gt_x_vel[collision_index+1:].mean(-2)).mean()
            percepted_pred_pre_vel_err_avg = np.mean(np.abs(percepted_pred_x_vel[:collision_index] - gt_x_vel[:collision_index]))
            percepted_pred_post_vel_err_avg = np.mean(np.abs(percepted_pred_x_vel[collision_index+1:] - gt_x_vel[collision_index+1:]))

            vel_error.update({
                'pred_pre_vel_error': pred_pre_vel_error,
                'pred_post_vel_error': pred_post_vel_error,
                'pred_pre_vel_err_avg': pred_pre_vel_err_avg,
                'pred_post_vel_err_avg': pred_post_vel_err_avg,
                'percepted_pre_vel_error': percepted_pre_vel_error,
                'percepted_post_vel_error': percepted_post_vel_error,
                'percepted_pre_vel_err_avg': percepted_pre_vel_err_avg,
                'percepted_post_vel_err_avg': percepted_post_vel_err_avg,
                'percepted_pred_pre_vel_error': percepted_pred_pre_vel_error,
                'percepted_pred_post_vel_error': percepted_pred_post_vel_error,
                'percepted_pred_pre_vel_err_avg': percepted_pred_pre_vel_err_avg,
                'percepted_pred_post_vel_err_avg': percepted_pred_post_vel_err_avg
            })
        else:
            raise ValueError(f"Unknown gt_positions shape: {gt_positions.shape}")


        # from _render import PygameRender
        # render = PygameRender(fps, 256, 256, 256/10.)
        # for obs in pred_trajs:
            # render.render(obs)
        # render.close()
        # assert False

        return metrics, vel_error

    #TODO: Add vel error
    # raise NotImplementedError

    return metrics

def _decode_loss(metrics, loss_name):
    if loss_name == 'mse':
        loss = np.mean([v['mse'] for k, v in metrics.items()])
    elif loss_name == 'mae':
        loss = np.mean([v['mae'] for k, v in metrics.items()])
    elif loss_name == 'rmse':
        loss = np.mean([np.sqrt(v['mse']) for k, v in metrics.items()])
    elif loss_name == 'r2':
        loss = -np.mean([v['r2'] for k, v in metrics.items()])
    elif loss_name == 'areaweighted_mse':
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")
    return loss
def _fit(loss_fn, args, max_tries=100, maxiter=int(1e5), early_stop_threshold=0.1, optimizer='Nelder-Mead', verbose=False, initial_params=None,):
    if verbose:
        print('~'*50)
        print(f"Optimizing {loss_fn.__name__.upper()}: optimizer={optimizer}, early_stop_threshold={early_stop_threshold}, max_tries={max_tries}, maxiter={maxiter}")
    lowest_loss, best_res, best_params = 1e10, None, None
    for ti in range(max_tries):
        if verbose:
            print(f"Try {ti+1}/{max_tries}")
        params = np.random.rand(RANDOM_PARAM_NUM) if initial_params is None else copy.deepcopy(initial_params)
        res = minimize(loss_fn, params, args=args, method=optimizer, options={'maxiter': maxiter, 'disp': verbose})
        if verbose:
            print(res)
        if res.fun < lowest_loss:
            lowest_loss = res.fun
            best_res = res
            best_params = res.x
        if res.fun < early_stop_threshold:
            assert lowest_loss < early_stop_threshold, (lowest_loss, early_stop_threshold)
            assert best_params is not None, "Best params is None"
            break;
    assert best_params is not None, "Best params is None"
    if verbose:
        print(best_res)
        print(f"Best loss: {lowest_loss}")
        print(f"Best params: {best_params}")
    return best_params

LOSS_ITER_INITSTATEPARAMS = 0
def _loss_fn_on_initstateparams(initial_state_params, program, params, trajs, fps, loss_name, STRIDE=1, set_params=None, set_initial_state_params=None,):
    try:
        pred_trajs, params, initial_state_params = program._simulate(params, initial_state_params, trajs[0], fps, len(trajs)-1, STRIDE=STRIDE, set_params=set_params, set_initial_state_params=set_initial_state_params,)
    except Exception as e:
        raise e
        return 2000
    metrics = _metrics_fn(pred_trajs, trajs,)
    loss = _decode_loss(metrics, loss_name)
    global LOSS_ITER_INITSTATEPARAMS
    if LOSS_ITER_INITSTATEPARAMS % 1000 == 0:
        print(f"\t InitState Iter {LOSS_ITER_INITSTATEPARAMS}: Loss={loss}")
    LOSS_ITER_INITSTATEPARAMS += 1
    return loss
LOSS_ITER_PARAMS = 0
def _loss_fn_on_params(params, program, trajs, fps, loss_name, STRIDE=1, set_params=None, set_initial_state_params=None, nofit=False, oracle_nofit=False,):
    try:
        pred_trajs = program._simulate_from_partial_trajs(params, trajs, fps, len(trajs)-1, STRIDE=STRIDE, set_params=set_params, set_initial_state_params=set_initial_state_params, nofit=nofit, oracle_nofit=oracle_nofit)[0]
    except Exception as e:
        raise e
        return 2000
    metrics = _metrics_fn(pred_trajs, trajs,)
    loss = _decode_loss(metrics, loss_name)
    global LOSS_ITER_PARAMS
    if LOSS_ITER_PARAMS % 1000 == 0:
        print(f"LossParams Iter {LOSS_ITER_PARAMS}: Loss={loss}")
    LOSS_ITER_PARAMS += 1
    return loss
LOSS_ITER_PARAMS_BATCHED = 0
def _loss_fn_on_params_batched(params, program, trajs_list, fps, loss_name, STRIDE=1, set_params=None, set_initial_state_params=None, nofit=False, oracle_nofit=False,):
    assert isinstance(trajs_list, list) and isinstance(trajs_list[0], list), trajs_list
    try:
        pred_trajs_list = [program._simulate_from_partial_trajs(params, trajs, fps, len(trajs)-1, STRIDE=STRIDE, set_params=set_params, set_initial_state_params=set_initial_state_params, nofit=nofit, oracle_nofit=oracle_nofit)[0] for trajs in trajs_list]
    except Exception as e:
        raise e
        return 2000
    metrics_list = [_metrics_fn(pred_trajs_list, trajs,) for pred_trajs_list, trajs in zip(pred_trajs_list, trajs_list)]
    loss_list = [_decode_loss(metrics, loss_name) for metrics in metrics_list]
    loss = np.mean(loss_list)
    global LOSS_ITER_PARAMS_BATCHED
    if LOSS_ITER_PARAMS_BATCHED % 1000 == 0:
        print(f"LossParams_BATCHED Iter {LOSS_ITER_PARAMS_BATCHED}: Loss={loss}, LossList={loss_list[:5]}")
    LOSS_ITER_PARAMS_BATCHED += 1
    return loss
LOSS_ITER_ALL = 0
def _loss_fn_on_all(params, program, trajs, fps, loss_name, STRIDE=1, set_params=None, set_initial_state_params=None,):
    params, initial_state_params = params[:len(params)//2], params[len(params)//2:]
    try:
        pred_trajs = program._simulate(params, initial_state_params, trajs[0], fps, len(trajs)-1, STRIDE=STRIDE, set_params=set_params, set_initial_state_params=set_initial_state_params,)[0]
    except Exception as e:
        raise e
        return 2000
    metrics = _metrics_fn(pred_trajs, trajs,)
    loss = _decode_loss(metrics, loss_name)
    global LOSS_ITER_ALL
    if LOSS_ITER_ALL % 1000 == 0:
        print(f"LossAll Iter {LOSS_ITER_ALL}: Loss={loss}")
    LOSS_ITER_ALL += 1
    return loss
LOSS_ITER_ALL_BATCHED = 0
def _loss_fn_on_all_batched(params, program, trajs_list, fps, loss_name, STRIDE=1, set_params=None, set_initial_state_params=None,):
    assert isinstance(trajs_list, list) and isinstance(trajs_list[0], list), trajs_list
    _feat_num = len(params) // (len(trajs_list) + 1)
    initial_state_params_list = [params[(i+1)*_feat_num:(i+2)*_feat_num] for i in range(len(trajs_list))]
    params = params[:_feat_num]
    try:
        pred_trajs_list = [program._simulate(params, initial_state_params, trajs[0], fps, len(trajs)-1, STRIDE=STRIDE, set_params=set_params, set_initial_state_params=set_initial_state_params,)[0] for trajs, initial_state_params in zip(trajs_list, initial_state_params_list)]
    except Exception as e:
        raise e
        return 2000
    metrics_list = [_metrics_fn(pred_trajs, trajs,) for pred_trajs, trajs in zip(pred_trajs_list, trajs_list)]
    loss_list = [_decode_loss(metrics, loss_name) for metrics in metrics_list]
    loss = np.mean(loss_list)
    global LOSS_ITER_ALL_BATCHED
    if LOSS_ITER_ALL_BATCHED % 1000 == 0:
        print(f"LossAllBatched Iter {LOSS_ITER_ALL_BATCHED}: Loss={loss}, LossList={loss_list[:5]}")
    LOSS_ITER_ALL_BATCHED += 1
    return loss
