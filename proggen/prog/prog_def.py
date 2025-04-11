#!/usr/bin/env python
# coding=utf-8

import copy, numpy as np

from Box2D import b2World, b2Vec2, b2Transform
from Box2D import b2TestOverlap, b2PolygonShape, b2CircleShape, b2EdgeShape
from Box2D import b2FixtureDef, b2BodyDef, b2_dynamicBody, b2_staticBody, b2_kinematicBody

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class FixtureDef:
    def __init__(self, name, shape_name, bounded_friction_restitution=True,):
        self.name = name
        self.shape_name = shape_name
        self._kwargs = None
        self.bounded_friction_restitution = bounded_friction_restitution
    def init_def(self, params, categoryBits=None, maskBits=None):
        assert (categoryBits is None) == (maskBits is None), "categoryBits and maskBits must be both None or both not None"
        out = copy.deepcopy(self)
        out._kwargs = {
            'density': abs(params.get_params(f'{self.name}_fixture_density')),
            'friction': abs(params.get_params(f'{self.name}_fixture_friction')) if not self.bounded_friction_restitution else sigmoid(params.get_params(f'{self.name}_fixture_friction')),
            'restitution': abs(params.get_params(f'{self.name}_fixture_restitution')) if not self.bounded_friction_restitution else sigmoid(params.get_params(f'{self.name}_fixture_restitution')),
        }
        if categoryBits is not None:
            out._kwargs['categoryBits'] = categoryBits
        if maskBits is not None:
            out._kwargs['maskBits'] = maskBits
        return out
    def add_to_body(self, body, state, transform):
        _kwargs = copy.deepcopy(self._kwargs)
        assert state['shape'] == self.shape_name, (state['shape'], self.shape_name)
        if self.shape_name == 'circle':
            circle = b2CircleShape(radius=state['radius'], )
            circle.position = _exec_transform(transform, (0, 0))
            fixture_def = b2FixtureDef(**_kwargs, shape=circle)
            body.CreateFixture(fixture_def)
        elif self.shape_name == 'polygon':
            polygon = b2PolygonShape(vertices=[_exec_transform(transform, tuple(v)) for v in state['vertices']])
            fixture_def = b2FixtureDef(**_kwargs, shape=polygon)
            body.CreateFixture(fixture_def)
        else:
            vertices = [_exec_transform(transform, tuple(v)) for v in state['vertices']]
            for edge in zip(vertices, vertices[1:] + [vertices[0]]):
                shape = b2EdgeShape(vertices=edge)
                fixture_def = b2FixtureDef(**_kwargs, shape=shape)
                body.CreateFixture(fixture_def)
class ContactableGraph:
    def __init__(self, contactable):
        self.contactable = contactable
        fixture_names = sorted(set([fn for fns in self.contactable.keys() for fn in fns]))
        self.categoryBits = {fn: 1 << i for i, fn in enumerate(fixture_names)}
        self.maskBits = {
            fn: sum([1 << i for i, fn2 in enumerate(fixture_names) if fn != fn2 and self.contactable[tuple(sorted((fn, fn2)))]])
            for fn in fixture_names
        }
    def query_category(self, fixture_name):
        return self.categoryBits[fixture_name]
    def query_mask(self, fixture_name):
        return self.maskBits[fixture_name]
def _exec_transform(transform, v):
    pos_diff = transform['position_diff']
    angle_diff = transform['angle_diff']
    x, y = v
    return (
        x * np.cos(angle_diff) - y * np.sin(angle_diff) + pos_diff[0],
        x * np.sin(angle_diff) + y * np.cos(angle_diff) + pos_diff[1],
    )
def float_vec(vec):
    return (float(v) for v in vec)
BODYTYPE = {
    'dynamic': b2_dynamicBody,
    'static': b2_staticBody,
    'kinematic': b2_kinematicBody,
}
class BodyDef:
    def __init__(self, name, fixture_names, body_type,):
        self.name = name
        self.fixture_names = fixture_names
        self.body_type = body_type.lower()
    def init(self, params, world, fixture_definitions, initial_state):
        self.initial_fixure_positions = {fn: b2Vec2(*float_vec(initial_state[fn]['position'])) for fn in self.fixture_names}
        self.initial_fixture_velocities = {fn: b2Vec2(*float_vec(initial_state[fn]['velocity'])) for fn in self.fixture_names}
        self.initial_fixture_angles = {fn: float(initial_state[fn]['angle']) for fn in self.fixture_names}
        self.initial_fixture_angular_velocities = {fn: float(initial_state[fn]['angular_velocity']) for fn in self.fixture_names}
        #TODO: Actually, this is only a temporary solution. We need to handle the case from the perception module
        # assert len(set(self.initial_fixture_velocities)) == 1, "All fixtures in a body must have the same initial velocity"
        self.initial_body_velocity = list(self.initial_fixture_velocities.values())[0]
        # assert len(set(self.initial_fixture_angular_velocities)) == 1, "All fixtures in a body must have the same initial angular velocity"
        self.initial_body_angular_velocity = list(self.initial_fixture_angular_velocities.values())[0]
        if len(set(self.initial_fixure_positions.values())) == 1:
            self.initial_body_position = list(self.initial_fixure_positions.values())[0]
        else:
            self.initial_body_position = b2Vec2(0., 0.)
            for fn in self.fixture_names:
                self.initial_body_position += self.initial_fixure_positions[fn]
            self.initial_body_position /= len(self.fixture_names)
        if len(set(self.initial_fixture_angles.values())) == 1:
            self.initial_body_angle = list(self.initial_fixture_angles.values())[0]
        else:
            self.initial_body_angle = 0.
        self.fixture_local_transforms = dict()
        for fn in self.fixture_names:
            self.fixture_local_transforms[fn] = {
                'position_diff': self.initial_fixure_positions[fn] - self.initial_body_position,
                'angle_diff': self.initial_fixture_angles[fn] - self.initial_body_angle,
            }

        body_def = b2BodyDef(
            position=self.initial_body_position,
            angle=float(self.initial_body_angle),
            type=BODYTYPE[self.body_type],
            # angularDamping=0.01,
        )
        body = world.CreateBody(body_def)
        for fn in self.fixture_names:
            fixture_definitions[fn].add_to_body(body, initial_state[fn], self.fixture_local_transforms[fn])
        body.linearVelocity = self.initial_body_velocity
        body.angularVelocity = self.initial_body_angular_velocity
        return body
    def update_state(self, state, body):
        for fn in self.fixture_names:
            state[fn]['position'] = tuple(body.transform * self.fixture_local_transforms[fn]['position_diff'])
            state[fn]['angle'] = body.angle + self.fixture_local_transforms[fn]['angle_diff']
            state[fn]['velocity'] = body.linearVelocity
            state[fn]['angular_velocity'] = body.angularVelocity
class JointDef:
    def __init__(self, name, body_names, joint_type, set_gt_anchor_for_debug=False,):
        self.name = name
        self.body_names = body_names
        assert len(self.body_names) == 2, "Joint must connect two bodies"
        self.joint_type = joint_type
        self.set_gt_anchor_for_debug = set_gt_anchor_for_debug
    def init(self, params, world, bodies):
        body1 = bodies[self.body_names[0]]
        body2 = bodies[self.body_names[1]]
        if self.joint_type == 'revolute':
            joint = world.CreateRevoluteJoint(
                bodyA=body1,
                bodyB=body2,
                anchor=(
                    body1.transform * b2Vec2(*(params.get_params(f'revjoint_anchor_{self.name}', 2)))
                    if not self.set_gt_anchor_for_debug
                    else body1.position
                ),
            )
        elif self.joint_type == 'distance':
            joint = world.CreateDistanceJoint(
                bodyA=body1,
                bodyB=body2,
                anchorA=body1.position,
                anchorB=body2.position,
            )
        elif self.joint_type == 'prismatic':
            joint = world.CreatePrismaticJoint(
                bodyA=body1,
                bodyB=body2,
                anchor=(
                    body1.transform * b2Vec2(*(params.get_params(f'prismjoint_anchor_{self.name}', 2)))
                    if not self.set_gt_anchor_for_debug
                    else body1.position
                ),
                axis=(
                    b2Vec2(*(params.get_params(f'prismatic_axis_{self.name}', 2)))
                    if not self.set_gt_anchor_for_debug
                    else b2Vec2(1, 0)
                ),
            )
        elif self.joint_type == 'weld':
            joint = world.CreateWeldJoint(
                bodyA=body1,
                bodyB=body2,
                localAnchorA=body1.transform * b2Vec2(*(params.get_params(f'weldjoint_anchor_{self.name}', 2))),
                localAnchorB=body2.transform * b2Vec2(*(params.get_params(f'weldjoint_anchor_{self.name}', 2))),
            )
        else:
            raise ValueError("Unknown joint type: {}".format(self.joint_type))
        return joint


