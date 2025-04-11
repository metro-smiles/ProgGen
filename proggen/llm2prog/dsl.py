import itertools
import copy

from ..utils.eval_utils import time_limit, swallow_io, TimeoutException
from ..prog import BodyDef, JointDef, ContactableGraph

# Write documenation for the DSL here, including can_overlap, add_body, and
# add_joint. Describing their arguments, return values, and functionality.
DSLDoc = '''
can_overlap(s1, s2) - Declare that shapes s1 and s2 can overlap.
body = add_body(shapes, body_type) - Add a body with a list of shapes and a body type (static, dynamic, kinematic).
joint = add_joint(body1, body2, joint_type) - Add a joint between two bodies with a joint type (revolute, prismatic, distance, weld).
'''.strip()
DSLOneshotExample = '''
# This example code creates a simple environment with a static body and two dynamic bodies connected by a revolute joint. The static body, body1, has one shape, shape1. The first dynamic body, body2, has one shape, shape2. The second dynamic body, body3, has two shapes, shape3 and shape4. The two dynamic bodies are connected by a revolute joint, joint1. shape2 and shape3 can overlap, and shape2 and shape4 can overlap.
can_overlap('shape2', 'shape3')
can_overlap('shape2', 'shape4')
body1 = add_body(['shape1'], 'static')
body2 = add_body(['shape2'], 'dynamic')
body3 = add_body(['shape3', 'shape4'], 'dynamic')
joint1 = add_joint(body2, body3, 'revolute')
'''.strip()

can_overlap_cache = dict()
bodies = dict()
joints = dict()
def can_overlap(s1, s2):
    global can_overlap_cache
    can_overlap_cache[tuple(sorted([s1, s2]))] = True
def cannot_overlap(s1, s2):
    global can_overlap_cache
    can_overlap_cache[tuple(sorted([s1, s2]))] = False
def add_body(shapes, body_type):
    global bodies
    body_name = '_'.join(sorted(shapes))
    bodies[body_name] = BodyDef(body_name, shapes, body_type)
    return body_name
def add_joint(body1, body2, joint_type):
    global joints
    joint_name = '-'.join(sorted([body1, body2]))
    joints[joint_name] = JointDef(joint_name, [body1, body2], joint_type)
    return joint_name

def code2prog(code, fixture_names,):
    assert isinstance(code, str)

    global can_overlap_cache, bodies, joints
    can_overlap_cache = {tuple(sorted([fn1, fn2])): False for fn1, fn2 in itertools.combinations(fixture_names, 2)}
    bodies = dict()
    joints = dict()
    try:
        with time_limit(1):
            with swallow_io():
                exec(code,)
                contactable_graph = ContactableGraph({k: not v for k, v in can_overlap_cache.items()}) # Overlap -> Cannot contact
        return (contactable_graph, bodies, joints), None
    except TimeoutException as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)

