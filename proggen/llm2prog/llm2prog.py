#!/usr/bin/env python
# coding=utf-8

from dotmap import DotMap
from pprint import pprint
import itertools

import numpy as np
import base64
from matplotlib import colormaps
import cv2
import PIL.Image as Image

from .dsl import DSLDoc, DSLOneshotExample
from ..utils.llm import LLM

class LLM2Prog:
    def __init__(self, template, llm_hyperparameters = {
        'model': 'gpt-4o',
        'temperature': 1.0,
    }, seed=0,):
        self.llm = LLM(default_args=llm_hyperparameters, seed=seed)
        self.template = template
        assert callable(self.template)
        self.rng = np.random.default_rng(seed)

    def __call__(self, trajs_as_text, trajs_as_video, PPM, verbose=True,):
        assert isinstance(trajs_as_text, list)
        assert isinstance(trajs_as_video, np.ndarray)
        assert isinstance(PPM, int) or isinstance(PPM, float)
        assert len(trajs_as_text) == trajs_as_video.shape[0]
        prompt = self.template(trajs_as_text, trajs_as_video, self.rng, PPM)
        with self.llm.track('LLM2Prog') as tracker:
            generated = self.llm([
                {'role': 'system', 'content': SYSTEM_MESSAGE,},
                {'role': 'user', 'content': prompt,},
            ]).choices[0].message.content
        # pprint(generated)
        # print(tracker)
        # assert False
        return self.parse_code(generated, verbose=verbose,)
    def parse_code(self, generated, verbose=False,):
        if verbose:
            print(generated)
            print('~'*80)
        code = ''
        IN_CODE_BLOCK = False
        for li, line in enumerate(generated.split('\n')):
            if line.startswith('```'):
                IN_CODE_BLOCK = not IN_CODE_BLOCK
                if IN_CODE_BLOCK:
                    assert not code, 'Multiple code blocks are not supported'
                continue
            if IN_CODE_BLOCK:
                code += line + '\n'
        assert code, 'No code block found'
        return code.strip()

SYSTEM_MESSAGE = '''
Your goal is to write a program to simulate the environment that the agent is interacting with. Specially, we know that the environment is a 2D world following physical laws, so we wrap up the low-level physical laws into a simulator. Your task is to set up the simulator with the right constraints such as which shapes belong to the same body, if two shapes can overlap with each other or not (can_overlap means they do not collide), and if two bodies are connected by a joint. The simulator will then handle all the details including the shape shapes, positions, angle, and so on. You should ONLY focus on the high-level constraints. The simulator will handle the rest.
'''

def rotate_vertex(v, angle):
    x, y = v
    return x*np.cos(angle) - y*np.sin(angle), x*np.sin(angle) + y*np.cos(angle)
def get_shape_names(trajs_as_text):
    shapes = set(fn for s in trajs_as_text for fn in s.keys())
    return sorted(shapes)
def text_box_positions(text_box_params, config, IMG_SIZE):
    _, (min_x, min_y, max_x, max_y), (text_width, text_height) = text_box_params
    hor, ver = config
    IMG_WIDTH, IMG_HEIGHT = IMG_SIZE
    OFFSET = 10
    if hor == -1:
        text_box_min_x = min_x - text_width
        text_box_max_x = min_x
    elif hor == 1:
        text_box_min_x = max_x
        text_box_max_x = max_x + text_width
    else:
        center_x = (min_x + max_x) // 2
        text_box_min_x = center_x - text_width // 2
        text_box_max_x = center_x + text_width // 2
    if ver == -1:
        text_box_min_y = min_y - text_height - OFFSET
        text_box_max_y = min_y - OFFSET
    elif ver == 1:
        text_box_min_y = max_y - OFFSET
        text_box_max_y = max_y + text_height - OFFSET
    else:
        center_y = (min_y + max_y) // 2
        text_box_min_y = center_y - text_height // 2 - OFFSET
        text_box_max_y = center_y + text_height // 2 - OFFSET
    if text_box_min_x < 0:
        text_box_min_x = 0
        text_box_max_x = text_width
    if text_box_max_x > IMG_WIDTH:
        text_box_min_x = IMG_WIDTH - text_width
        text_box_max_x = IMG_WIDTH
    if text_box_min_y < 0:
        text_box_min_y = 0
        text_box_max_y = text_height
    if text_box_max_y > IMG_HEIGHT:
        text_box_min_y = IMG_HEIGHT - text_height
        text_box_max_y = IMG_HEIGHT
    return text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y
def annotate_image_boundingbox(image, state, shape_names, PPM):
    CMAP = colormaps['gist_rainbow']
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    IMG_HEIGHT, IMG_WIDTH, _ = image.shape
    IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
    fontFace, fontScale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1., 2
    rectangles = []
    notations = []
    for i, (shape_name, shape) in enumerate(state.items()):
        x, y = shape['position']
        if 'vertices' in shape:
            vertices = shape['vertices']
            vertices = [rotate_vertex(v, shape['angle']) for v in vertices]
            min_x, min_y = np.min(vertices, axis=0)
            max_x, max_y = np.max(vertices, axis=0)
            min_x, max_x = min_x + x, max_x + x
            min_y, max_y = min_y + y, max_y + y
        else:
            assert 'radius' in shape, shape
            min_x = x - shape['radius']
            max_x = x + shape['radius']
            min_y = y - shape['radius']
            max_y = y + shape['radius']
        min_x, max_x, min_y, max_y = [int(v*PPM) for v in [min_x, max_x, min_y, max_y]]
        min_y, max_y = IMG_HEIGHT - max_y, IMG_HEIGHT - min_y
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), np.array(CMAP(i / len(state)))*255, 2)

        text_size = cv2.getTextSize(shape_names[shape_name], fontFace, fontScale, thickness)[0]
        rectangles.append((min_x, min_y, max_x, max_y))
        notations.append((shape_names[shape_name], (min_x, min_y, max_x, max_y), text_size))

    smallest_overlap, smallest_overlap_config = float('inf'), None
    for text_box_configs in itertools.product(itertools.product([-1, 0, 1], repeat=2), repeat=len(notations)):
        overlap_area = 0
        text_boxes = []
        for text_box, config in zip(notations, text_box_configs):
            text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y = text_box_positions(text_box, config, IMG_SIZE)
            text_boxes.append((text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y))
        for i, text_box in enumerate(text_boxes):
            text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y = text_box
            for j, box in enumerate(rectangles):
                if i == j:
                    continue
                min_x, min_y, max_x, max_y = box
                overlap_area += max(0, min(max_x, text_box_max_x) - max(min_x, text_box_min_x)) * max(0, min(max_y, text_box_max_y) - max(min_y, text_box_min_y))
            for j, other_text_box in enumerate(text_boxes):
                if i == j:
                    continue
                other_text_box_min_x, other_text_box_max_x, other_text_box_min_y, other_text_box_max_y = other_text_box
                overlap_area += max(0, min(text_box_max_x, other_text_box_max_x) - max(text_box_min_x, other_text_box_min_x)) * max(0, min(text_box_max_y, other_text_box_max_y) - max(text_box_min_y, other_text_box_min_y))
        if overlap_area < smallest_overlap:
            smallest_overlap, smallest_overlap_config = overlap_area, text_box_configs
            if overlap_area < 1:
                # assert False
                break

    for i, (text_box, config) in enumerate(zip(notations, smallest_overlap_config)):
        shape_name = text_box[0]
        text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y = text_box_positions(text_box, config, IMG_SIZE)
        cv2.putText(image, shape_name, (text_box_min_x, text_box_max_y), fontFace, fontScale, np.array(CMAP(i / len(state)))*255, thickness)
        # cv2.rectangle(image, (text_box_min_x, text_box_min_y), (text_box_max_x, text_box_max_y), (0, 255, 0), 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def annotate_image_contour(image, state, shape_names, PPM):
    CMAP = colormaps['gist_rainbow']
    image = image.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    IMG_HEIGHT, IMG_WIDTH, _ = image.shape
    IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
    fontFace, fontScale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1., 2
    rectangles = []
    notations = []
    # for i, (shape_name, shape) in enumerate(state.items()):
    for i, shape_name in enumerate(shape_names):
        shape = state[shape_name]
        x, y = shape['position']
        if 'vertices' in shape:
            vertices = shape['vertices']
            vertices = [rotate_vertex(v, shape['angle']) for v in vertices]
            vertices = [(v[0] + x, v[1] + y) for v in vertices]
            vertices = [(int(v[0]*PPM), IMG_HEIGHT - int(v[1]*PPM)) for v in vertices]
            cv2.polylines(image, [np.array(vertices)], True, np.array(CMAP(i / len(state)))*255, 2)
            min_x, min_y = np.min(vertices, axis=0)
            max_x, max_y = np.max(vertices, axis=0)
        else:
            assert 'radius' in shape, shape
            center = (int(x*PPM), IMG_HEIGHT - int(y*PPM))
            radius = int(shape['radius']*PPM)
            cv2.circle(image, center, radius, np.array(CMAP(i / len(state)))*255, 2)
            min_x, min_y = center[0] - radius, center[1] - radius
            max_x, max_y = center[0] + radius, center[1] + radius
        text_size = cv2.getTextSize(shape_names[shape_name], fontFace, fontScale, thickness)[0]
        rectangles.append((min_x, min_y, max_x, max_y))
        notations.append((shape_names[shape_name], (min_x, min_y, max_x, max_y), text_size))

    smallest_overlap, smallest_overlap_config = float('inf'), None
    for text_box_configs in itertools.product(itertools.product([-1, 0, 1], repeat=2), repeat=len(notations)):
        overlap_area = 0
        text_boxes = []
        for text_box, config in zip(notations, text_box_configs):
            text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y = text_box_positions(text_box, config, IMG_SIZE)
            text_boxes.append((text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y))
        for i, text_box in enumerate(text_boxes):
            text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y = text_box
            for j, box in enumerate(rectangles):
                if i == j:
                    continue
                min_x, min_y, max_x, max_y = box
                overlap_area += max(0, min(max_x, text_box_max_x) - max(min_x, text_box_min_x)) * max(0, min(max_y, text_box_max_y) - max(min_y, text_box_min_y))
            for j, other_text_box in enumerate(text_boxes):
                if i == j:
                    continue
                other_text_box_min_x, other_text_box_max_x, other_text_box_min_y, other_text_box_max_y = other_text_box
                overlap_area += max(0, min(text_box_max_x, other_text_box_max_x) - max(text_box_min_x, other_text_box_min_x)) * max(0, min(text_box_max_y, other_text_box_max_y) - max(text_box_min_y, other_text_box_min_y))
        if overlap_area < smallest_overlap:
            smallest_overlap, smallest_overlap_config = overlap_area, text_box_configs
            if overlap_area < 1:
                # assert False
                break

    for i, (text_box, config) in enumerate(zip(notations, smallest_overlap_config)):
        shape_name = text_box[0]
        text_box_min_x, text_box_max_x, text_box_min_y, text_box_max_y = text_box_positions(text_box, config, IMG_SIZE)
        cv2.putText(image, shape_name, (text_box_min_x, text_box_max_y), fontFace, fontScale, np.array(CMAP(i / len(state)))*255, thickness)
        # cv2.rectangle(image, (text_box_min_x, text_box_min_y), (text_box_max_x, text_box_max_y), (0, 255, 0), 2)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
def build_trajs_image(trajs_as_video, rng):
    FRAME_NUM = 5

    # FRAME_INTERVAL = 5
    # if len(trajs_as_video) <= FRAME_NUM * FRAME_INTERVAL:
        # start_frame_index = 0
    # else:
        # start_frame_index = rng.integers(len(trajs_as_video)-FRAME_NUM*FRAME_INTERVAL)
    # images = [trajs_as_video[fi] for fi in range(start_frame_index, start_frame_index+FRAME_NUM*FRAME_INTERVAL, FRAME_INTERVAL)]

    indexes = [rng.integers(len(trajs_as_video))]
    images = trajs_as_video[indexes[-1]][None]
    while len(indexes) < FRAME_NUM:
        distances = [np.linalg.norm(trajs_as_video[fi][None] - images) if fi not in indexes else 0 for fi in range(len(trajs_as_video))]
        indexes.append(np.argmax(distances))
        images = np.concatenate([images, trajs_as_video[indexes[-1]][None]], axis=0)
    images = [trajs_as_video[fi] for fi in sorted(indexes)]

    # Add alpha channel
    images = [cv2.cvtColor(image, cv2.COLOR_RGB2RGBA) for image in images]
    # Assign lower transparency to earlier images
    for i, image in enumerate(images):
        image[:, :, 3] = 255 - 50*i
    # Blend images
    pil_images = [Image.fromarray(image) for image in images]
    trajs_image = Image.alpha_composite(pil_images[0], pil_images[1])
    for pil_image in pil_images[2:]:
        trajs_image = Image.alpha_composite(trajs_image, pil_image)
    # Convert to numpy array
    trajs_image = np.array(trajs_image)
    return trajs_image
def template_OriginalScreenshot_AnnotatedContour_WithoutSemanticLabels_StackVideo(trajs_as_text, trajs_as_video, rng, PPM):
    shapes = get_shape_names(trajs_as_text)
    anonymous_shapes = [f'shape{i}' for i in range(len(shapes))]
    frame_index = rng.integers(len(trajs_as_text))
    # annotated_screenshot = annotate_image_boundingbox(trajs_as_video[frame_index], trajs_as_text[frame_index], {k:v for k,v in zip(shapes, anonymous_shapes)}, PPM)
    annotated_screenshot = annotate_image_contour(trajs_as_video[frame_index], trajs_as_text[frame_index], {k:v for k,v in zip(shapes, anonymous_shapes)}, PPM)
    import matplotlib.pyplot as plt
    plt.imshow(annotated_screenshot)
    plt.show()
    # assert False
    annotated_screenshot = base64.b64encode(cv2.imencode('.png', annotated_screenshot)[1]).decode('utf-8')
    # trajs_as_video = [annotate_image_contour(trajs_as_video[fi], trajs_as_text[fi], {k:v for k,v in zip(shapes, anonymous_shapes)}, PPM) for fi in range(len(trajs_as_video))]
    trajs_image = build_trajs_image(trajs_as_video, rng)
    import matplotlib.pyplot as plt
    plt.imshow(trajs_image)
    plt.show()
    # assert False
    trajs_image = base64.b64encode(cv2.imencode('.png', trajs_image)[1]).decode('utf-8')
    return [
        {'type': 'text', 'text': f'''
The agent is interacting with a 2D world. As shown in the following the screenshot, it has the following shapes: {anonymous_shapes}.
'''.strip()},
        {'type': 'image_url',
         'image_url': {'url': f'data:image/png;base64,{annotated_screenshot}'}},
        {'type': 'text', 'text': f'''
Your job is to write a program to simulate the environment that the agent is interacting with. We provide you with the following domain-specific language (DSL) to specify the environment. You can use the DSL to describe the environment in a high-level way, and the simulator will handle the low-level details. Here is the description of the DSL:

"""

{DSLDoc}

"""

Here are explanations of the notations:

"""

Shape: A shape is a convex shape in the environment.
Body: A body is a collection of shapes. Similarly to Box2D, all shapes belonging to the same body are always relative static to each other. Note that shapes connected by a joint are NOT considered to be in the same body since they can move relative to each other. The shapes connected by a joint should be defined in the different bodies, separately.
Body type: A body can be static, dynamic, or kinematic. A static body does not move. A dynamic body can move freely. A kinematic body can move but not by the physics engine.
Joint: A joint is a connection between two bodies. A joint can be a revolute joint, a prismatic joint, or a distance joint.
Joint type: A joint can be a revolute joint, a prismatic joint, or a distance joint. A revolute joint allows two bodies to rotate around a point. A prismatic joint allows two bodies to slide along an axis. A distance joint keeps two bodies at a fixed distance.

"""

Here is an example of how to use the DSL to specify the environment:

```

{DSLOneshotExample}

```

We further show an image to demonstrate the dynamics of shapes in the environment. This image is composed by a few screenshots of the environment. It shows the movement of shapes while the agent is interacting with the environment. The screenshots are taken sequentially. The lower the transparencies of the images, the earlier the screenshots are taken. The higher the transparencies of the images, the later the screenshots are taken. We compose the screenshots given their transparencies. The final image is shown below.
'''.strip()},
        {'type': 'image_url',
         'image_url': {'url': f'data:image/png;base64,{trajs_image}'}},
         {'type': 'text', 'text': f'''

Please write a program to simulate the environment based on the screenshot and the composed image that shows the dynamics of the environment. You can analyze the environment given the provided screenshot first. What is the environment? What dynamics are encoded in this environment? What do you expect to happen in this environment after this screenshot? Will the shapes move, in which way and which direction? To simulate such an environment, what components are needed? How many shapes are there in the environment? What are they representing in the environment? Are any of the shapes belonging to the same body (meaning that they will always be relative static to each other)? How many bodies are there in the environment? Are they static, dynamic, or kinematic? Are there joints connecting bodies? How many joints are there? Are they revolute joints, prismatic joints, or distance joints? If defined as specified, what will happen in the simulated environment? Will it behave the same as the true environment as shown in the screenshot? If not, please revise the program (setups of the simulator) accordingly. You can use the DSL to specify the environment in a high-level way, and the simulator will handle the low-level details. Please surround your code with triple backticks (```) so that it is formatted correctly.
'''.strip()},
    ]


def template_OriginalScreenshot_AnnotatedContour_WithoutSemanticLabels(trajs_as_text, trajs_as_video, rng, PPM):
    shapes = get_shape_names(trajs_as_text)
    anonymous_shapes = [f'shape{i}' for i in range(len(shapes))]
    frame_index = rng.integers(len(trajs_as_text))
    # annotated_screenshot = annotate_image_boundingbox(trajs_as_video[frame_index], trajs_as_text[frame_index], {k:v for k,v in zip(shapes, anonymous_shapes)}, PPM)
    annotated_screenshot = annotate_image_contour(trajs_as_video[frame_index], trajs_as_text[frame_index], {k:v for k,v in zip(shapes, anonymous_shapes)}, PPM)
    import matplotlib.pyplot as plt
    plt.imshow(annotated_screenshot)
    plt.show()
    # assert False
    annotated_screenshot = base64.b64encode(cv2.imencode('.png', annotated_screenshot)[1]).decode('utf-8')
    return [
        {'type': 'text', 'text': f'''
The agent is interacting with a 2D world. As shown in the following the screenshot, it has the following shapes: {anonymous_shapes}.
'''.strip()},
        {'type': 'image_url',
         'image_url': {'url': f'data:image/png;base64,{annotated_screenshot}'}},
        {'type': 'text', 'text': f'''
Your job is to write a program to simulate the environment that the agent is interacting with. We provide you with the following domain-specific language (DSL) to specify the environment. You can use the DSL to describe the environment in a high-level way, and the simulator will handle the low-level details. Here is the description of the DSL:

"""

{DSLDoc}

"""

Here are explanations of the notations:

"""

Shape: A shape is a convex shape in the environment.
Body: A body is a collection of shapes. Similarly to Box2D, all shapes belonging to the same body are always relative static to each other. Note that shapes connected by a joint are NOT considered to be in the same body since they can move relative to each other. The shapes connected by a joint should be defined in the different bodies, separately.
Body type: A body can be static, dynamic, or kinematic. A static body does not move. A dynamic body can move freely. A kinematic body can move but not by the physics engine.
Joint: A joint is a connection between two bodies. A joint can be a revolute joint, a prismatic joint, or a distance joint.
Joint type: A joint can be a revolute joint, a prismatic joint, or a distance joint. A revolute joint allows two bodies to rotate around a point. A prismatic joint allows two bodies to slide along an axis. A distance joint keeps two bodies at a fixed distance.

"""

Here is an example of how to use the DSL to specify the environment:

```

{DSLOneshotExample}

```

Please write a program to simulate the environment based on the screenshot above. You can analyze the environment given the provided screenshot first. What is the environment? What dynamics are encoded in this environment? What do you expect to happen in this environment after this screenshot? Will the shapes move, in which way and which direction? To simulate such an environment, what components are needed? How many shapes are there in the environment? What are they representing in the environment? Are any of the shapes belonging to the same body (meaning that they will always be relative static to each other)? How many bodies are there in the environment? Are they static, dynamic, or kinematic? Are there joints connecting bodies? How many joints are there? Are they revolute joints, prismatic joints, or distance joints? If defined as specified, what will happen in the simulated environment? Will it behave the same as the true environment as shown in the screenshot? If not, please revise the program (setups of the simulator) accordingly. You can use the DSL to specify the environment in a high-level way, and the simulator will handle the low-level details. Please surround your code with triple backticks (```) so that it is formatted correctly.
'''.strip()},
    ]

