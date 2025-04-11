#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp, sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

import cv2

from proggen.data import Dataset
from proggen.utils.render import OpenCVRender

def test_data():
    dataset = Dataset('collision', 'train')
    data = dataset[10]
    print(data)

    video = data['frames'] # numpy array of shape (T, H, W, C)
    # Play the video
    for frame in video:
        cv2.imshow('frame', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    trajs = data['trajs'] # list of trajectories
    renderer = OpenCVRender(256, 256, 25.6)
    for traj in trajs:
        img = renderer.render(traj)
        cv2.imshow('img', img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    renderer.close()

if __name__ == '__main__':
    test_data()



