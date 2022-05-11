# Generate an instant-ngp cameras.json from dd-nerf-dataset
import random
import util
import cv2
import open3d as o3d
import numpy as np
import click
import json
from pathlib import Path

if __name__ == '__main__':


    import sys

    camerasfile = sys.argv[1]
    samples = int(sys.argv[2])


    path = Path(camerasfile)
    print(path.parent.absolute())

    outputfile = path.parent.absolute() / 'cameras.json'

    # Load the cameras file and bring everything into local ENU coordiante system in meters.
    print(f"loading {click.style(camerasfile, fg='yellow', bold=True)}")
    cameras = util.CamerasXML().read(camerasfile)

    camera_names = sorted(cameras.cameras.keys())

    payload = {
        'frames' : [],
    }

    DOWNSAMPLE = 1.0

    up = np.zeros(3)
    xyzs = []
    frames = []

    for cameraname in camera_names:

        camera = cameras.cameras[cameraname]

        if not camera.structured:
            continue

        payload['fl_x']           = camera.sensor.camera.K[0,0]
        payload['fl_y']           = camera.sensor.camera.K[1,1]
        payload['k1']             = camera.sensor.camera.k1
        payload['k2']             = camera.sensor.camera.k2
        payload['p1']             = camera.sensor.camera.p1
        payload['p2']             = camera.sensor.camera.p2
        payload['k3']             = camera.sensor.camera.k3
        payload['k4']             = camera.sensor.camera.k4
        payload['k5']             = 0.0
        payload['k6']             = 0.0
        payload['cx']             = camera.sensor.camera.K[0,2]
        payload['cy']             = camera.sensor.camera.K[1,2]
        payload['w']              = camera.sensor.resolution.x
        payload['h']              = camera.sensor.resolution.y
        payload['camera_angle_x'] = 2 * np.arctan(0.5 * camera.sensor.resolution.x / camera.sensor.camera.K[0,0])
        payload['camera_angle_y'] = 2 * np.arctan(0.5 * camera.sensor.resolution.y / camera.sensor.camera.K[0,0])
        payload['aabb_scale']     = 1

        pose = np.linalg.inv(camera.project.pose)
        pose = np.asarray(pose)
        pose[0:3,2] *= -1 # flip the z axis
        pose[0:3,1] *= -1 # flip the y axis
        pose = pose[[1,0,2,3],:] # swap y and z
        pose[2,:] *= -1 # flip whole world upside down

        up += pose[0:3,1]
        xyz = camera.project.position()
        xyzs.append(xyz)
        distance = np.linalg.norm(camera.project.position())

        frame = {
            'file_path' : f'./images/{camera.label}',
            'sharpness' : 800.0,
            'transform_matrix' : pose,
        }

        frames.append((distance, frame))

    frames = sorted(frames)[:samples]
    # frames = random.sample(frames, samples)

    for _, frame in frames:
        payload['frames'].append(frame)

    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = util.rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    for i, frame in enumerate(payload['frames']):
        payload['frames'][i]['transform_matrix'] = np.matmul(R, payload['frames'][i]['transform_matrix']) # rotate up to be the z axis


    center = util.center_attention(payload['frames'])
    print(center)

    for f in payload['frames']:
        f["transform_matrix"][0:3,3] -= center

    avglen = 0.
    for f in payload["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3,3])

    avglen /= len(payload['frames'])
    print("avg camera distance from origin", avglen)
    for f in payload["frames"]:
        f["transform_matrix"][0:3,3] *= 0.5 / avglen # scale to "nerf sized"

    for i, frame in enumerate(payload['frames']):
        payload['frames'][i]['transform_matrix'] = util.encode(payload['frames'][i]['transform_matrix'])

    print("Frames:", len(payload['frames']))
    # import random
    # payload['frames'] = random.sample(payload['frames'], samples)

    # write cameras.json file
    with open(outputfile, mode='w') as fd:
        fd.write(json.dumps(payload, indent=4))
