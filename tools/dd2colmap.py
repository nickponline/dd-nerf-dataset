# Generate an instant-ngp cameras.json from dd-nerf-dataset
import random
import util
import random
import numpy as np
import click
import json
from pathlib import Path
import sys

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom


def centralize(out):
	# find a central point they are all looking at
	print("computing center of attention...")
	totw = 0.0
	totp = np.array([0.0, 0.0, 0.0])
	for f in out["frames"]:
		mf = (f["transform_matrix"])[0:3,:]
		for g in out["frames"]:
			mg = (g["transform_matrix"])[0:3,:]
			p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
			if w > 0.0001:
				totp += p*w
				totw += w
	totp /= totw
	print(totp) # the cameras are looking at totp
	for f in out["frames"]:
		f["transform_matrix"][0:3,3] -= totp
		f["transform_matrix"] = f["transform_matrix"].tolist()
	return out

if __name__ == '__main__':

    camerasfile = sys.argv[1]

    path = Path(camerasfile)
    print(path.parent.absolute())

    # outputfile = path.parent.absolute() / 'transforms.json'
    outputfile = camerasfile.replace("cameras.xml", "transforms.json")
    print(outputfile)
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
        payload['k3']             = camera.sensor.camera.k3
        payload['k4']             = camera.sensor.camera.k4
        payload['p1']             = camera.sensor.camera.p1
        payload['p2']             = camera.sensor.camera.p2
        payload['k5']             = 0.0
        payload['k6']             = 0.0
        payload['cx']             = camera.sensor.camera.K[0,2]
        payload['cy']             = camera.sensor.camera.K[1,2]
        payload['w']              = camera.sensor.resolution.x
        payload['h']              = camera.sensor.resolution.y
        payload['aabb_scale']     = 16 # MAYBE THIS

        pose = np.linalg.inv(camera.project.pose)
        pose = np.asarray(pose)
       
        # MAYBE THIS
        pose[0:3,2] *= -1 # flip the z axis
        pose[0:3,1] *= -1 # flip the y axis
        pose = pose[[1,0,2,3],:] # swap y and z
        pose[2,:] *= -1 # flip whole world upside down

        # pose = pose.tolist()

        frame = {
            'file_path' : f'./images/{camera.label}',
            'transform_matrix' : pose,
        }
        frames.append(frame)
    
    payload['frames'] = random.sample(frames, min(len(frames), 400))
    payload = centralize(payload)
    print(json.dumps(payload, indent=4))
    print(outputfile)
    with open(outputfile, mode='w') as fd:
        fd.write(json.dumps(payload, indent=4))
