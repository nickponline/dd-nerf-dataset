DroneDeploy NeRF Dataset
===

DroneDeploy NeRF Dataset ("DNF") contains a number of real world scenes captured by drone that include high resolution images, camera poses (intrinsics and extrinsics) and dense color point clouds. It is intended for computer vision research.

Information
---

| Name | Image | Images | Points | Size | Preview |
|-------|------|-------:|-------:|-----:|---|
| house1 | ![house1](/imgs/house1.png?raw=true "house1")  |     220 |  3M | 1.7GB | [Pointcloud preview](https://drive.google.com/file/d/15AJlYmev3gd_DWsMHR3WTMchVKJq0RVx/view?usp=sharing)
| house2 | ![house2](/imgs/house2.png?raw=true "house2")  |     952 |  25M | 8.6GB | [Pointcloud preview](https://drive.google.com/file/d/1UotQdkuMQU2LXC5IGLy8QVWw4qmLDX7c/view?usp=sharing)
| house3 | ![house3](/imgs/house3.png?raw=true "house3")  |     182 |  5M | 1.4GB | [Pointcloud preview](https://drive.google.com/file/d/1Y8U9u0F7uU1Yy35b_4qjNut5r12gDYeh/view?usp=sharing)
| house4 | ![house4](/imgs/house4.png?raw=true "house4")  |     367 |  10M | 5.2GB | [Pointcloud preview](https://drive.google.com/file/d/1YjJTNUABCzdZEepC1NqWzvqt6HrxF8vp/view?usp=sharing)
| ruins1 | ![ruins1](/imgs/ruins1.png?raw=true "ruins1") |     391 |  6M | coming soon | [Pointcloud preview](https://drive.google.com/file/d/1Oy5otafHEyTs5L73I2btxk2P-rmuh37n/view?usp=sharing)
| ruins2 | ![ruins2](/imgs/ruins2.png?raw=true "ruins2") |     1174 |  56M | coming soon | [Pointcloud preview](https://drive.google.com/file/d/1Awi2eG69Iyggzn0jCDqg45nV41KwO5ZB/view?usp=sharing)
| ruins3 | ![ruins3](/imgs/ruins3.png?raw=true "ruins3") |     539 |  12M | coming soon | [Pointcloud preview](https://drive.google.com/file/d/1BMPBVFeCn4HbksKtB1BPfEzsRjQx_GRE/view?usp=sharing)
| tower1 | ![tower1](/imgs/tower1.png?raw=true "tower1") |     783 |  42M | coming soon | [Pointcloud preview](https://drive.google.com/file/d/1d4V0_MqoYebqlSndAnjKeyoKSUbZJs97/view?usp=sharing)
| tower2 | ![tower2](/imgs/tower2.png?raw=true "tower2") |     684 |  16M | coming soon | [Pointcloud preview](https://drive.google.com/file/d/1mzU9sGT5unzV2Pd9rfSkicS-O181pt2k/view?usp=sharing)
| pipes1 | ![pipes1](/imgs/pipes1.png?raw=true "pipes1") |     98 |  7M | coming soon | [Pointcloud preview](https://drive.google.com/file/d/1Zme6vZsdaQcncvHxKa-y-n-C2APTDUrz/view?usp=sharing)


Download
---
Download Links: [https://drive.google.com/drive/folders/1kMHAS5l9Zy9yWHp6N4icm2xC7sIeThs7](https://drive.google.com/drive/folders/1kMHAS5l9Zy9yWHp6N4icm2xC7sIeThs7)


Structure
---

```
house1/
   - cameras.xml
   - images
      - 00001.jpg
      - 00002.jpg
      - 00003.jpg
   - points
      - points.ply
house2/ ...
house3/ ...
  ...
```

The `cameras.xml` contains the poses of each camera along with the distortion parameters, intrinsics and extrinsics. See `tools/util.py` for working with this file directly. The `images` directory contains the images and `points/points.ply` contains the dense pointcloud with `x,y,z,red,green,blue`.

Tools
---

The `tools` directory contains an expanding collection of tools for working with and transforming the dataset. Contributions are welcome.

`tools/dd2ngp.py` - create a `cameras.json` ready for use with [Instant Neural Graphics Primitives](https://github.com/NVlabs/instant-ngp). The default `cameras.json` is included in each zip of the dataset and uses the 50 nearest images to the center of the scene. If you want to use different cameras you can use this script to re-generate a `cameras.json`.


Usage
---
```sh
instant-ngp$ ./build/testbed --scene ./datasets/house1/cameras.json
```
![ngp](/imgs/ngp.png?raw=true "ngp")

or

```sh
python scripts/run.py --mode="nerf" --scene="house1/cameras.json" --screenshot_transforms="house1/cameras.json"  --width="1216" --height="912" --screenshot_dir="house1/output" --near_distance="0"  --screenshot_spp="16"
```
![ngp](/imgs/scipt.jpg?raw=true "script")



## License and Citation

```bibtex
@misc{Pilkington2022,
  author = {Nicholas Pilkington},
  title = {DroneDeploy NeRF Dataset},
  year = {2022},
  month = may
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nickponline/dd-nerf-dataset}},
}
```

This work is made available for academic use under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/). For commercial use and queries please contact support@dronedeploy.com.
