# NS3D: Neuro-Symbolic Grounding of 3D Objects and Relations


![figure](figure.png)
<br />
<br />
**NS3D: Neuro-Symbolic Grounding of 3D Objects and Relations**
<br />
[Joy Hsu](http://web.stanford.edu/~joycj/),
[Jiayuan Mao](http://jiayuanm.com/),
[Jiajun Wu](https://jiajunwu.com/)
<br />
In Conference on Computer Vision and Pattern Recognition (CVPR) 2023
<br />

## Dataset
Our dataset download process follows the [ReferIt3D benchmark](https://github.com/referit3d/referit3d).

Specifically, you will need to
- (1) Download `sr3d_train.csv` and `sr3d_test.csv` from this [link](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV)
- (2) Download scans from ScanNet and process them according to this [link](https://github.com/referit3d/referit3d/blob/eccv/referit3d/data/scannet/README.md). This should result in a `keep_all_points_with_global_scan_alignment.pkl` file.

## Installation

Run the following commands to install necessary dependencies.

```Console
  conda create -n ns3d python=3.7.11
  conda activate ns3d
  pip -r requirements.txt
```

Install [Jacinle](https://github.com/vacancy/Jacinle).
```Console
  git clone https://github.com/vacancy/Jacinle --recursive
  export PATH=<path_to_jacinle>/bin:$PATH
```

Install the referit3d python package from [ReferIt3D](https://github.com/referit3d/referit3d).
```Console
    git clone https://github.com/referit3d/referit3d
    cd referit3d
    pip install -e .
```

And compile CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413).
```Console
    cd models/scene_graph/point_net_pp/pointnet2
    python setup.py install
```


## Evaluation


## Training


## Acknowledgements
