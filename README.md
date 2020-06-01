# Pytorch Implementation of TrajectoryNet

This library runs code associated with the TrajectoryNet paper


## Installation

Download code
```
git clone http://github.com/krishanswamylab/TrajectoryNet.git
```


## Example
<p align="center">
<img align="middle" src="./figures/eb_high_quality.png" alt="EB PHATE Scatterplot" height="300" />
<img align="middle" src="./figures/EB-Trajectory.gif" alt="Trajectory of density over time" height="300" />
</p>

## Basic Usage

Run with
```
python main.py --dataset EB
```

To use a custom dataset expose the coordinates and timepoint information according to the `SCData` class in `dataset.py`

### References
[1] Tong, A., Huang, J., Wolf, G., van Dijk, D., and Krishnaswamy, S. TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics. In International Conference on Machine Learning, 2020. [[arxiv]](http://arxiv.org/abs/2002.04461)

---

If you found this library useful, please consider citing
```
@inproceedings{tong2020trajectorynet,
  title = {{{TrajectoryNet}}: {{A Dynamic Optimal Transport Network}} for {{Modeling Cellular Dynamics}}},
  shorttitle = {{{TrajectoryNet}}},
  booktitle = {Proceedings of the 37th {{International Conference}} on {{Machine Learning}}},
  author = {Tong, Alexander and Huang, Jessie and Wolf, Guy and {van Dijk}, David and Krishnaswamy, Smita},
  year = {2020}
}
```
