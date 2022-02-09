Pytorch Implementation of TrajectoryNet
=======================================

This library runs code associated with the TrajectoryNet paper [1].

In brief, TrajectoryNet is a Continuous Normalizing Flow model which can
perform dynamic optimal transport using energy regularization and / or a
combination of velocity, density, and growth regularizations to better match
cellular trajectories. 

Our setting is similar to that of `WaddingtonOT
<https://broadinstitute.github.io/wot/>`_. In that we have access to a bunch of
population measurements of cells over time and would like to model the dynamics
of cells over that time period. TrajectoryNet is trained end-to-end and is
continuous both in gene space and in time.


Installation
------------

TrajectoryNet is available in `pypi`. Install by running the following

.. code-block:: bash

    pip install TrajectoryNet

This code was tested with python 3.7 and 3.8.

Example
-------

.. image:: figures/eb_high_quality.png
    :alt: EB PHATE Scatterplot
    :height: 300

.. image:: figures/EB-Trajectory.gif
    :alt: Trajectory of density over time
    :height: 300


Basic Usage
-----------

Run with

.. code-block:: bash

    python -m TrajectoryNet.main --dataset SCURVE

To run TrajectoryNet on the `S Curve` example in the paper.  To use a
custom dataset expose the coordinates and timepoint information according
to the example jupyter notebooks in the `/notebooks/` folder.

If you have an `AnnData <https://anndata.readthedocs.io>`_ object then take a look at
`notebooks/Example_Anndata_to_TrajectoryNet.ipynb
<https://github.com/KrishnaswamyLab/TrajectoryNet/tree/master/notebooks>`_,
which shows how to load one of the example `scvelo <https://scvelo.readthedocs.io>`_ anndata objects into
TrajectoryNet. Alternatively you can use the custom (compressed) format for
TrajectoryNet as described below.

For this format TrajectoryNet requires the following:

1. An embedding matrix titled `[embedding_name]` (Cells x Dimensions)
2. A sample labels array titled `sample_labels` (Cells)

To run TrajectoryNet with a custom dataset use:

.. code-block:: bash

    python -m TrajectoryNet.main --dataset [PATH_TO_NPZ_FILE] --embedding_name [EMBEDDING_NAME]
    python -m TrajectoryNet.eval --dataset [PATH_TO_NPZ_FILE] --embedding_name [EMBEDDING_NAME]


See `notebooks/EB-Eval.ipynb` for an example on how to use TrajectoryNet on
a PCA embedding to get trajectories in the gene space.


References
----------
[1] Tong, A., Huang, J., Wolf, G., van Dijk, D., and Krishnaswamy, S. TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics. In International Conference on Machine Learning, 2020. `arxiv <http://arxiv.org/abs/2002.04461>`_ `ICML <https://proceedings.icml.cc/paper/2020/hash/9d740bd0f36aaa312c8d504e28c42163>`_

---

If you found this library useful, please consider citing::

    @inproceedings{tong2020trajectorynet,
      title = {TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics},
      shorttitle = {TrajectoryNet},
      booktitle = {Proceedings of the 37th International Conference on Machine Learning},
      author = {Tong, Alexander and Huang, Jessie and Wolf, Guy and {van Dijk}, David and Krishnaswamy, Smita},
      year = {2020}
    }
