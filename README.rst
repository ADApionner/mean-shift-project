.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/mean_shift_lab.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/mean_shift_lab
    .. image:: https://readthedocs.org/projects/mean_shift_lab/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://mean_shift_lab.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/mean_shift_lab/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/mean_shift_lab
    .. image:: https://img.shields.io/pypi/v/mean_shift_lab.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/mean_shift_lab/
    .. image:: https://img.shields.io/conda/vn/conda-forge/mean_shift_lab.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/mean_shift_lab
    .. image:: https://pepy.tech/badge/mean_shift_lab/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/mean_shift_lab
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/mean_shift_lab

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==============
mean_shift_lab
==============

**Mean-shift Project documentation**

This project implements the Mean Shift clustering algorithm from scratch and compares its performance and accuracy against the reference implementation in ``scikit-learn``.

Authors:

Programmer: Wiktor Stawiski
Tester: Antoni Bajor

Repository Structure
====================

.. code-block:: text

    my_project/
    ├── AUTHORS.rst
    ├── LICENSE.txt
    ├── README.rst             
    ├── setup.cfg              
    ├── requirements.txt      
    ├── src/
    │   └── mean_shift_lab/
    │       ├── __init__.py
    │       └── mean_shift.py  <-- Source code 
    ├── tests/
    │   └── test_mean_shift.py <-- Unit tests
    └── experiments/
        └── compare_sklearn.py <-- Comparison script

Requirements & Installation
===========================

This project is structured using ``PyScaffold``.

Windows PowerShell:

.. code-block:: powershell

    virtualenv .venv
    # Aktywacja (Windows): .venv\Scripts\activate
    pip install numpy matplotlib scikit-learn pytest 
    pip install -e .

Algorithm Description
=====================

Mean Shift is a non-parametric clustering algorithm that does not require prior knowledge of the number of clusters. It works by iteratively shifting data points towards the mode (the highest density of data points) in a process known as *mode seeking*.

Implementation Logic
--------------------

1. **Initialization:** Every data point is treated as a starting point.

2. **Shift Phase (Hill Climbing):** For each point, a window of radius ``bandwidth`` is defined. The "center of mass" (mean) of all neighbors within this window is calculated using a **Flat Kernel**. The point is shifted to this new mean. This is repeated until convergence (when points stop moving significantly).

3. **Merging Phase:** Points that converge to the same (or very close) locations are grouped together. These locations become the centroids.

4. **Labeling:** All original points are assigned to the cluster of their nearest centroid.

Usage
=====

You can use the ``MyMeanShift`` class just like any Scikit-Learn estimator.

.. code-block:: python

    import numpy as np
    from mean_shift_lab.mean_shift import MyMeanShift

    # 1. Prepare data
    X_train = np.array([
        [1, 1], [1.5, 1.5], [1, 2],  # Cluster 1
        [10, 10], [10, 11], [11, 10] # Cluster 2
    ])
     
    # 2. Initialize the model
    # 'bandwidth' determines the radius of the search window
    ms = MyMeanShift(bandwidth=2.0)

    # 3. Fit the model to data 
    ms.fit(X_train)

    # 4. Access results
    print("Centroids:", ms.centroids_) 
    print("Labels:", ms.labels_)

Comparison & Results
====================

We performed a comparative analysis between our custom implementation and ``sklearn.cluster.MeanShift``.

Methodology
-----------

* **Dataset:** Synthetic data generated using ``make_blobs`` (1000 samples, 3 centers, cluster_std=1.0).
* **Parameters:** Both implementations used ``bandwidth=1.5``.

Results
-------

* **Accuracy:** The custom implementation successfully identified the correct number of clusters. The calculated centroids are practically identical to the reference implementation.
* **Boundaries:** Decision boundaries align almost perfectly between the two models.
  
To reproduce the comparison experiment:

.. code-block:: bash

    python scripts/compare_sklearn.py

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.