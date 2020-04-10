.. AIToolbox documentation master file, created by
   sphinx-quickstart on Fri Apr 10 11:52:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AIToolbox's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   user_guide
   examples
   api/aitoolbox


Here is some normal text

`Link to AIToolbox repo <https://github.com/mv1388/aitoolbox>`_

.. math::
    y \sim \mathcal{N}(0, 1)


.. math:: \beta \sim \text{Poisson}(\lambda=5)
   :label: beta_prior

The prior on :math:`\beta` is a Poisson distribution with rate parameter of 5 :eq:`beta_prior`.




Section
-------

Here's some normal text.

 ::

  # And here's some code
  for i in range(5):
    print(i)

  # code block keeps going until un-indent

Normal text again


*Italic* text

Subsection
^^^^^^^^^^

.. automodule:: aitoolbox.torchtrain.train_loop
   :members:
   :inherited-members:
   :show-inheritance:




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
