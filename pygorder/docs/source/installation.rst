Installation
============

Using uv
++++++++

The recommended way to install ``gorder`` as a Python package is to use the
`uv <https://github.com/astral-sh/uv>`_ package manager. To add ``gorder`` to your
uv project, run:

.. code-block:: bash

    uv add git+https://github.com/VachaLab/gorder.git#subdirectory=pygorder

Using pip
+++++++++

If you don't use uv, you can also install ``gorder`` using pip:

.. code-block:: bash

    pip install git+https://github.com/VachaLab/gorder.git#subdirectory=pygorder

Optional dependencies
+++++++++++++++++++++

If you intend to provide leaflet assignment or membrane normals manually using a dictionary,
or if you want to access maps of order parameters, collected leaflet assignments, or calculated
membrane normals directly from Python, you also need to `install NumPy <https://numpy.org/install>`_.

You do not need NumPy for regular usage.