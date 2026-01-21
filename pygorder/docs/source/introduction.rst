Introduction
============

``gorder`` is a tool designed for comprehensive, reliable, simple, and fast calculations of lipid order parameters from Gromacs simulations. While ``gorder`` is primarily a CLI application, it also provides a **Python API** for advanced users who need programmatic access to its functionality.

Most users do not need to learn the Python API and can use the CLI instead. For general usage, refer to the `gorder manual <https://vachalab.github.io/gorder-manual/>`_, which covers all features, including leaflet classification, error estimation, geometric selection, dynamic membrane normal estimation, and ordermap construction.

This documentation describes the Python API for those who want to integrate ``gorder`` with their Python code.