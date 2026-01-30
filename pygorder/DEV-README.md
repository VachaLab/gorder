## For developers

### Running tests

- Normally: `uv run pytest`
- If you change the core `gorder` code or anything inside `src`: `uv run --reinstall pytest`

### Creating pyi files

- Activate a conda environment.
- Run `./make_types.sh`.
(If libpython cannot be found, run `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`)

### Generating documentation

- Activate a conda environment.
- Run maturin: `maturin develop --release`.
- Go to `docs` and run `make`: `cd docs && make clean html`.
- Open `build/index.html`.