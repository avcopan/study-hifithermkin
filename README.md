# hifithermkin

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-border.json)](https://github.com/copier-org/copier)

Description here...

## Installation

Installation instructions...

## Usage

Usage instructions...

## The import and its provenance

`pixi run import-rgd1` and `pixi run import-stjohn` write RGD1 and stjohn-radicals into one
AutoStorage database each (`python -m hifithermkin.importer --help`); `python -m
hifithermkin.check` reports on a database against its sources.

Convention for what goes where, per Gaussian log:

- `ModelRow` is the wavefunction model only: program, version, method (`M06-2X`), basis
  (`def2-TZVP`) and the keywords that change the result (`reference` as `rks`/`uks`/`roks`,
  `empiricaldispersion`, a non-default `integral` grid, `scf` convergence, `stable`). Optimizer
  and job keywords are not model keywords.
- `CalculationRow.input_provenance` fully specifies the input: `dataset`, `archive`,
  `member`, `md5` of the log, `routes` (every route of the log), and `input`, a list with one
  entry per Gaussian job in the log, rebuilt from the input the log echoes: `link0` (the `%`
  lines), `route`, `title`, `charge`, `multiplicity`, `geometry` (the echoed atom lines as
  written; empty when the job reads it from a checkpoint) and `text`, the same in Gaussian
  input-file form.
- `CalculationRow.output_provenance` holds what the program reported: version, NImag,
  frequencies, zero-point energy, thermal corrections, temperature and pressure.

TSs get their reaction graph from the atom-mapped SMILES. When it disagrees with the h5's
geometries the TS is still imported, with a *flag* row in `X.db.failures.csv` (`flag:bond_changes`:
the bonds that only the SMILES or only the geometries break or form; `flag:unchanged_bond_missing`:
a bond that does not change is missing from the TS geometry's graph), named with the atoms
(numbered from 1, the map numbers).

The check reads the points in chunks, so it runs on the full databases: frequencies, InChI and
energies cover every point, and round trips and the TS graph sections a random `--sample`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## Credits

This package was created with [copier](https://copier.readthedocs.io/) and the [avcopan/copier-pixi-python](https://github.com/avcopan/copier-pixi-python) project template.
