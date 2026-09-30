# CLAR

Code for CLAR (Cross-Region Lag-Aligned Reads), a multi-region demand forecasting
model. CLAR mines lagged cross-region relations from the training data and, for
each forecast step, reads a source region at the time step given by the lag of
the relation.

## Requirements

Tested with Python 3.11, PyTorch 2.13 and NumPy 2.4. The data scripts also use
pandas 3.0, PyArrow 25 (NYC_TAXI), GeoPandas 1.1 (NYC_BIKE) and h5py 3.16
(GBA_2000).

```
pip install -r requirements.txt
```

## Data

CLAR reads a converted dataset directory `data/<NAME>/` with two files:

- `data.npz`: `y` (regions x time steps, float32), `dow` (day of week, 0 is
  Monday) and `tod` (time-of-day slot)
- `meta.json`: name, steps per day, time step, period, missing-value code and
  source

The converted demand datasets are included:

```
unzip data/NYC_TAXI.zip -d data/
unzip data/NYC_BIKE.zip -d data/
unzip data/CHI_SCOOTER.zip -d data/
```

The traffic datasets are downloaded and converted:

```
python data/download.py --dataset PEMS_BAY GBA_2000
python data/convert.py --dataset PEMS_BAY GBA_2000
```

`download.py` writes the raw files to `data/raw/<NAME>/` and `convert.py` writes
`data/<NAME>/`. Both take `--dataset` with any of `NYC_TAXI`, `NYC_BIKE`,
`CHI_SCOOTER`, `PEMS_BAY`, `GBA_2000` or `all`, so the demand datasets can also
be rebuilt from their raw files.

| Dataset | Source | Download |
|---|---|---|
| NYC_TAXI | NYC TLC yellow taxi trip records, 2016 | 1.8 GB |
| NYC_BIKE | Citi Bike trip records, 2023, joined to the NYC taxi zones | 1.6 GB |
| CHI_SCOOTER | City of Chicago E-Scooter Trips, 2025 | 1.4 GB |
| PEMS_BAY | PEMS-BAY, CSV release on Zenodo (record 5724362) | 86 MB |
| GBA_2000 | LargeST, Greater Bay Area (Kaggle `liuxu77/largest`) | 7.6 GB |

## Training

Demand datasets (NYC_TAXI, NYC_BIKE, CHI_SCOOTER):

```
python -m model.run --data-dir data/NYC_TAXI --window 6 --horizon 6 --seeds 0 1 2 --output-dir logs/clar/NYC_TAXI
```

Traffic datasets (PEMS_BAY, GBA_2000), where 0 marks a missing reading:

```
python -m model.run --data-dir data/PEMS_BAY --window 12 --horizon 12 --bank-size 24 --loss-mask 0 --seeds 0 1 2 --output-dir logs/clar/PEMS_BAY
```

Results (`results.json`) and checkpoints (`seed<k>.pt`) are written to
`--output-dir` (default `logs/clar`). Give each dataset its own directory, as
above, so that one run does not overwrite another. `python -m model.run --help`
lists all options.
