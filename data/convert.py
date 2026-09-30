#!/usr/bin/env python3
"""Convert the raw files of the datasets used in the paper into CLAR's input format.

Each dataset is read from data/raw/<NAME>/ (see download.py) and written to
data/<NAME>/data.npz (y [regions, time steps] float32, dow and tod int64) and
data/<NAME>/meta.json.

    python data/convert.py --dataset NYC_TAXI
    python data/convert.py --dataset PEMS_BAY GBA_2000
    python data/convert.py --dataset all
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = ("NYC_TAXI", "NYC_BIKE", "CHI_SCOOTER", "PEMS_BAY", "GBA_2000")


def save(out_dir: Path, name: str, y, dow, tod, meta: dict) -> Path:
    """Write data.npz and meta.json of one dataset."""
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "data.npz", y=np.asarray(y, dtype=np.float32),
             dow=np.asarray(dow, dtype=np.int64), tod=np.asarray(tod, dtype=np.int64))
    (target / "meta.json").write_text(json.dumps({"name": name, **meta}, indent=2) + "\n")
    return target


def _calendar(index, step_seconds: int):
    """Day of week and time-of-day slot of every time step."""
    seconds = index.hour.to_numpy() * 3600 + index.minute.to_numpy() * 60 + index.second.to_numpy()
    return index.dayofweek.to_numpy(), seconds // step_seconds


def _count(pd, trips, time_column, zone_column, start, end, step_seconds, zones):
    """Trip starts per zone and time step, zones 1..zones."""
    step = pd.Timedelta(seconds=step_seconds)
    trips["time_bin"] = trips[time_column].dt.floor(step)
    index = pd.date_range(start, periods=int((end - start).value // step.value), freq=step)
    counts = trips.groupby([zone_column, "time_bin"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=range(1, zones + 1), columns=index, fill_value=0)
    return counts.to_numpy(), index


def nyc_taxi(raw: Path, out: Path) -> Path:
    """NYC TLC yellow taxi pickups per taxi zone, 30 min, 2016."""
    import pandas as pd
    start, end, zones = pd.Timestamp("2016-01-01"), pd.Timestamp("2017-01-01"), 263
    paths = sorted(raw.glob("yellow_tripdata_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no yellow_tripdata_*.parquet files in {raw}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=["tpep_pickup_datetime", "PULocationID"])
        frame = frame[(frame["tpep_pickup_datetime"] >= start) & (frame["tpep_pickup_datetime"] < end)
                      & (frame["PULocationID"] >= 1) & (frame["PULocationID"] <= zones)]
        if not frame.empty:
            frames.append(frame)
    trips = pd.concat(frames, ignore_index=True)
    y, index = _count(pd, trips, "tpep_pickup_datetime", "PULocationID", start, end, 1800, zones)
    dow, tod = _calendar(index, 1800)
    return save(out, "NYC_TAXI", y, dow, tod, {
        "steps_per_day": 48, "time_step": "30min", "start": "2016-01-01", "end": "2017-01-01",
        "missing_value": None, "source": "NYC TLC yellow taxi trip records"})


def nyc_bike(raw: Path, out: Path) -> Path:
    """Citi Bike trip starts per NYC taxi zone, 30 min, 2023."""
    import geopandas as gpd
    import pandas as pd
    start, end, zones = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"), 263
    shapes = sorted(raw.rglob("taxi_zones.shp"))
    if not shapes:
        raise FileNotFoundError(f"taxi_zones.shp not found under {raw}")
    gdf = gpd.read_file(shapes[0])
    gdf = gdf[(gdf["LocationID"] >= 1) & (gdf["LocationID"] <= zones)].sort_values("LocationID").reset_index(drop=True)
    if gdf["LocationID"].astype(int).tolist() != list(range(1, zones + 1)):
        raise ValueError(f"expected contiguous taxi-zone ids 1..{zones}")
    paths = sorted(raw.glob("*-citibike-tripdata*.csv"))
    if not paths:
        raise FileNotFoundError(f"no *-citibike-tripdata*.csv files in {raw}")
    step = pd.Timedelta(seconds=1800)
    index = pd.date_range(start, periods=int((end - start).value // step.value), freq=step)
    zone_shapes = gdf[["LocationID", "geometry"]].copy()
    total = None
    for path in paths:
        frame = pd.read_csv(path, usecols=["started_at", "start_lat", "start_lng"], low_memory=False)
        frame["started_at"] = pd.to_datetime(frame["started_at"], errors="coerce")
        frame = frame.dropna(subset=["started_at", "start_lat", "start_lng"])
        frame = frame[(frame["started_at"] >= start) & (frame["started_at"] < end)
                      & (frame["start_lat"] != 0.0) & (frame["start_lng"] != 0.0)]
        if frame.empty:
            continue
        points = gpd.GeoDataFrame(frame[["started_at"]],
                                  geometry=gpd.points_from_xy(frame["start_lng"], frame["start_lat"]),
                                  crs="EPSG:4326").to_crs(zone_shapes.crs)
        joined = gpd.sjoin(points, zone_shapes, predicate="within", how="inner")
        if joined.empty:
            continue
        joined["time_bin"] = joined["started_at"].dt.floor(step)
        counts = joined.groupby(["LocationID", "time_bin"]).size()
        total = counts if total is None else total.add(counts, fill_value=0)
    y = total.unstack(fill_value=0).reindex(index=range(1, zones + 1), columns=index, fill_value=0).to_numpy()
    dow, tod = _calendar(index, 1800)
    return save(out, "NYC_BIKE", y, dow, tod, {
        "steps_per_day": 48, "time_step": "30min", "start": "2023-01-01", "end": "2024-01-01",
        "missing_value": None, "source": "Citi Bike trip records joined to NYC taxi zones"})


def chi_scooter(raw: Path, out: Path) -> Path:
    """Chicago e-scooter trip starts per community area, 1 h, 2025-01-01 to 2025-12-30."""
    import pandas as pd
    start, end, zones = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), 77
    path = raw / "trips_2025.csv"
    if not path.is_file():
        raise FileNotFoundError(f"trip file not found: {path}")
    trips = pd.read_csv(path, usecols=["start_time", "start_community_area_number"])
    trips = trips.dropna(subset=["start_community_area_number"])
    trips["start_community_area_number"] = trips["start_community_area_number"].astype(int)
    trips["start_time"] = pd.to_datetime(trips["start_time"])
    trips = trips[(trips["start_time"] >= start) & (trips["start_time"] < end)
                  & (trips["start_community_area_number"] >= 1)
                  & (trips["start_community_area_number"] <= zones)]
    y, index = _count(pd, trips, "start_time", "start_community_area_number", start, end, 3600, zones)
    dow, tod = _calendar(index, 3600)
    return save(out, "CHI_SCOOTER", y, dow, tod, {
        "steps_per_day": 24, "time_step": "1h", "start": "2025-01-01", "end": "2025-12-31",
        "missing_value": None, "source": "City of Chicago E-Scooter Trips (data.cityofchicago.org/d/2i5w-ykuw)"})


def pems_bay(raw: Path, out: Path) -> Path:
    """PEMS-BAY speeds of 325 sensors on a gapless 5 min grid; absent readings are 0."""
    path = raw / "PEMS-BAY.csv"
    if not path.is_file():
        raise FileNotFoundError(f"speed file not found: {path}")
    start, steps = datetime(2017, 1, 1), 181 * 288
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        sensors = [name.strip() for name in next(reader)[1:]]
        grid = np.zeros((steps, len(sensors)), dtype=np.float32)
        for row in reader:
            grid[int((datetime.fromisoformat(row[0]) - start).total_seconds() // 300)] = \
                [float(value) for value in row[1:]]
    times = [start + timedelta(minutes=5 * i) for i in range(steps)]
    dow = [t.weekday() for t in times]
    tod = [(t.hour * 3600 + t.minute * 60 + t.second) // 300 for t in times]
    return save(out, "PEMS_BAY", grid.T, dow, tod, {
        "steps_per_day": 288, "time_step": "5min", "start": "2017-01-01", "end": "2017-07-01",
        "missing_value": 0.0, "sensor_ids": sensors,
        "source": "PEMS-BAY (Li et al., ICLR 2018), CSV release zenodo.org/records/5724362"})


def _bfs_order(adjacency, lat, lon, ids):
    """Breadth-first order from the sensor nearest the centroid, ties by sensor ID."""
    n = adjacency.shape[0]
    linked = (adjacency != 0) | (adjacency != 0).T
    linked.fill_diagonal_(False)
    centre = ((lat - lat.mean()) ** 2 + (lon - lon.mean()) ** 2).argmin().item()
    neighbours = [sorted(linked[i].nonzero().flatten().tolist(), key=lambda j: ids[j]) for i in range(n)]
    seen, visited, queue = {centre}, [], deque([centre])
    while queue:
        node = queue.popleft()
        visited.append(node)
        for j in neighbours[node]:
            if j not in seen:
                seen.add(j)
                queue.append(j)
    visited += sorted((i for i in range(n) if i not in seen), key=lambda j: ids[j])
    return visited


def gba_2000(raw: Path, out: Path, size: int = 2000) -> Path:
    """LargeST Greater Bay Area flow, 15 min, 2019, the first 2,000 sensors of a BFS from the centre."""
    import h5py
    import pandas as pd
    import torch
    meta = pd.read_csv(raw / "ca_meta.csv")
    gba = meta[meta.District == 4].reset_index(drop=True)
    if len(gba) != 2352:
        raise ValueError(f"District 4 holds {len(gba)} sensors, expected 2352")
    columns = gba.ID2.to_numpy()
    with h5py.File(raw / "ca_his_raw_2019.h5", "r") as f:
        stamps = pd.to_datetime(f["/t/axis1"][:])
        names = [x.decode() for x in f["/t/axis0"][:]]
        if [names[i] for i in columns] != [str(i) for i in gba.ID.tolist()]:
            raise ValueError("ID2 does not index the sensor columns by ID")
        total = f["/t/block0_values"].shape[0]
        binned = np.empty((total // 3, len(columns)), dtype=np.float32)
        for first in range(0, total, 5760):
            stop = min(first + 5760, total)
            block = f["/t/block0_values"][first:stop, :][:, columns].reshape(-1, 3, len(columns))
            observed = ~np.isnan(block)
            count = observed.sum(axis=1)
            summed = np.where(observed, block, 0.0).sum(axis=1)
            binned[first // 3: stop // 3] = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
    binned = np.round(binned)
    stamps = stamps[::3]
    adjacency = torch.from_numpy(
        np.load(raw / "ca_rn_adj.npy", mmap_mode="r")[columns][:, columns].astype(np.float32))
    lat = torch.tensor(gba.Lat.to_numpy(), dtype=torch.float32)
    lon = torch.tensor(gba.Lng.to_numpy(), dtype=torch.float32)
    index = sorted(_bfs_order(adjacency, lat, lon, gba.ID.astype(int).tolist())[:size])
    y = binned.T[index]
    return save(out, f"GBA_{size}", y, stamps.dayofweek.to_numpy(),
                (stamps.hour * 4 + stamps.minute // 15).to_numpy(), {
                    "steps_per_day": 96, "time_step": "15min", "start": "2019-01-01", "end": "2020-01-01",
                    "missing_value": 0.0, "sensor_ids": [int(gba.ID[i]) for i in index],
                    "source": "LargeST (NeurIPS 2023) Greater Bay Area, Caltrans District 4"})


CONVERTERS = {"NYC_TAXI": nyc_taxi, "NYC_BIKE": nyc_bike, "CHI_SCOOTER": chi_scooter,
              "PEMS_BAY": pems_bay, "GBA_2000": gba_2000}


def main() -> None:
    """Command-line entry."""
    parser = argparse.ArgumentParser(description="Convert raw datasets into data/<NAME>/")
    parser.add_argument("--dataset", nargs="+", required=True, choices=DATASETS + ("all",))
    parser.add_argument("--raw-dir", type=Path, default=HERE / "raw")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    names = DATASETS if "all" in args.dataset else args.dataset
    for name in names:
        target = CONVERTERS[name](args.raw_dir / name, args.out_dir)
        print(f"{name}: wrote {target}")


if __name__ == "__main__":
    main()
