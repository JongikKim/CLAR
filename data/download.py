#!/usr/bin/env python3
"""Download the raw files of the datasets used in the paper into data/raw/<NAME>/.

    python data/download.py --dataset PEMS_BAY GBA_2000
    python data/download.py --dataset all

Approximate download sizes: NYC_TAXI 1.8 GB, NYC_BIKE 1.6 GB (6.5 GB of CSV
after extraction), CHI_SCOOTER 1.4 GB, PEMS_BAY 86 MB, GBA_2000 7.6 GB (the
LargeST archive; three of its files, 7.8 GB, are kept).
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.parse
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
DATASETS = ("NYC_TAXI", "NYC_BIKE", "CHI_SCOOTER", "PEMS_BAY", "GBA_2000")
TLC = "https://d37ci6vzurychx.cloudfront.net"
CITIBIKE = "https://s3.amazonaws.com/tripdata"
SCOOTER = "https://data.cityofchicago.org/resource/2i5w-ykuw.csv"
PEMS_BAY = "https://zenodo.org/api/records/5724362/files/PEMS-BAY.csv/content"
LARGEST = "https://www.kaggle.com/api/v1/datasets/download/liuxu77/largest"


def fetch(url: str, dest: Path, force: bool = False) -> Path:
    """Stream a URL to a file, skipping files that already exist."""
    if dest.exists() and not force:
        print(f"  exists: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  downloading {dest.name}", flush=True)
    with urlopen(url, timeout=600) as response, part.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 22)
    part.rename(dest)
    return dest


def _csvs_from(archive: zipfile.ZipFile, target: Path) -> int:
    """Extract every CSV of an archive, nested monthly archives included."""
    written = 0
    for member in archive.namelist():
        name = Path(member).name
        if "__MACOSX" in member or name.startswith("._"):
            continue
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as inner:
                written += _csvs_from(inner, target)
        elif name.lower().endswith(".csv"):
            with archive.open(member) as source, (target / name).open("wb") as handle:
                shutil.copyfileobj(source, handle)
            written += 1
    return written


def nyc_taxi(raw: Path, force: bool) -> None:
    """NYC TLC yellow taxi trip records for 2016."""
    for month in range(1, 13):
        name = f"yellow_tripdata_2016-{month:02d}.parquet"
        fetch(f"{TLC}/trip-data/{name}", raw / name, force)


def _taxi_zones(raw: Path, force: bool) -> None:
    """NYC taxi zone shapefile, extracted under raw/taxi_zones/."""
    archive = fetch(f"{TLC}/misc/taxi_zones.zip", raw / "taxi_zones.zip", force)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw / "taxi_zones")


def nyc_bike(raw: Path, force: bool) -> None:
    """Citi Bike trip records for 2023 and the NYC taxi zone shapefile."""
    raw.mkdir(parents=True, exist_ok=True)
    missing = [f"2023{m:02d}" for m in range(1, 13)
               if force or not list(raw.glob(f"2023{m:02d}-citibike-tripdata*.csv"))]
    if missing:
        try:
            for month in missing:
                archive = fetch(f"{CITIBIKE}/{month}-citibike-tripdata.zip",
                                raw / f"{month}-citibike-tripdata.zip", force)
                with zipfile.ZipFile(archive) as zf:
                    _csvs_from(zf, raw)
                archive.unlink()
        except (HTTPError, URLError):
            archive = fetch(f"{CITIBIKE}/2023-citibike-tripdata.zip",
                            raw / "2023-citibike-tripdata.zip", force)
            with zipfile.ZipFile(archive) as zf:
                _csvs_from(zf, raw)
            archive.unlink()
    _taxi_zones(raw, force)


def chi_scooter(raw: Path, force: bool) -> None:
    """City of Chicago E-Scooter Trips starting in 2025, through the Socrata API."""
    query = urllib.parse.urlencode({
        "$select": "trip_id,start_time,end_time,vendor,start_community_area_number,"
                   "start_centroid_latitude,start_centroid_longitude,trip_distance,trip_duration",
        "$where": "start_time >= '2025-01-01T00:00:00' AND start_time < '2026-01-01T00:00:00'",
        "$limit": 20000000})
    fetch(f"{SCOOTER}?{query}", raw / "trips_2025.csv", force)


def pems_bay(raw: Path, force: bool) -> None:
    """PEMS-BAY speeds, CSV release on Zenodo (record 5724362, CC BY 4.0)."""
    fetch(PEMS_BAY, raw / "PEMS-BAY.csv", force)


def gba_2000(raw: Path, force: bool) -> None:
    """LargeST (Kaggle liuxu77/largest): the 2019 flow, sensor metadata and road network."""
    wanted = ("ca_his_raw_2019.h5", "ca_meta.csv", "ca_rn_adj.npy")
    if all((raw / name).exists() for name in wanted) and not force:
        print("  exists: " + ", ".join(wanted))
        return
    archive = fetch(LARGEST, raw / "largest.zip", force)
    with zipfile.ZipFile(archive) as zf:
        for name in wanted:
            print(f"  extracting {name}", flush=True)
            zf.extract(name, raw)
    archive.unlink()


DOWNLOADERS = {"NYC_TAXI": nyc_taxi, "NYC_BIKE": nyc_bike, "CHI_SCOOTER": chi_scooter,
               "PEMS_BAY": pems_bay, "GBA_2000": gba_2000}


def main() -> None:
    """Command-line entry."""
    parser = argparse.ArgumentParser(description="Download raw datasets into data/raw/<NAME>/")
    parser.add_argument("--dataset", nargs="+", required=True, choices=DATASETS + ("all",))
    parser.add_argument("--raw-dir", type=Path, default=HERE / "raw")
    parser.add_argument("--force", action="store_true", help="download files that already exist")
    args = parser.parse_args()
    names = DATASETS if "all" in args.dataset else args.dataset
    for name in names:
        print(name)
        try:
            DOWNLOADERS[name](args.raw_dir / name, args.force)
        except (HTTPError, URLError) as error:
            sys.exit(f"{name}: download failed ({error})")


if __name__ == "__main__":
    main()
