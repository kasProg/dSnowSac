"""Loads per-basin CAMELS forcing + observed streamflow, and derives
daily PET (not provided directly by CAMELS' Daymet forcing -- see
notes/logs.md).

File formats confirmed by reading actual extracted files directly, not
assumed from documentation:

Forcing (basin_mean_forcing/daymet/{HUC}/{gauge_id}_lump_cida_forcing_leap.txt):
    line 1: basin-mean latitude (deg)
    line 2: basin-mean elevation (m)
    line 3: basin area (m^2)
    line 4: column header
    data:   Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)
    whitespace-separated, one row per day, 1980-01-01 through 2014-12-31.

Streamflow (usgs_streamflow/{HUC}/{gauge_id}_streamflow_qc.txt):
    no header. columns: gauge_id year month day discharge_cfs qc_flag
    whitespace-separated. Missing days: discharge = -999.00, flag = "M"
    (confirmed by grep across the archive, e.g. gauge 01013500 2014-10).
    Observed flags in this archive: "A" (approved), "A:e" (approved,
    estimated), "M" (missing).

HUC subfolder does NOT reliably match the gauge_id's own leading digits
(e.g. gauge 06614800's forcing file lives under .../daymet/10/, not
.../daymet/06/) -- located by glob, not by computing the HUC from the ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CAMELS_DIR = Path(__file__).resolve().parent / "camels"
DATASET_DIR = CAMELS_DIR / "basin_dataset_public_v1p2"
PET_CACHE_DIR = CAMELS_DIR / "pet"

CFS_TO_MM_PER_DAY_PER_KM2 = 2.446575  # see module docstring derivation in notes/logs.md

# Hamon (1963) PET, as given by USACE HEC-HMS's technical reference
# (authoritative, unambiguous source for the exact constants -- several
# variants with different-looking constants circulate in the literature,
# all algebraically equivalent once you track units through consistently).
# Kept for reference/comparison; load_basin_timeseries() now uses
# hargreaves_pet() instead -- see that function's docstring for why.
_HAMON_C = 0.1651  # mm/day per g/m^3, at N=12hr daylight

# Hargreaves-Samani (1985) constants and the FAO-56 (Allen et al. 1998)
# extraterrestrial-radiation procedure it needs, verified against
# PyETo's implementation (MIT-licensed FAO-56 reference library --
# formula/constants only, no code copied) since several
# differently-scaled-looking variants of "Hargreaves" circulate in the
# literature and it's easy to mix conventions across two sources. See
# notes/logs.md for the verification and a sanity-check value.
_SOLAR_CONSTANT = 0.0820  # MJ / m^2 / min


def _sol_dec(day_of_year: np.ndarray) -> np.ndarray:
    """Solar declination, radians. FAO-56 eq. 24."""
    return 0.409 * np.sin(2 * np.pi / 365 * day_of_year - 1.39)


def _sunset_hour_angle(lat_rad: float, sol_dec: np.ndarray) -> np.ndarray:
    """FAO-56 eq. 25. Clamped to [-1, 1] before arccos -- polar-latitude
    edge case, not expected for CONUS basins, guarded anyway."""
    x = np.clip(-np.tan(lat_rad) * np.tan(sol_dec), -1.0, 1.0)
    return np.arccos(x)


def _inv_rel_dist_earth_sun(day_of_year: np.ndarray) -> np.ndarray:
    """FAO-56 eq. 23."""
    return 1 + 0.033 * np.cos(2 * np.pi / 365 * day_of_year)


def et_rad(lat_deg: float, day_of_year: np.ndarray) -> np.ndarray:
    """Extraterrestrial radiation Ra, MJ/m^2/day. FAO-56 eq. 21."""
    lat_rad = np.deg2rad(lat_deg)
    dr = _inv_rel_dist_earth_sun(day_of_year)
    delta = _sol_dec(day_of_year)
    ws = _sunset_hour_angle(lat_rad, delta)
    return (
        (1440.0 / np.pi) * _SOLAR_CONSTANT * dr
        * (ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
    )


def hargreaves_pet(
    tmax_c: np.ndarray, tmin_c: np.ndarray, lat_deg: float, day_of_year: np.ndarray
) -> np.ndarray:
    """Daily PET (mm/day) via Hargreaves & Samani (1985), FAO-56 form:
    ETo = 0.0023 * (Tmean + 17.8) * sqrt(Tmax - Tmin) * 0.408 * Ra
    (the 0.408 factor converts Ra from MJ/m^2/day to mm/day-equivalent).
    Needs only temperature + latitude/day-of-year (via Ra) -- no
    measured radiation/humidity/wind, same "temperature-only" data
    profile as Hamon, but generally regarded as more accurate (it's the
    FAO's own recommended fallback when full Penman-Monteith inputs
    aren't available). tmax_c/tmin_c: deg C. day_of_year: 1-366.
    """
    t_mean = (tmax_c + tmin_c) / 2.0
    ra = et_rad(lat_deg, day_of_year)
    trange = np.maximum(tmax_c - tmin_c, 0.0)  # guard against a bad tmax<tmin day
    pet = 0.0023 * (t_mean + 17.8) * np.sqrt(trange) * 0.408 * ra
    return np.maximum(pet, 0.0)


def hamon_pet(tmax_c: np.ndarray, tmin_c: np.ndarray, dayl_s: np.ndarray) -> np.ndarray:
    """Daily PET (mm/day) via Hamon (1963). tmax_c/tmin_c: deg C.
    dayl_s: day length, seconds (Daymet's own `dayl(s)` column -- already
    accounts for the basin's latitude via Daymet's own solar geometry
    model, so no separate latitude term is needed here)."""
    t_mean = (tmax_c + tmin_c) / 2.0
    n_hours = dayl_s / 3600.0
    e_s_mb = 6.108 * np.exp(17.27 * t_mean / (t_mean + 237.3))  # saturation vapor pressure, mb
    p_t = 216.7 * e_s_mb / (t_mean + 273.3)  # saturation vapor density, g/m^3
    pet = _HAMON_C * (n_hours / 12.0) * p_t
    return np.maximum(pet, 0.0)  # guard against a pathological negative from extreme T


def _find_file(subdir: Path, gauge_id: str, suffix_glob: str) -> Path:
    matches = list(subdir.glob(f"*/{gauge_id}{suffix_glob}"))
    if not matches:
        raise FileNotFoundError(f"No file matching {gauge_id}{suffix_glob} under {subdir}")
    assert len(matches) == 1, f"multiple matches for {gauge_id}: {matches}"
    return matches[0]


@dataclass
class BasinTimeseries:
    dates: pd.DatetimeIndex
    prcp: np.ndarray       # mm/day
    tmax: np.ndarray       # deg C
    tmin: np.ndarray       # deg C
    tmean: np.ndarray      # deg C, (tmax+tmin)/2 -- what we feed as Snow17's/SAC-SMA's TMP
    pet: np.ndarray        # mm/day, Hamon
    q_obs: np.ndarray      # mm/day, NaN where missing/QC-flagged "M"
    lat: float
    elev: float
    area_km2: float


def _load_raw_forcing(gauge_id: str) -> tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """(dates, prcp, tmax, tmin, lat, elev, area_km2) -- forcing only, no
    PET, no streamflow. Split out from load_basin_timeseries() so PET
    caching (get_pet(), below) can depend on just this, not the
    streamflow merge."""
    forcing_path = _find_file(
        DATASET_DIR / "basin_mean_forcing" / "daymet", gauge_id, "_lump_cida_forcing_leap.txt"
    )
    with open(forcing_path) as f:
        lat = float(f.readline().strip())
        elev = float(f.readline().strip())
        area_m2 = float(f.readline().strip())
    forcing = pd.read_csv(forcing_path, sep=r"\s+", skiprows=3)
    dates = pd.to_datetime(
        forcing[["Year", "Mnth", "Day"]].rename(columns={"Mnth": "month", "Day": "day", "Year": "year"})
    )
    prcp = forcing["prcp(mm/day)"].to_numpy(dtype=np.float64)
    tmax = forcing["tmax(C)"].to_numpy(dtype=np.float64)
    tmin = forcing["tmin(C)"].to_numpy(dtype=np.float64)
    area_km2 = area_m2 / 1e6
    return dates, prcp, tmax, tmin, lat, elev, area_km2


def get_pet(gauge_id: str, dates: pd.Series, tmax: np.ndarray, tmin: np.ndarray, lat: float) -> np.ndarray:
    """Hargreaves PET for one basin's full record, cached to
    data/camels/pet/{gauge_id}_pet.csv (date, pet_mm_day -- plain CSV,
    not a binary format, specifically so the values are inspectable, not
    just an implementation detail buried inside a training run). Computed
    fresh and written on first access if the cache file doesn't exist yet;
    data/build_pet.py precomputes it for every selected basin up front so
    a training run reads from cache rather than triggering computation
    implicitly the first time a basin is touched."""
    cache_path = PET_CACHE_DIR / f"{gauge_id}_pet.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if len(cached) == len(dates) and (cached["date"].to_numpy() == pd.DatetimeIndex(dates).to_numpy()).all():
            return cached["pet"].to_numpy(dtype=np.float64)
        # Cache exists but doesn't match this basin's current date range
        # (e.g. re-extracted data) -- fall through and recompute/overwrite
        # rather than silently returning a mismatched series.

    day_of_year = dates.dt.dayofyear.to_numpy()
    pet = hargreaves_pet(tmax, tmin, lat, day_of_year)

    PET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": pd.DatetimeIndex(dates), "pet": pet}).to_csv(cache_path, index=False)
    return pet


def load_basin_timeseries(gauge_id: str, all_forcing_days: bool = False) -> BasinTimeseries:
    """all_forcing_days=False: only days with both forcing and a
    streamflow record. True: every forcing day, q_obs NaN where the
    streamflow record has none -- a model run (e.g. a warmup before the
    gauge's record starts) needs contiguous forcing, not streamflow."""
    dates, prcp, tmax, tmin, lat, elev, area_km2 = _load_raw_forcing(gauge_id)
    tmean = (tmax + tmin) / 2.0
    pet = get_pet(gauge_id, dates, tmax, tmin, lat)

    flow_path = _find_file(DATASET_DIR / "usgs_streamflow", gauge_id, "_streamflow_qc.txt")
    flow = pd.read_csv(
        flow_path, sep=r"\s+", header=None,
        names=["gauge_id", "year", "month", "day", "q_cfs", "flag"],
        dtype={"gauge_id": str},
    )
    flow_dates = pd.to_datetime(flow[["year", "month", "day"]])
    q_mm_day = flow["q_cfs"].to_numpy(dtype=np.float64) * CFS_TO_MM_PER_DAY_PER_KM2 / area_km2
    q_mm_day[(flow["flag"] == "M").to_numpy() | (flow["q_cfs"].to_numpy() < 0)] = np.nan

    # Align forcing and streamflow on date (both nominally span 1980-01-01
    # to 2014-12-31 daily, but don't assume -- merge explicitly).
    forcing_df = pd.DataFrame({
        "date": dates, "prcp": prcp, "tmax": tmax, "tmin": tmin,
        "tmean": tmean, "pet": pet,
    })
    flow_df = pd.DataFrame({"date": flow_dates, "q_obs": q_mm_day})
    how = "left" if all_forcing_days else "inner"
    merged = forcing_df.merge(flow_df, on="date", how=how).sort_values("date")

    return BasinTimeseries(
        dates=pd.DatetimeIndex(merged["date"]),
        prcp=merged["prcp"].to_numpy(),
        tmax=merged["tmax"].to_numpy(),
        tmin=merged["tmin"].to_numpy(),
        tmean=merged["tmean"].to_numpy(),
        pet=merged["pet"].to_numpy(),
        q_obs=merged["q_obs"].to_numpy(),
        lat=lat, elev=elev, area_km2=area_km2,
    )
