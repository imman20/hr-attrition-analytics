"""Saya membangkitkan kohor rekrutmen simulasi dan snapshot tanpa kebocoran."""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2024
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2025-01-01")
ROOT = Path(__file__).resolve().parents[2]
DEPARTMENTS = ["Operasional", "Penjualan", "Teknologi", "Keuangan", "SDM"]
PROTECTED = ["age", "gender", "marital_status"]
NUMERIC = [
    "level",
    "monthly_salary",
    "last_raise_pct",
    "performance",
    "distance_km",
    "overtime_hours",
    "engagement",
    "tenure_years",
    "months_since_promotion",
]
CATEGORICAL = ["department", "job_title"]
FEATURES = NUMERIC + CATEGORICAL


def generate(n=1500, seed=SEED):
    """Saya menanamkan hazard Weibull sejak masuk; semua observasi adalah simulasi."""
    rng = np.random.default_rng(seed)
    dep = rng.choice(DEPARTMENTS, n, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    level = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.35, 0.23, 0.07])
    overtime = np.clip(rng.normal(15 + 10 * (dep == "Operasional") + 6 * (dep == "Penjualan"), 9), 0, 60)
    engagement = np.clip(rng.normal(3.8 - 0.018 * overtime, 0.65), 1, 5)
    distance = np.clip(rng.gamma(2.2, 7, n), 1, 80)
    raise_pct = np.clip(rng.normal(5, 2, n), 0, 12)
    salary = np.round((3_800_000 + level * 2_200_000 + (dep == "Teknologi") * 2_000_000) * rng.lognormal(0, 0.12, n), -3)
    hire = START + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    eta = 0.028 * (overtime - 20) - 0.6 * (engagement - 3.5) + 0.012 * (distance - 15) - 0.08 * (raise_pct - 5)
    eta += 0.28 * (dep == "Penjualan") + 0.16 * (dep == "Operasional") - 0.1 * (level - 2)
    duration_days = 365.25 * (-np.log(rng.uniform(size=n)) / (0.13 * np.exp(eta))) ** (1 / 1.2)
    theoretical_exit = hire + pd.to_timedelta(np.maximum(duration_days.astype(int), 1), unit="D")
    exited = theoretical_exit < END
    exit_date = pd.Series(theoretical_exit).where(exited)
    # Jadwal promosi independen dari hazard; snapshot mengabaikan tanggal mendatang.
    promotion = pd.Series(hire + pd.to_timedelta(rng.integers(180, 800, n), unit="D"))
    promotion = promotion.where((rng.random(n) < 0.25) & (promotion < exit_date.fillna(END)))
    reason = np.where(exited, np.where(rng.random(n) < 0.84, "Sukarela", "Tidak sukarela"), "Aktif")
    frame = pd.DataFrame(
        dict(
            employee_id=[f"E{i:04d}" for i in range(1, n + 1)],
            department=dep,
            job_title=np.array(["Staf", "Analis", "Supervisor", "Manajer"])[level - 1],
            level=level,
            monthly_salary=salary,
            last_raise_pct=raise_pct.round(2),
            last_raise_date=hire,
            performance=rng.choice([2, 3, 4, 5], n, p=[0.1, 0.45, 0.35, 0.1]),
            distance_km=distance.round(1),
            overtime_hours=overtime.round(1),
            engagement=engagement.round(2),
            age=np.clip(rng.normal(25 + level * 4, 5, n).astype(int), 20, 60),
            gender=rng.choice(["Perempuan", "Laki-laki"], n),
            marital_status=rng.choice(["Menikah", "Belum menikah"], n, p=[0.55, 0.45]),
            last_promotion_date=promotion,
            manager_id=[f"M{i:02d}" for i in rng.integers(1, 61, n)],
            hire_date=hire,
            exit_date=exit_date,
            exit_reason=reason,
        )
    )
    frame["tenure_years"] = (frame.exit_date.fillna(END) - frame.hire_date).dt.days / 365.25
    return frame


def snapshot(frame, when, labels=True):
    """Saya memakai hanya karyawan aktif pada awal hari, dengan label 12 bulan penuh."""
    when = pd.Timestamp(when)
    horizon = when + pd.DateOffset(years=1)
    if labels and horizon > END:
        raise ValueError("Horizon label melewati akhir observasi.")
    out = frame[(frame.hire_date <= when) & (frame.exit_date.isna() | (frame.exit_date >= when))].copy()
    out["tenure_years"] = (when - out.hire_date).dt.days / 365.25
    known_promotion = out.last_promotion_date.where(out.last_promotion_date < when)
    out["months_since_promotion"] = (when - known_promotion.fillna(out.hire_date)).dt.days / 30.4375
    if labels:
        out["target"] = ((out.exit_date >= when) & (out.exit_date < horizon)).astype(int)
    return out


def turnover(frame):
    """Saya membagi jumlah keluar dengan rerata headcount harian, tanpa hari keluar."""
    monthly = []
    for day in pd.date_range(START, END - pd.Timedelta(days=1), freq="ME"):
        for dep in DEPARTMENTS:
            active = (frame.hire_date <= day) & (frame.exit_date.isna() | (frame.exit_date > day)) & (frame.department == dep)
            monthly.append(dict(month=day.strftime("%Y-%m"), department=dep, headcount=int(active.sum())))
    annual = []
    for year in [2022, 2023, 2024]:
        lo, hi = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
        for dep in ["Semua"] + DEPARTMENTS:
            sub = frame if dep == "Semua" else frame[frame.department == dep]
            # Masa kerja harian [masuk, keluar). Kalender dengan penyebut 365/366 hari.
            stop = sub.exit_date.fillna(hi).clip(upper=hi)
            begin = sub.hire_date.clip(lower=lo)
            days = (stop - begin).dt.days.clip(lower=0).sum()
            avg = days / (hi - lo).days
            events = sub[(sub.exit_date >= lo) & (sub.exit_date < hi)]
            voluntary = int((events.exit_reason == "Sukarela").sum())
            annual.append(
                dict(
                    year=year,
                    department=dep,
                    exits=len(events),
                    voluntary=voluntary,
                    involuntary=len(events) - voluntary,
                    average_headcount=avg,
                    turnover_rate=len(events) / avg if avg else 0,
                    voluntary_rate=voluntary / avg if avg else 0,
                    involuntary_rate=(len(events) - voluntary) / avg if avg else 0,
                    mean_exit_tenure=events.tenure_years.mean(),
                )
            )
    return pd.DataFrame(monthly), pd.DataFrame(annual)
