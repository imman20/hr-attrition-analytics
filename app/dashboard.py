"""Saya menyajikan metrik kohor simulasi dengan filter departemen."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="HR Attrition Analytics", layout="wide")
st.title("HR Attrition Analytics")
st.warning("Data simulasi. Saya memakai hasil ini untuk diskusi retensi, bukan dasar pemutusan kerja atau sanksi.")


@st.cache_data
def load_outputs():
    return (
        pd.read_csv(ROOT / "reports/annual_turnover.csv"),
        pd.read_csv(ROOT / "reports/monthly_headcount.csv"),
        pd.read_csv(ROOT / "reports/km_curves.csv"),
        pd.read_csv(ROOT / "reports/current_risk.csv"),
        json.loads((ROOT / "reports/metrics.json").read_text()),
    )


if not (ROOT / "reports/metrics.json").exists():
    st.info("Saya membangun keluaran terlebih dahulu dengan make all dari root repo.")
    st.stop()
annual, monthly, km, risk, metrics = load_outputs()
department = st.sidebar.selectbox("Departemen", ["Semua"] + sorted(monthly.department.unique()))
year = st.sidebar.selectbox("Tahun", [2024, 2023, 2022])
row = annual[(annual.department == department) & (annual.year == year)].iloc[0]
a, b, c, d = st.columns(4)
a.metric("Turnover tahunan", f"{row.turnover_rate:.1%}")
b.metric("Rerata headcount harian", f"{row.average_headcount:.1f}")
c.metric("Exit sukarela", int(row.voluntary))
d.metric("Exit tidak sukarela", int(row.involuntary))
st.caption("Saya memakai employee-days / jumlah hari kalender sebagai penyebut. Hari keluar tidak masuk headcount aktif.")
filtered = monthly if department == "Semua" else monthly[monthly.department == department]
st.plotly_chart(
    px.line(filtered, x="month", y="headcount", color="department", title="Headcount akhir bulan, kohor rekrutmen 2022"), width="stretch"
)
curves = km[km.grouping == "department"]
if department != "Semua":
    curves = curves[curves.group == department]
st.plotly_chart(
    px.line(curves, x="day", y="survival", color="group", line_shape="hv", title="Kaplan-Meier sejak tanggal masuk"), width="stretch"
)
st.subheader("Risiko keluar 12 bulan pada karyawan aktif, snapshot 1 Januari 2025")
st.caption(
    "Saya menampilkan probabilitas, bukan label pasti. Proyeksi 2025 belum tervalidasi. Drivers adalah perubahan probabilitas terhadap acuan training per fitur, bukan efek kausal."
)
shown = risk if department == "Semua" else risk[risk.department == department]
st.dataframe(shown.head(30), hide_index=True, width="stretch")
st.subheader("Evaluasi holdout 2024")
st.dataframe(pd.DataFrame(metrics["models"]).T[["auc", "average_precision", "brier", "validation_ap"]], width="stretch")
st.caption(
    f"Saya memilih {metrics['selected_model']} melalui validation 2023 sebelum membuka test 2024. Usia, gender dan status menikah tidak dipakai sebagai fitur keputusan; potensi proxy tetap perlu ditinjau."
)
st.image(str(ROOT / "assets/evaluation.png"))
