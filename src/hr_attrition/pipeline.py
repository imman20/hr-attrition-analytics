"""Saya menghasilkan data, laporan, figur, dan notebook dalam satu alur kerja."""

import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from nbclient import NotebookClient
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from .data import END, FEATURES, ROOT, SEED, generate, snapshot, turnover
from .model import fit_models, risk_table

SURVIVAL_FEATURES = ["overtime_hours", "engagement", "distance_km", "last_raise_pct", "level", "monthly_salary"]


def survival_analysis(frame):
    observed = (frame.exit_date.fillna(END) - frame.hire_date).dt.days
    events = frame.exit_date.notna()
    km = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    groupings = [frame.department, pd.Series(np.where(frame.overtime_hours >= 20, "Lembur >=20 jam", "Lembur <20 jam"), index=frame.index)]
    pvalues = {}
    for axis, grouping, label in zip(axes, groupings, ["department", "overtime"]):
        for value in sorted(grouping.unique()):
            mask = grouping == value
            estimator = KaplanMeierFitter().fit(observed[mask], events[mask], label=value)
            estimator.plot_survival_function(ax=axis, ci_show=False)
            for day, survival in zip(estimator.survival_function_.index, estimator.survival_function_.iloc[:, 0]):
                km.append(dict(grouping=label, group=value, day=day, survival=survival))
        axis.set(xlabel="Hari sejak masuk", ylabel="Probabilitas bertahan", ylim=(0, 1.02))
        pvalues[label] = float(multivariate_logrank_test(observed, grouping, events).p_value)
        axis.set_title(f"Kaplan-Meier: {label}\nlog-rank p={pvalues[label]:.4g}")
    fig.tight_layout()
    fig.savefig(ROOT / "assets/survival.png", dpi=150)
    plt.close(fig)
    covariates = pd.get_dummies(frame[SURVIVAL_FEATURES + ["department"]], drop_first=True, dtype=float)
    covariates["monthly_salary"] /= 1_000_000
    covariates["duration"] = observed
    covariates["event"] = events
    cox = CoxPHFitter(penalizer=0.01).fit(covariates, "duration", "event")
    cox.summary.to_csv(ROOT / "reports/cox_coefficients.csv")
    diagnostic = proportional_hazard_test(cox, covariates, time_transform="rank").summary
    diagnostic["bonferroni_p"] = np.minimum(diagnostic.p * len(diagnostic), 1)
    diagnostic.to_csv(ROOT / "reports/cox_ph_diagnostics.csv")
    pd.DataFrame(km).to_csv(ROOT / "reports/km_curves.csv", index=False)
    cox.plot()
    plt.title("Cox PH: log hazard ratio dan interval 95%")
    plt.tight_layout()
    plt.savefig(ROOT / "assets/cox.png", dpi=150)
    plt.close()
    return dict(
        logrank_p=pvalues,
        concordance=float(cox.concordance_index_),
        ph_min_p=float(diagnostic.p.min()),
        ph_rejections_bonferroni=int((diagnostic.bonferroni_p < 0.05).sum()),
    )


def build_notebook():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# HR Attrition Analytics\n\nSaya memakai **data simulasi**, bukan catatan perusahaan. Saya meninjau keluaran pipeline dan memeriksa ulang penyebut turnover serta pemisahan waktu evaluasi."
            ),
            nbformat.v4.new_code_cell(
                "from pathlib import Path\nimport sys, json\nimport pandas as pd\nroot = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsys.path.insert(0, str(root/'src'))\nfrom hr_attrition.data import generate, turnover\nfrom hr_attrition.model import split_cohorts\ndata = generate()\nassert len(data) == 1500\ndata.head()"
            ),
            nbformat.v4.new_markdown_cell(
                "## Headcount dan turnover\n\nSaya memakai employee-days / jumlah hari kalender sebagai rerata headcount. Hari keluar tidak dihitung sebagai hari aktif. Turnover sukarela dan tidak sukarela memakai penyebut yang sama."
            ),
            nbformat.v4.new_code_cell("monthly, annual = turnover(data)\nannual[annual.department == 'Semua']"),
            nbformat.v4.new_markdown_cell(
                "## Evaluasi 12 bulan\n\nSaya memisahkan identitas karyawan sebelum membangun snapshot 2023 untuk training dan 2024 untuk testing. Variabel protected, tanggal keluar, alasan keluar, serta tenure final dikeluarkan dari fitur keputusan."
            ),
            nbformat.v4.new_code_cell(
                "train, test = split_cohorts(data)\nassert set(train.employee_id).isdisjoint(test.employee_id)\nmetrics = json.loads((root/'reports/metrics.json').read_text())\npd.DataFrame(metrics['models']).T"
            ),
            nbformat.v4.new_markdown_cell(
                "## Survival dan asumsi\n\nSaya memperlakukan semua exit sebagai event dan akhir 2024 sebagai censoring administratif. Cox memerlukan proportional hazards. Saya melaporkan tes Schoenfeld residual dengan transformasi rank dan koreksi Bonferroni."
            ),
            nbformat.v4.new_code_cell("pd.read_csv(root/'reports/cox_ph_diagnostics.csv')"),
            nbformat.v4.new_code_cell(
                "from IPython.display import Image, display\ndisplay(Image(filename=str(root/'assets/survival.png')))\ndisplay(Image(filename=str(root/'assets/evaluation.png')))"
            ),
            nbformat.v4.new_markdown_cell(
                "## Interpretasi dan batas\n\nSaya memakai risiko untuk mendukung percakapan retensi sukarela. Perubahan probabilitas terhadap nilai acuan bukan efek kausal dan bukan alasan pemutusan kerja. Simulasi penghematan mengasumsikan exit berkurang 20%, bukan hasil intervensi yang sudah terbukti."
            ),
        ]
    )
    notebook.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
    # Saya menjalankan kernel dengan interpreter yang sama dengan pipeline.
    from jupyter_client import KernelManager

    class RuntimeKernelManager(KernelManager):
        def format_kernel_cmd(self, extra_arguments=None):
            cmd = super().format_kernel_cmd(extra_arguments)
            cmd[0] = sys.executable
            return cmd

    client = NotebookClient(notebook, timeout=240, kernel_manager_class=RuntimeKernelManager, resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, ROOT / "notebooks/01_hr_attrition.ipynb")


def run():
    for folder in ["data", "reports", "assets"]:
        (ROOT / folder).mkdir(exist_ok=True)
    frame = generate()
    frame.to_csv(ROOT / "data/employees_simulated.csv", index=False, date_format="%Y-%m-%d")
    monthly, annual = turnover(frame)
    monthly.to_csv(ROOT / "reports/monthly_headcount.csv", index=False)
    annual.to_csv(ROOT / "reports/annual_turnover.csv", index=False)
    survival = survival_analysis(frame)
    models, metrics, selected, train, test = fit_models(frame)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    predictions = test[["employee_id", "target"]].copy()
    for name, model in models.items():
        p = model.predict_proba(test[FEATURES])[:, 1]
        predictions[name] = p
        RocCurveDisplay.from_predictions(test.target, p, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(test.target, p, name=name, ax=axes[1])
        observed, predicted = calibration_curve(test.target, p, n_bins=6, strategy="quantile")
        axes[2].plot(predicted, observed, "o-", label=name)
    axes[1].axhline(test.target.mean(), color="grey", linestyle="--", label="Prevalensi")
    axes[1].legend(fontsize=8)
    axes[2].plot([0, 1], [0, 1], "--", color="grey")
    axes[2].legend(fontsize=8)
    axes[2].set(xlabel="Probabilitas prediksi", ylabel="Proporsi aktual", title="Kalibrasi holdout 2024")
    fig.tight_layout()
    fig.savefig(ROOT / "assets/evaluation.png", dpi=150)
    plt.close(fig)
    predictions.to_csv(ROOT / "reports/test_predictions.csv", index=False)
    chosen = models[selected]
    importance = permutation_importance(chosen, test[FEATURES], test.target, scoring="roc_auc", n_repeats=12, random_state=SEED)
    imp = pd.DataFrame(dict(feature=FEATURES, auc_decrease=importance.importances_mean, sd=importance.importances_std)).sort_values(
        "auc_decrease", ascending=False
    )
    imp.to_csv(ROOT / "reports/permutation_importance.csv", index=False)
    coefs = models["Logistic regression"].named_steps["estimator"].coef_[0]
    pd.DataFrame(dict(feature=models["Logistic regression"].named_steps["preprocess"].get_feature_names_out(), coefficient=coefs)).to_csv(
        ROOT / "reports/logistic_coefficients.csv", index=False
    )
    current = snapshot(frame, END, labels=False)
    risks = risk_table(chosen, current, train)
    risks.to_csv(ROOT / "reports/current_risk.csv", index=False)
    # Artefak lokal dibangun ulang; dashboard publik memakai CSV, bukan pickle unduhan.
    joblib.dump(chosen, ROOT / "reports/model.joblib")
    fig, axis = plt.subplots(figsize=(8, 4))
    for dep, group in monthly.groupby("department"):
        axis.plot(pd.to_datetime(group.month), group.headcount, label=dep)
    axis.set(xlabel="Bulan", ylabel="Headcount akhir bulan", title="Kohor rekrutmen 2022: data simulasi")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "assets/headcount.png", dpi=150)
    plt.close(fig)
    last = annual[(annual.year == 2024) & (annual.department != "Semua")].sort_values("turnover_rate", ascending=False).iloc[0]
    exits = frame[frame.exit_date.notna()]
    target_exits = exits[(exits.exit_date.dt.year == 2024) & (exits.department == last.department)]
    costs = dict(
        assumed_salary_months=6,
        total_turnover_cost=float((exits.monthly_salary * 6).sum()),
        year_2024_cost=float((exits.loc[exits.exit_date.dt.year == 2024, "monthly_salary"] * 6).sum()),
        highest_department=str(last.department),
        highest_rate=float(last.turnover_rate),
        targeted_exits=len(target_exits),
        relative_reduction=0.2,
        scenario_saving=float((target_exits.monthly_salary * 6).sum() * 0.2),
    )
    summary = dict(
        seed=SEED,
        employees=len(frame),
        exits=len(exits),
        active=len(current),
        train_n=len(train),
        test_n=len(test),
        train_landmark="2023-01-01",
        test_landmark="2024-01-01",
        selected_model=selected,
        selection_rule="Average precision validation 2023, sebelum evaluasi test 2024",
        models=metrics,
        survival=survival,
        costs=costs,
    )
    (ROOT / "reports/metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = pd.DataFrame(metrics).T[["auc", "average_precision", "brier", "prevalence", "validation_ap"]]
    report = f"""# Laporan analisis attrition: simulasi\n\nSaya memakai seed {SEED} untuk {len(frame)} karyawan yang direkrut selama 2022 dan diamati sampai akhir 2024. Saya memperoleh {len(exits)} exit dan {len(current)} karyawan aktif pada akhir pengamatan. Tidak ada data perusahaan nyata.\n\n## Metrik HR\n\n{annual[annual.department == "Semua"].to_markdown(index=False, floatfmt=".4f")}\n\nSaya memakai rerata headcount harian: employee-days / 365 atau 366, dengan interval kerja [tanggal masuk, tanggal keluar). Turnover sukarela dan tidak sukarela dijumlahkan menjadi total turnover. Headcount akhir bulan tersedia di monthly_headcount.csv dan rincian departemen di annual_turnover.csv. Rerata tenure hanya untuk karyawan yang keluar dan tidak sama dengan expected lifetime seluruh kohor.\n\n## Survival\n\nSaya memakai waktu sejak rekrutmen dengan right censoring administratif akhir 2024. Tidak ada left truncation karena seluruh tanggal masuk tercakup observasi. Semua exit adalah event, sehingga ini bukan model cause-specific sukarela. Log-rank departemen p={survival["logrank_p"]["department"]:.5g}; lembur p={survival["logrank_p"]["overtime"]:.5g}. Concordance Cox in-sample={survival["concordance"]:.4f}, bukan ukuran generalisasi.\n\nTes PH Schoenfeld rank menghasilkan p terkecil {survival["ph_min_p"]:.5g}, dengan {survival["ph_rejections_bonferroni"]} penolakan setelah koreksi Bonferroni. Saya menyajikan setiap p-value di cox_ph_diagnostics.csv. Jika asumsi ditolak, satu hazard ratio konstan harus dibaca hati-hati dan model time-varying atau stratifikasi perlu diuji sebelum penggunaan. Koefisien, hazard ratio, CI dan p-value tersedia di cox_coefficients.csv.\n\n![Survival](../assets/survival.png)\n\n## Prediksi keluar 12 bulan\n\nSaya memisahkan identitas dengan rasio 70:30, membangun snapshot training 1 Januari 2023 (n={len(train)}) dan test 1 Januari 2024 (n={len(test)}). Karyawan yang sudah keluar dikeluarkan sebelum snapshot. Training dibagi lagi 75:25 secara stratified untuk memilih model dari average precision validation. Saya memilih {selected}, lalu melatih ulang pada seluruh training. Test hanya dipakai untuk pelaporan. Ambang confusion matrix 0,5 ditetapkan sebelumnya.\n\n{table.to_markdown(floatfmt=".4f")}\n\nSaya membandingkan average precision terhadap prevalensi pada test. Brier score mengukur galat probabilitas; kurva kuantil menunjukkan kalibrasi tanpa melakukan calibration fitting di test. Feature engineering tenure memakai tanggal snapshot, promosi mendatang diabaikan. Usia, gender, status menikah, ID, tanggal/alasan keluar dan tenure final tidak menjadi fitur. Departemen dan gaji masih dapat menjadi proxy sehingga pengecualian protected features tidak membuktikan fairness.\n\n![Evaluasi](../assets/evaluation.png)\n\nConfusion matrix [TN, FP; FN, TP]: {json.dumps({k: v["confusion_matrix"] for k, v in metrics.items()})}. Permutation importance tersedia di permutation_importance.csv; saya melaporkan penurunan AUC pada holdout, bukan sebab akibat. Korelasi antarf fitur membatasi interpretasi.\n\n## Daftar risiko dan etika\n\nSaya memberi peringkat karyawan aktif pada 1 Januari 2025 dengan model terpilih tanpa melatih ulang memakai test. Risiko itu memproyeksikan 2025 yang belum diobservasi, sehingga belum ada validasi aktual untuk daftar tersebut. Drivers menunjukkan perubahan probabilitas terhadap median/modus training per fitur secara terpisah, bukan SHAP atau efek intervensi. Semua ID adalah simulasi. Saya memakai daftar untuk diskusi retensi sukarela dan perbaikan kondisi kerja, bukan dasar pemutusan kerja, sanksi, atau penilaian individu.\n\n## Simulasi biaya\n\nSaya mengasumsikan biaya setiap exit sebesar 6 bulan gaji, tanpa membedakan sukarela/tidak sukarela. Estimasi seluruh periode Rp{costs["total_turnover_cost"]:,.0f}; pada 2024 Rp{costs["year_2024_cost"]:,.0f}. Departemen dengan turnover 2024 tertinggi adalah {last.department} ({last.turnover_rate:.2%}, {len(target_exits)} exit). Jika frekuensi exit departemen ini berkurang relatif 20% dengan gaji keluar rata-rata tetap, penghematan ilustratif Rp{costs["scenario_saving"]:,.0f}. Ini asumsi aritmetika, bukan dampak kausal atau penghematan yang pernah dicapai. Biaya program retensi belum dikurangkan.\n\n## Keterbatasan\n\nSaya mensimulasikan kohor rekrutmen, tanpa replacement hire setelah 2022. Gaji dan indikator perilaku adalah pengukuran baseline yang dianggap tetap, sehingga tidak merepresentasikan riwayat HR bulanan sesungguhnya. Evaluasi hanya satu pembagian identitas dan satu transisi tahun, tanpa external validation. Confidence interval performa belum dihitung. Sebelum penggunaan nyata diperlukan persetujuan pekerja, tinjauan proxy fairness, validasi prospektif dan tata kelola akses.\n"""
    (ROOT / "reports/laporan_analisis.md").write_text(report)
    build_notebook()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
