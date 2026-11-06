# Verifikasi eksekusi

Saya memasang seluruh versi pada requirements.txt dan menjalankan `make all` dari root pada Python 3.12. Lint lulus, 12 unit test lulus, pipeline lengkap selesai dan seluruh sel kode notebook mempunyai execution count tanpa output error.

Saya menjalankan Streamlit AppTest dari root: empat KPI tampil tanpa exception. Filter departemen Penjualan dan tahun 2023 menghasilkan turnover 25,8%, rerata headcount 290,8, 63 exit sukarela dan 12 exit tidak sukarela, sesuai annual_turnover.csv.

Saya merekonsiliasi AUC, average precision, Brier, prevalensi dan confusion matrix di metrics.json dengan test_predictions.csv. Generator, snapshot, pemisahan identitas, horizon label, promosi mendatang dan penyebut employee-days memiliki tes khusus.

Saya mempertahankan hasil simulasi secara transparan: random forest terpilih melalui validation AP, tetapi memiliki AUC test lebih rendah daripada logistic regression dan tidak memprediksi exit pada cutoff 0,5. Proyeksi risiko 2025 belum mempunyai outcome observasi. Pemeriksaan ini tidak mencakup deployment Community Cloud atau pengujian pada data perusahaan nyata.
