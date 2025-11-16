# Dynamic Pricing Dashboard

This repository packages the utilities required to generate dynamic pricing
recommendations and now ships with a Streamlit dashboard for quick
experimentation.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r dynamic_pricing/requirements.txt
```

## Running the Streamlit dashboard

Make sure the trained model artifact `dynamic_pricing_model.joblib` is present in
the project root (or inside `artifacts/`). Then launch the dashboard with:

```bash
streamlit run streamlit_app.py
```

The Streamlit app provides two workflows that consume the same real-time
features used during model training:

1. **Halaman Rekomendasi** – pilih paket dari file
   `realdatatest - Sheet3.csv` lalu masukkan data harian seperti `competitive_price`,
   `competitor_count`, `category_quantity`, dan faktor makro (`total_visitors`,
   `monthly_event_days`, `temperature_celsius`, `prcp_mm`). Model random forest
   akan mengembalikan rekomendasi harga untuk paket yang dipilih.
2. **Halaman Forecasting** – masukkan faktor makro harian sekali dan isi tabel
   metrik per kategori (entertainment, experience, rental, in_room_service).
   Aplikasi akan menghitung prediksi harga untuk seluruh paket di katalog dan
   menyediakan ringkasan per kategori beserta opsi unduh CSV.