"""Streamlit dashboard for daily price recommendations and forecasts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
SERVICE_CATALOG_PATH = PROJECT_DIR / "realdatatest - Sheet3.csv"
MODEL_CANDIDATES: List[Path] = [
    PROJECT_DIR / "dynamic_pricing_model.joblib",
    PROJECT_DIR / "artifacts" / "dynamic_pricing_model.joblib",
]
CATEGORIES = ["entertainment", "experience", "rental", "in_room_service"]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained pricing model."""
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "Model artifact not found. Place 'dynamic_pricing_model.joblib' in the project "
        "root or inside 'artifacts/'."
    )


@st.cache_data(show_spinner=False)
def load_service_catalog() -> pd.DataFrame:
    """Return the catalog containing service/package metadata."""
    if not SERVICE_CATALOG_PATH.exists():
        raise FileNotFoundError(
            "File 'realdatatest - Sheet3.csv' is required to map package names to categories."
        )
    catalog = pd.read_csv(SERVICE_CATALOG_PATH)
    catalog.columns = [col.strip() for col in catalog.columns]
    return catalog


def make_feature_frame(records: List[Dict[str, float]]) -> pd.DataFrame:
    """Create a feature frame that matches the training schema."""
    return pd.DataFrame(records)[
        [
            "category",
            "competitive_price",
            "competitor_count",
            "category_quantity",
            "total_visitors",
            "monthly_event_days",
            "temperature_celsius",
            "prcp_mm",
        ]
    ]


def predict_prices(model, records: List[Dict[str, float]]) -> np.ndarray:
    """Run inference with the trained pipeline."""
    feature_df = make_feature_frame(records)
    return model.predict(feature_df)


def render_recommendation_page(model, catalog: pd.DataFrame) -> None:
    """Display the single-package recommendation experience."""
    st.header("Sistem Rekomendasi Harga Harian")
    st.markdown(
        "Masukkan metrik harian untuk paket yang dipilih. Model akan memberikan rekomendasi "
        "harga berdasarkan data kompetitor dan kondisi eksternal yang Anda input."
    )

    catalog = catalog.copy()
    catalog["display_name"] = (
        catalog["package_name"] + " – " + catalog["service_name"] + " (" + catalog["category"] + ")"
    )
    selected_display = st.selectbox(
        "Pilih Paket", catalog["display_name"].tolist(), index=0 if len(catalog) else None
    )

    selection = catalog.loc[catalog["display_name"] == selected_display].iloc[0]
    st.info(
        f"Kategori paket: **{selection['category']}** — Harga dasar historis: ``{selection['base_price_idr']:,}``"
    )

    with st.form("single_recommendation"):
        st.subheader("Data Harian")
        category_cols = st.columns(3)
        competitive_price = category_cols[0].number_input(
            "Competitive Price (IDR)", min_value=0.0, value=selection["base_price_idr"], step=1000.0
        )
        competitor_count = category_cols[1].number_input(
            "Competitor Count", min_value=0.0, value=5.0, step=1.0
        )
        category_quantity = category_cols[2].number_input(
            "Category Quantity", min_value=0.0, value=10.0, step=1.0
        )

        st.markdown("**Faktor Makro (sama untuk semua kategori pada hari terkait)**")
        macro_cols = st.columns(4)
        total_visitors = macro_cols[0].number_input("Total Visitors", min_value=0.0, value=100000.0, step=1000.0)
        monthly_event_days = macro_cols[1].number_input("Monthly Event Days", min_value=0.0, value=3.0, step=1.0)
        temperature_celsius = macro_cols[2].number_input(
            "Temperature (°C)", min_value=-10.0, value=28.0, step=0.1
        )
        prcp_mm = macro_cols[3].number_input("Precipitation (mm)", min_value=0.0, value=50.0, step=1.0)

        submitted = st.form_submit_button("Dapatkan Rekomendasi")

    if submitted:
        record = {
            "category": selection["category"],
            "competitive_price": competitive_price,
            "competitor_count": competitor_count,
            "category_quantity": category_quantity,
            "total_visitors": total_visitors,
            "monthly_event_days": monthly_event_days,
            "temperature_celsius": temperature_celsius,
            "prcp_mm": prcp_mm,
        }
        prediction = float(predict_prices(model, [record])[0])

        delta_vs_competitor = prediction - competitive_price
        st.success("Rekomendasi harga berhasil dihitung.")
        st.metric("Harga Rekomendasi (IDR)", f"{prediction:,.0f}", f"{delta_vs_competitor:,.0f} vs kompetitor")
        st.markdown("### Detail Fitur")
        st.dataframe(pd.DataFrame([record]), use_container_width=True)


def render_forecast_page(model, catalog: pd.DataFrame) -> None:
    """Display the bulk forecasting page."""
    st.header("Forecasting Harga Harian untuk Seluruh Paket")
    st.markdown(
        "Gunakan halaman ini untuk memasukkan data harian (real-time) dan menghasilkan prediksi "
        "untuk seluruh paket dalam katalog berdasarkan kategorinya masing-masing."
    )

    forecast_date = st.date_input("Tanggal Prediksi")

    st.subheader("Faktor Makro Harian")
    macro_cols = st.columns(4)
    total_visitors = macro_cols[0].number_input("Total Visitors", min_value=0.0, value=100000.0, step=1000.0)
    monthly_event_days = macro_cols[1].number_input("Monthly Event Days", min_value=0.0, value=3.0, step=1.0)
    temperature_celsius = macro_cols[2].number_input("Temperature (°C)", min_value=-10.0, value=28.0, step=0.1)
    prcp_mm = macro_cols[3].number_input("Precipitation (mm)", min_value=0.0, value=50.0, step=1.0)

    st.subheader("Metrik per Kategori")
    st.caption("Isi nilai kompetitor yang relevan untuk masing-masing kategori.")
    default_rows = [
        {"category": cat, "competitive_price": 1_000_000.0, "competitor_count": 5.0, "category_quantity": 10.0}
        for cat in CATEGORIES
    ]
    category_metrics = st.data_editor(
        pd.DataFrame(default_rows),
        num_rows="dynamic",
        use_container_width=True,
        disabled=["category"],
    )

    if st.button("Generate Forecast"):
        if category_metrics.empty:
            st.warning("Harap isi metrik per kategori terlebih dahulu.")
            return

        metrics_map: Dict[str, Dict[str, float]] = {}
        for _, row in category_metrics.iterrows():
            category = str(row.get("category", "")).strip()
            if not category:
                continue
            metrics_map[category] = {
                "competitive_price": float(row.get("competitive_price", 0.0)),
                "competitor_count": float(row.get("competitor_count", 0.0)),
                "category_quantity": float(row.get("category_quantity", 0.0)),
            }

        missing_categories = [cat for cat in catalog["category"].unique() if cat not in metrics_map]
        if missing_categories:
            st.error(
                "Nilai untuk kategori berikut belum diisi: " + ", ".join(sorted(set(missing_categories)))
            )
            return

        records: List[Dict[str, float]] = []
        result_rows: List[Dict[str, float]] = []
        for _, svc in catalog.iterrows():
            category = svc["category"]
            category_data = metrics_map[category]
            record = {
                "category": category,
                "competitive_price": category_data["competitive_price"],
                "competitor_count": category_data["competitor_count"],
                "category_quantity": category_data["category_quantity"],
                "total_visitors": total_visitors,
                "monthly_event_days": monthly_event_days,
                "temperature_celsius": temperature_celsius,
                "prcp_mm": prcp_mm,
            }
            records.append(record)
            result_rows.append(
                {
                    "date": forecast_date,
                    "service_name": svc["service_name"],
                    "package_name": svc["package_name"],
                    "category": category,
                    "base_price_idr": svc["base_price_idr"],
                }
            )

        predictions = predict_prices(model, records)
        for idx, price in enumerate(predictions):
            result_rows[idx]["recommended_price"] = float(price)
            result_rows[idx]["delta_vs_base"] = float(price) - float(result_rows[idx]["base_price_idr"])

        result_df = pd.DataFrame(result_rows)
        st.success("Forecast harian berhasil dibuat.")
        st.dataframe(result_df, use_container_width=True)

        summary = (
            result_df.groupby("category")["recommended_price"].agg(["count", "mean", "min", "max"]).rename(columns={
                "count": "jumlah_paket",
                "mean": "harga_rata_rata",
                "min": "harga_minimum",
                "max": "harga_maksimum",
            })
        )
        st.markdown("### Ringkasan per Kategori")
        st.dataframe(summary, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Unduh Hasil (CSV)", data=csv, file_name=f"forecast_{forecast_date}.csv", mime="text/csv"
        )


def main() -> None:
    st.set_page_config(page_title="Dynamic Pricing Dashboard", layout="wide")
    model = load_model()
    catalog = load_service_catalog()

    page = st.sidebar.radio("Pilih Halaman", ["Rekomendasi", "Forecasting"])
    st.sidebar.success("Model siap digunakan untuk perhitungan real-time.")

    if page == "Rekomendasi":
        render_recommendation_page(model, catalog)
    else:
        render_forecast_page(model, catalog)


if __name__ == "__main__":
    main()
