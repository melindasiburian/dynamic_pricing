"""Streamlit dashboard for daily price recommendations and forecasts."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

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


@st.cache_data(show_spinner=False)
def fetch_daily_metrics(target_date: date) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Simulate real-time metrics retrieved from an external API."""

    rng = np.random.default_rng(int(target_date.strftime("%Y%m%d")))
    macro_metrics = {
        "total_visitors": float(rng.integers(60_000, 140_000)),
        "monthly_event_days": float(rng.integers(1, 6)),
        "temperature_celsius": float(rng.normal(28, 3)),
        "prcp_mm": float(max(rng.normal(40, 15), 0)),
    }

    category_metrics: Dict[str, Dict[str, float]] = {}
    for category in CATEGORIES:
        category_metrics[category] = {
            "competitive_price": float(rng.uniform(150_000, 5_000_000)),
            "competitor_count": float(rng.integers(3, 15)),
            "category_quantity": float(rng.integers(5, 40)),
        }

    return macro_metrics, category_metrics


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
        "Data operasional harian diambil secara otomatis dari layanan internal (simulasi API). "
        "Pilih kategori, layanan, dan paket untuk melihat rekomendasi harga terbaru."
    )

    analysis_date = st.date_input("Tanggal Analisis", value=date.today())
    macro_metrics, category_metrics = fetch_daily_metrics(analysis_date)

    st.caption("Data makro-ekonomi yang berlaku untuk seluruh kategori pada tanggal terpilih.")
    macro_cols = st.columns(4)
    macro_cols[0].metric("Total Visitors", f"{macro_metrics['total_visitors']:,.0f}")
    macro_cols[1].metric("Monthly Event Days", f"{macro_metrics['monthly_event_days']:,.0f}")
    macro_cols[2].metric("Temperature (°C)", f"{macro_metrics['temperature_celsius']:.1f}")
    macro_cols[3].metric("Precipitation (mm)", f"{macro_metrics['prcp_mm']:.1f}")

    st.caption("Metrik kompetitor per kategori diambil otomatis dari sistem monitoring.")
    st.dataframe(
        pd.DataFrame.from_dict(category_metrics, orient="index"),
        use_container_width=True,
    )

    category = st.selectbox("Pilih Kategori", options=CATEGORIES)
    service_options = (
        catalog.loc[catalog["category"] == category, "service_name"].drop_duplicates().sort_values().tolist()
    )
    if not service_options:
        st.warning("Tidak ada layanan yang tersedia untuk kategori terpilih.")
        return

    service_name = st.selectbox("Pilih Service", options=service_options)
    package_options = (
        catalog.loc[(catalog["category"] == category) & (catalog["service_name"] == service_name)]
        .sort_values("package_name")
    )
    if package_options.empty:
        st.warning("Tidak ada paket yang tersedia untuk service terpilih.")
        return

    package_name = st.selectbox("Pilih Package", options=package_options["package_name"].tolist())
    selection = package_options.loc[package_options["package_name"] == package_name].iloc[0]

    st.info(
        f"Kategori paket: **{selection['category']}** — Harga dasar historis: ``{selection['base_price_idr']:,}``"
    )

    record = {
        "category": category,
        "competitive_price": category_metrics[category]["competitive_price"],
        "competitor_count": category_metrics[category]["competitor_count"],
        "category_quantity": category_metrics[category]["category_quantity"],
        "total_visitors": macro_metrics["total_visitors"],
        "monthly_event_days": macro_metrics["monthly_event_days"],
        "temperature_celsius": macro_metrics["temperature_celsius"],
        "prcp_mm": macro_metrics["prcp_mm"],
    }
    prediction = float(predict_prices(model, [record])[0])

    st.success("Rekomendasi harga real-time siap digunakan.")
    delta_vs_competitor = prediction - record["competitive_price"]
    st.metric("Harga Rekomendasi Hari Ini (IDR)", f"{prediction:,.0f}", f"{delta_vs_competitor:,.0f} vs kompetitor")

    st.markdown("### Prediksi Beberapa Hari ke Depan")
    horizon = st.slider("Jumlah hari ke depan", min_value=1, max_value=7, value=3)
    future_rows: List[Dict[str, float]] = []
    for offset in range(1, horizon + 1):
        forecast_date = analysis_date + timedelta(days=offset)
        future_macro, future_category_metrics = fetch_daily_metrics(forecast_date)
        future_record = {
            "category": category,
            "competitive_price": future_category_metrics[category]["competitive_price"],
            "competitor_count": future_category_metrics[category]["competitor_count"],
            "category_quantity": future_category_metrics[category]["category_quantity"],
            "total_visitors": future_macro["total_visitors"],
            "monthly_event_days": future_macro["monthly_event_days"],
            "temperature_celsius": future_macro["temperature_celsius"],
            "prcp_mm": future_macro["prcp_mm"],
        }
        future_price = float(predict_prices(model, [future_record])[0])
        future_rows.append(
            {
                "date": forecast_date,
                "recommended_price": future_price,
                "delta_vs_competitor": future_price - future_record["competitive_price"],
            }
        )

    st.dataframe(pd.DataFrame(future_rows), use_container_width=True)

    st.markdown("### Detail Fitur Hari Ini")
    st.dataframe(pd.DataFrame([record]), use_container_width=True)


def render_forecast_page(model, catalog: pd.DataFrame) -> None:
    """Display the bulk forecasting page."""
    st.header("Forecasting Harga Harian untuk Seluruh Paket")
    st.markdown(
        "Semua metrik harian diambil secara otomatis dari simulasi API internal sehingga tim pricing "
        "cukup memilih tanggal yang diinginkan."
    )

    forecast_date = st.date_input("Tanggal Prediksi", value=date.today())
    macro_metrics, category_metrics = fetch_daily_metrics(forecast_date)

    st.subheader("Faktor Makro Harian")
    macro_cols = st.columns(4)
    macro_cols[0].metric("Total Visitors", f"{macro_metrics['total_visitors']:,.0f}")
    macro_cols[1].metric("Monthly Event Days", f"{macro_metrics['monthly_event_days']:,.0f}")
    macro_cols[2].metric("Temperature (°C)", f"{macro_metrics['temperature_celsius']:.1f}")
    macro_cols[3].metric("Precipitation (mm)", f"{macro_metrics['prcp_mm']:.1f}")

    st.subheader("Metrik per Kategori")
    st.caption("Nilai otomatis per kategori (tidak perlu input manual).")
    st.dataframe(pd.DataFrame.from_dict(category_metrics, orient="index"), use_container_width=True)

    if st.button("Generate Forecast"):
        records: List[Dict[str, float]] = []
        result_rows: List[Dict[str, float]] = []
        for _, svc in catalog.iterrows():
            category = svc["category"]
            category_data = category_metrics[category]
            record = {
                "category": category,
                "competitive_price": category_data["competitive_price"],
                "competitor_count": category_data["competitor_count"],
                "category_quantity": category_data["category_quantity"],
                "total_visitors": macro_metrics["total_visitors"],
                "monthly_event_days": macro_metrics["monthly_event_days"],
                "temperature_celsius": macro_metrics["temperature_celsius"],
                "prcp_mm": macro_metrics["prcp_mm"],
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
