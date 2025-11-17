"""Streamlit dashboard for daily price recommendations and forecasts."""
from __future__ import annotations

import os
from datetime import date, timedelta
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
SERVICE_CATALOG_PATH = PROJECT_DIR / "realdatatest - Sheet3.csv"
MODEL_CANDIDATES: List[Path] = [
    PROJECT_DIR / "dynamic_pricing_model.joblib",
    PROJECT_DIR / "artifacts" / "dynamic_pricing_model.joblib",
]
CATEGORIES = ["entertainment", "experience", "rental", "in_room_service"]
REALTIME_API_URL_ENV = "REALTIME_METRICS_API_URL"

HOLIDAY_CANDIDATES = {
    1: ["Tahun Baru Masehi", "Imlek", "Cuti Bersama Imlek"],
    2: ["Isra Mi'raj", "Hari Valentine"],
    3: ["Nyepi", "Hari Raya Nyepi", "Hari Perempuan"],
    4: ["Waisak", "Paskah", "Jumat Agung"],
    5: ["Hari Buruh", "Kenaikan Isa Almasih", "Hari Pendidikan Nasional"],
    6: ["Hari Lahir Pancasila", "Idul Adha", "Cuti Bersama Idul Adha"],
    7: ["Idul Adha", "Tahun Baru Hijriah", "Libur Sekolah"],
    8: ["Hari Kemerdekaan RI", "Libur Panjang Musim Panas"],
    9: ["Maulid Nabi", "Libur Sekolah"],
    10: ["Maulid Nabi", "Cuti Bersama Maulid"],
    11: ["Hari Pahlawan", "Cuti Bersama Akhir Tahun"],
    12: ["Natal", "Cuti Bersama Natal", "Tahun Baru"] ,
}


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


def format_rupiah(value: float) -> str:
    """Format number as Rupiah with dot separators."""

    try:
        rounded = round(float(value))
    except (TypeError, ValueError):
        return "Rp 0"
    return f"Rp {rounded:,.0f}".replace(",", ".")


def get_holiday_names(target_date: date, event_count: float) -> List[str]:
    """Return deterministic holiday names for the target month based on requested count."""

    month_holidays = HOLIDAY_CANDIDATES.get(target_date.month, [])
    if not month_holidays or event_count <= 0:
        return []

    rng = np.random.default_rng(int(target_date.strftime("%Y%m")))
    choices = rng.choice(month_holidays, size=min(int(event_count), len(month_holidays)), replace=False)
    return sorted(set(choices.tolist()))


def _extract_duration(name: str) -> Optional[str]:
    lower = name.lower()
    for label in ("daily", "weekly", "monthly"):
        if label in lower:
            return label
    return None


def _extract_pax(name: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*pax", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_tier(name: str) -> Optional[str]:
    tiers = [
        "royal",
        "platinum",
        "premium",
        "deluxe",
        "standard",
        "basic",
        "economy",
    ]
    lower = name.lower()
    for tier in tiers:
        if tier in lower:
            return tier
    return None


def apply_pricing_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust recommended prices so they follow intuitive hierarchies within each service."""

    adjusted = df.copy()
    duration_rank = {"daily": 0, "weekly": 1, "monthly": 2}
    tier_rank = {
        "royal": 5,
        "platinum": 4,
        "premium": 3,
        "deluxe": 2,
        "standard": 1,
        "basic": 0,
        "economy": -1,
    }

    for service_name, group_idx in adjusted.groupby("service_name").groups.items():
        subset = adjusted.loc[group_idx]

        # Enforce daily < weekly < monthly for rentals or duration-marked packages
        duration_subset = subset.assign(duration=subset["package_name"].apply(_extract_duration)).dropna(subset=["duration"])
        if len(duration_subset) > 1:
            ordered = duration_subset.sort_values("duration", key=lambda s: s.map(duration_rank))
            prev_price = None
            for _, row in ordered.iterrows():
                idx = row.name
                base = float(adjusted.at[idx, "base_price_idr"])
                recommended = float(adjusted.at[idx, "recommended_price"])
                minimum_increment = max(base * 0.03, 25_000)
                if prev_price is not None and recommended <= prev_price:
                    recommended = prev_price + minimum_increment
                adjusted.at[idx, "recommended_price"] = recommended
                prev_price = recommended

        # Enforce fewer pax > more pax
        pax_subset = subset.assign(pax=subset["package_name"].apply(_extract_pax)).dropna(subset=["pax"])
        if len(pax_subset) > 1:
            ordered = pax_subset.sort_values("pax")
            prev_price = None
            for _, row in ordered.iterrows():
                idx = row.name
                base = float(adjusted.at[idx, "base_price_idr"])
                recommended = float(adjusted.at[idx, "recommended_price"])
                minimum_gap = max(base * 0.02, 15_000)
                if prev_price is not None and recommended >= prev_price:
                    recommended = max(prev_price - minimum_gap, minimum_gap)
                prev_price = recommended
                adjusted.at[idx, "recommended_price"] = recommended

        # Enforce tier hierarchy: premium > deluxe > basic, etc.
        tier_subset = subset.assign(tier=subset["package_name"].apply(_extract_tier)).dropna(subset=["tier"])
        if len(tier_subset) > 1:
            ordered = tier_subset.sort_values("tier", key=lambda s: s.map(tier_rank), ascending=False)
            prev_price = None
            for _, row in ordered.iterrows():
                idx = row.name
                base = float(adjusted.at[idx, "base_price_idr"])
                recommended = float(adjusted.at[idx, "recommended_price"])
                minimum_gap = max(base * 0.05, 35_000)
                if prev_price is not None and recommended <= prev_price:
                    recommended = prev_price + minimum_gap
                prev_price = recommended
                adjusted.at[idx, "recommended_price"] = recommended

    return adjusted


def _generate_dummy_metrics(target_date: date) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Generate deterministic pseudo-random metrics for demo use."""

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


def _fetch_metrics_from_api(target_date: date) -> Optional[Tuple[Dict[str, float], Dict[str, Dict[str, float]]]]:
    """Retrieve metrics from a real API when configured via environment variable.

    The expected JSON shape is:

    ```json
    {
        "macro": {"total_visitors": 123, "monthly_event_days": 3, ...},
        "category": {
            "entertainment": {"competitive_price": 1, "competitor_count": 2, "category_quantity": 3},
            ...
        }
    }
    ```
    """

    api_url = os.getenv(REALTIME_API_URL_ENV)
    if not api_url:
        return None

    try:
        response = requests.get(api_url, params={"date": target_date.isoformat()}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        st.warning(f"Gagal mengambil data real-time dari API ({exc}). Menggunakan data simulasi.")
        return None

    payload = response.json()
    macro = payload.get("macro") or {}
    category = payload.get("category") or {}

    if not macro or not category:
        st.warning("Respons API tidak lengkap. Menggunakan data simulasi.")
        return None

    return macro, category


@st.cache_data(show_spinner=False)
def fetch_daily_metrics(target_date: date) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Return metrics from a real API when available, otherwise deterministic dummy data."""

    api_metrics = _fetch_metrics_from_api(target_date)
    if api_metrics:
        return api_metrics

    return _generate_dummy_metrics(target_date)


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
        "Data operasional harian diambil otomatis dari API real-time bila ``REALTIME_METRICS_API_URL`` "
        "diisi. Jika API belum tersedia, sistem otomatis menggunakan data simulasi yang terkontrol."
    )

    analysis_date = st.date_input("Tanggal Analisis", value=date.today())
    macro_metrics, category_metrics = fetch_daily_metrics(analysis_date)

    st.caption("Data makro-ekonomi yang berlaku untuk seluruh kategori pada tanggal terpilih.")
    macro_cols = st.columns(4)
    macro_cols[0].metric("Total Visitors", f"{macro_metrics['total_visitors']:,.0f}")
    holiday_names = get_holiday_names(analysis_date, macro_metrics["monthly_event_days"])
    holiday_label = ", ".join(holiday_names) if holiday_names else "Tidak ada hari libur khusus"
    macro_cols[1].metric("Monthly Event Days", f"{macro_metrics['monthly_event_days']:,.0f} hari")
    macro_cols[2].metric("Temperature (°C)", f"{macro_metrics['temperature_celsius']:.1f}")
    macro_cols[3].metric("Precipitation (mm)", f"{macro_metrics['prcp_mm']:.1f}")
    st.caption(f"Hari libur/agenda bulan ini: {holiday_label}.")

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
        f"Kategori paket: **{selection['category']}** — Harga dasar historis: **{format_rupiah(selection['base_price_idr'])}**"
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
    st.metric(
        "Harga Rekomendasi Hari Ini",
        format_rupiah(prediction),
        f"{format_rupiah(delta_vs_competitor)} vs kompetitor",
    )

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

    future_df = pd.DataFrame(future_rows)
    future_df["recommended_price"] = future_df["recommended_price"].apply(format_rupiah)
    future_df["delta_vs_competitor"] = future_df["delta_vs_competitor"].apply(format_rupiah)
    st.dataframe(future_df, use_container_width=True)

    st.markdown("### Detail Fitur Hari Ini")
    details_df = pd.DataFrame([record])
    details_df["competitive_price"] = details_df["competitive_price"].apply(format_rupiah)
    st.dataframe(details_df, use_container_width=True)


def render_forecast_page(model, catalog: pd.DataFrame) -> None:
    """Display the bulk forecasting page."""
    st.header("Forecasting Harga Harian untuk Seluruh Paket")
    st.markdown(
        "Semua metrik harian diambil otomatis dari API real-time (jika ``REALTIME_METRICS_API_URL`` "
        "diisi) atau fallback ke simulasi internal sehingga tim pricing cukup memilih tanggal yang "
        "diinginkan."
    )

    forecast_date = st.date_input("Tanggal Prediksi", value=date.today())
    macro_metrics, category_metrics = fetch_daily_metrics(forecast_date)

    st.subheader("Faktor Makro Harian")
    macro_cols = st.columns(4)
    macro_cols[0].metric("Total Visitors", f"{macro_metrics['total_visitors']:,.0f}")
    forecast_holidays = get_holiday_names(forecast_date, macro_metrics["monthly_event_days"])
    holiday_label = ", ".join(forecast_holidays) if forecast_holidays else "Tidak ada hari libur khusus"
    macro_cols[1].metric("Monthly Event Days", f"{macro_metrics['monthly_event_days']:,.0f} hari")
    macro_cols[2].metric("Temperature (°C)", f"{macro_metrics['temperature_celsius']:.1f}")
    macro_cols[3].metric("Precipitation (mm)", f"{macro_metrics['prcp_mm']:.1f}")
    st.caption(f"Hari libur/agenda bulan ini: {holiday_label}.")

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

        result_df = apply_pricing_hierarchy(pd.DataFrame(result_rows))
        st.success("Forecast harian berhasil dibuat.")
        display_df = result_df.copy()
        display_df["base_price_idr"] = display_df["base_price_idr"].apply(format_rupiah)
        display_df["recommended_price"] = display_df["recommended_price"].apply(format_rupiah)
        display_df["delta_vs_base"] = display_df["delta_vs_base"].apply(format_rupiah)
        st.dataframe(display_df, use_container_width=True)

        summary = (
            result_df.groupby("category")["recommended_price"].agg(["count", "mean", "min", "max"]).rename(columns={
                "count": "jumlah_paket",
                "mean": "harga_rata_rata",
                "min": "harga_minimum",
                "max": "harga_maksimum",
            })
        )
        st.markdown("### Ringkasan per Kategori")
        summary_display = summary.copy()
        for col in ["harga_rata_rata", "harga_minimum", "harga_maksimum"]:
            summary_display[col] = summary_display[col].apply(format_rupiah)
        st.dataframe(summary_display, use_container_width=True)

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
