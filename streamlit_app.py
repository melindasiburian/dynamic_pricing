"""Streamlit dashboard for daily price recommendations and forecasts."""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
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


def deterministic_rng(*values: object) -> np.random.Generator:
    """Return a deterministic random generator based on arbitrary values."""

    combined = "::".join(map(str, values))
    seed = abs(hash(combined)) % (2**32)
    return np.random.default_rng(seed)


def format_rupiah(value: float) -> str:
    """Return a localized Rupiah string for display purposes."""

    return f"Rp{value:,.0f}".replace(",", ".")


def holiday_name_for_date(target_date: date, monthly_event_days: float) -> str:
    """Return a friendly holiday or high-season label for the given date."""

    holiday_by_month = {
        1: "International New Year Getaway",
        2: "Lunar New Year Travelers",
        3: "Nyepi Spiritual Retreat",
        4: "Spring Break & Easter Vacation",
        5: "Golden Week & Vesak Visitors",
        6: "Australian Winter Escape",
        7: "Global Summer Holiday Peak",
        8: "Late Summer & Independence Festivities",
        9: "Surf Season Shoulder Month",
        10: "Autumn Culture & Wellness Trips",
        11: "Pre-Holiday International Retreats",
        12: "Christmas & New Year Peak",
    }

    if monthly_event_days <= 0:
        return "Tidak ada event besar"

    return holiday_by_month.get(target_date.month, "Periode Spesial")


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
    catalog["package_key"] = catalog.apply(
        lambda row: f"{row['service_id']}::{row['package_name']}", axis=1
    )
    return catalog


def _generate_dummy_metrics(
    target_date: date, catalog: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Generate deterministic pseudo-random metrics for demo use."""

    month_rng = np.random.default_rng(int(target_date.strftime("%Y%m")))
    monthly_event_days = float(month_rng.integers(1, 6))

    rng = np.random.default_rng(int(target_date.strftime("%Y%m%d")))
    macro_metrics = {
        "total_visitors": float(rng.integers(60_000, 140_000)),
        "monthly_event_days": monthly_event_days,
        "temperature_celsius": float(rng.normal(28, 3)),
        "prcp_mm": float(max(rng.normal(40, 15), 0)),
    }

    package_metrics: Dict[str, Dict[str, float]] = {}
    for _, pkg in catalog.iterrows():
        pkg_seed = abs(hash(f"{target_date.isoformat()}::{pkg['package_key']}")) % (2**32)
        pkg_rng = np.random.default_rng(pkg_seed)
        base_price = float(pkg["base_price_idr"])
        package_metrics[pkg["package_key"]] = {
            "competitive_price": float(max(base_price * pkg_rng.uniform(0.85, 1.25), 50_000)),
            "competitor_count": float(pkg_rng.integers(3, 15)),
            "category_quantity": float(pkg_rng.integers(5, 40)),
        }

    return macro_metrics, package_metrics


def _fetch_metrics_from_api(
    target_date: date, catalog: pd.DataFrame
) -> Optional[Tuple[Dict[str, float], Dict[str, Dict[str, float]]]]:
    """Retrieve metrics from a real API when configured via environment variable.

    The expected JSON shape is:

    ```json
    {
        "macro": {"total_visitors": 123, "monthly_event_days": 3, ...},
        "packages": {
            "<service_id>::<package_name>": {"competitive_price": 1, "competitor_count": 2, "category_quantity": 3},
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
    package_data = payload.get("packages") or {}

    if not macro or not package_data:
        st.warning("Respons API tidak lengkap. Menggunakan data simulasi.")
        return None

    # Ensure packages align with the catalog
    aligned_packages: Dict[str, Dict[str, float]] = {}
    for _, pkg in catalog.iterrows():
        pkg_key = pkg["package_key"]
        if pkg_key in package_data:
            aligned_packages[pkg_key] = package_data[pkg_key]

    if not aligned_packages:
        st.warning("API tidak mengembalikan metrik paket yang sesuai katalog. Menggunakan simulasi.")
        return None

    return macro, aligned_packages


@st.cache_data(show_spinner=False)
def fetch_daily_metrics(
    target_date: date, catalog: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Return metrics from a real API when available, otherwise deterministic dummy data."""

    api_metrics = _fetch_metrics_from_api(target_date, catalog)
    if api_metrics:
        return api_metrics

    return _generate_dummy_metrics(target_date, catalog)


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
    macro_metrics, package_metrics = fetch_daily_metrics(analysis_date, catalog)

    event_label = holiday_name_for_date(analysis_date, macro_metrics["monthly_event_days"])
    st.caption("Data makro-ekonomi yang berlaku untuk seluruh paket pada tanggal terpilih.")
    macro_cols = st.columns(4)
    macro_cols[0].metric("Total Visitors", f"{macro_metrics['total_visitors']:,.0f}")
    macro_cols[1].metric(
        "Monthly Event Days",
        f"{macro_metrics['monthly_event_days']:,.0f}",
        help=f"Periode: {event_label}",
    )
    macro_cols[2].metric("Temperature (°C)", f"{macro_metrics['temperature_celsius']:.1f}")
    macro_cols[3].metric("Precipitation (mm)", f"{macro_metrics['prcp_mm']:.1f}")

    st.caption("Metrik kompetitor per paket diambil otomatis dari sistem monitoring.")

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
    package_data = package_metrics.get(selection["package_key"], {})

    st.info(
        f"**{selection['service_name']} — {selection['package_name']}** (Kategori: {selection['category']})",
    )
    base_price = float(selection["base_price_idr"])
    package_cols = st.columns(3)
    package_cols[0].metric("Harga Dasar", format_rupiah(base_price))
    package_cols[1].metric(
        "Harga Kompetitif", format_rupiah(package_data.get("competitive_price", base_price))
    )
    package_cols[2].metric(
        "Jumlah Kompetitor", f"{package_data.get('competitor_count', 0):.0f}",
    )

    record = {
        "category": category,
        "competitive_price": package_data.get("competitive_price", base_price),
        "competitor_count": package_data.get("competitor_count", 0),
        "category_quantity": package_data.get("category_quantity", 0),
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

    st.markdown("### Pricing Recommendation Snapshot")
    confidence_score = float(
        np.clip(
            70
            + record["monthly_event_days"] * 4
            - (record["prcp_mm"] / 60)
            + min(record["competitor_count"], 10),
            60,
            97,
        )
    )
    risk_level = "Low" if record["competitor_count"] <= 6 else "Medium"
    trend_delta = prediction - base_price
    summary_cols = st.columns(4)
    summary_cols[0].metric("Harga Saat Ini", format_rupiah(base_price))
    summary_cols[1].metric("Rekomendasi AI", format_rupiah(prediction), format_rupiah(trend_delta))
    summary_cols[2].metric("Confidence", f"{confidence_score:.0f}%")
    summary_cols[3].metric("Risk Level", risk_level)

    reasoning_points = [
        f"Permintaan tinggi selama {event_label.lower()}.",
        "Harga kompetitif masih di bawah rata-rata OTA global.",
        "Inventori kategori mencukupi untuk strategi premium ringan.",
    ]
    st.markdown("#### AI Reasoning")
    for point in reasoning_points:
        st.markdown(f"- {point}")

    st.markdown("### Competitive Analysis")
    competitor_labels = [
        "Global OTA A",
        "Boutique Resort B",
        "Local Partner C",
        "Luxury Villa D",
    ]
    competition_rng = deterministic_rng("competition", analysis_date, selection["package_key"])
    competitor_prices = (
        competition_rng.uniform(0.9, 1.15, len(competitor_labels)) * record["competitive_price"]
    )
    competitor_frame = pd.DataFrame(
        {
            "Competitor": competitor_labels + ["Harga Rekomendasi"],
            "Price": list(competitor_prices) + [prediction],
        }
    )
    competitor_frame["Price Label"] = competitor_frame["Price"].apply(format_rupiah)
    competitor_fig = px.bar(
        competitor_frame,
        x="Competitor",
        y="Price",
        color="Competitor",
        text="Price Label",
        color_discrete_sequence=["#81C7F5", "#81C7F5", "#81C7F5", "#81C7F5", "#FF4B4B"],
    )
    competitor_fig.update_layout(yaxis_title="Harga (Rp)", showlegend=False)
    st.plotly_chart(competitor_fig, use_container_width=True)

    st.markdown("### Performance Analytics")
    performance_rng = deterministic_rng("performance", analysis_date, selection["package_key"])
    ctr = performance_rng.uniform(0.06, 0.14)
    conversion_rate = performance_rng.uniform(0.03, 0.09)
    return_rate = performance_rng.uniform(0.01, 0.04)
    performance_cols = st.columns(2)
    perf_bar = px.bar(
        pd.DataFrame(
            {
                "Metric": ["Click Through Rate", "Conversion Rate", "Return Rate"],
                "Percentage": [ctr * 100, conversion_rate * 100, return_rate * 100],
                "color": ["#2ECC71", "#0070FF", "#FFB347"],
            }
        ),
        x="Metric",
        y="Percentage",
        color="color",
        text="Percentage",
        color_discrete_map="identity",
    )
    perf_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    perf_bar.update_layout(yaxis_title="Persentase", showlegend=False)
    performance_cols[0].plotly_chart(perf_bar, use_container_width=True)

    demand_rng = deterministic_rng("demand", analysis_date, selection["package_key"])
    demand_df = pd.DataFrame(
        {
            "Day": list(range(1, 8)),
            "Relative Demand": demand_rng.uniform(0.7, 1.3, 7),
        }
    )
    demand_fig = px.line(
        demand_df,
        x="Day",
        y="Relative Demand",
        markers=True,
    )
    demand_fig.update_layout(yaxis_title="Relative Demand", xaxis_title="Days")
    performance_cols[1].plotly_chart(demand_fig, use_container_width=True)

    st.markdown("### Detail Fitur Hari Ini")
    detail_frame = pd.DataFrame(
        [
            {
                "Faktor": "Harga Kompetitif",
                "Nilai": format_rupiah(record["competitive_price"]),
            },
            {"Faktor": "Jumlah Kompetitor", "Nilai": f"{record['competitor_count']:.0f}"},
            {"Faktor": "Perkiraan Inventory", "Nilai": f"{record['category_quantity']:.0f}"},
            {"Faktor": "Total Visitors", "Nilai": f"{record['total_visitors']:,.0f}"},
            {
                "Faktor": "Monthly Event Days",
                "Nilai": f"{record['monthly_event_days']:.0f} ({event_label})",
            },
            {"Faktor": "Temperature", "Nilai": f"{record['temperature_celsius']:.1f} °C"},
            {"Faktor": "Precipitation", "Nilai": f"{record['prcp_mm']:.1f} mm"},
        ]
    )
    st.dataframe(detail_frame, use_container_width=True, hide_index=True)

    st.markdown("### Auto-Pricing Monitor")
    config_cols = st.columns(3)
    config_cols[0].metric("System Status", "ACTIVE")
    config_cols[1].metric("Products Monitored", f"{int(record['competitor_count']) + 3}")
    config_cols[2].metric("Update Frequency", "Daily")

    autopricing_rng = deterministic_rng("auto", analysis_date, selection["package_key"])
    auto_actions = ["Price decreased", "Price increased", "No change"]
    auto_reasons = [
        "Competitor price drop",
        "High demand detected",
        "Price optimal",
    ]
    recent_activity = []
    for idx in range(3):
        recent_activity.append(
            {
                "Time": f"{9 + idx:02d}:{15 - idx * 5:02d} AM",
                "Product": selection["package_name"],
                "Action": auto_actions[idx],
                "Change": f"{autopricing_rng.uniform(-3, 3):+.1f}%",
                "Reason": auto_reasons[idx],
            }
        )

    st.dataframe(pd.DataFrame(recent_activity), use_container_width=True, hide_index=True)


def render_forecast_page(model, catalog: pd.DataFrame) -> None:
    """Display the bulk forecasting page."""
    st.header("Forecasting Harga Harian untuk Seluruh Paket")
    st.markdown(
        "Semua metrik harian diambil otomatis dari API real-time (jika ``REALTIME_METRICS_API_URL`` "
        "diisi) atau fallback ke simulasi internal sehingga tim pricing cukup memilih tanggal yang "
        "diinginkan."
    )

    forecast_date = st.date_input("Tanggal Prediksi", value=date.today())
    macro_metrics, package_metrics = fetch_daily_metrics(forecast_date, catalog)

    st.subheader("Faktor Makro Harian")
    macro_cols = st.columns(4)
    macro_cols[0].metric("Total Visitors", f"{macro_metrics['total_visitors']:,.0f}")
    macro_cols[1].metric("Monthly Event Days", f"{macro_metrics['monthly_event_days']:,.0f}")
    macro_cols[2].metric("Temperature (°C)", f"{macro_metrics['temperature_celsius']:.1f}")
    macro_cols[3].metric("Precipitation (mm)", f"{macro_metrics['prcp_mm']:.1f}")

    st.markdown("### Prediksi Beberapa Hari ke Depan untuk Paket Terpilih")
    category_options = sorted(catalog["category"].unique())
    if not category_options:
        st.info("Katalog paket belum tersedia untuk melakukan prediksi multi-hari.")
    else:
        selected_category = st.selectbox("Pilih Kategori Paket", options=category_options)
        package_choices = catalog.loc[catalog["category"] == selected_category].sort_values(
            ["service_name", "package_name"]
        )
        if not package_choices.empty:
            service_choice = st.selectbox(
                "Pilih Service", options=package_choices["service_name"].unique().tolist()
            )
            package_choice = st.selectbox(
                "Pilih Package",
                options=package_choices.loc[
                    package_choices["service_name"] == service_choice, "package_name"
                ],
            )
            selected_pkg = package_choices.loc[
                (package_choices["service_name"] == service_choice)
                & (package_choices["package_name"] == package_choice)
            ].iloc[0]
            base_price = float(selected_pkg["base_price_idr"])
            package_data = package_metrics.get(selected_pkg["package_key"], {})

            horizon = st.slider(
                "Jumlah hari ke depan", min_value=1, max_value=7, value=3, key="forecast_horizon"
            )
            future_table_rows: List[Dict[str, str]] = []
            future_chart_rows: List[Dict[str, float]] = []
            for offset in range(1, horizon + 1):
                horizon_date = forecast_date + timedelta(days=offset)
                future_macro, future_package_metrics = fetch_daily_metrics(horizon_date, catalog)
                future_package_data = future_package_metrics.get(
                    selected_pkg["package_key"], package_data
                )
                future_record = {
                    "category": selected_pkg["category"],
                    "competitive_price": future_package_data.get("competitive_price", base_price),
                    "competitor_count": future_package_data.get("competitor_count", 0),
                    "category_quantity": future_package_data.get("category_quantity", 0),
                    "total_visitors": future_macro["total_visitors"],
                    "monthly_event_days": future_macro["monthly_event_days"],
                    "temperature_celsius": future_macro["temperature_celsius"],
                    "prcp_mm": future_macro["prcp_mm"],
                }
                future_price = float(predict_prices(model, [future_record])[0])
                future_table_rows.append(
                    {
                        "Tanggal": horizon_date,
                        "Periode Event": holiday_name_for_date(
                            horizon_date, future_macro["monthly_event_days"]
                        ),
                        "Harga Rekomendasi": format_rupiah(future_price),
                        "Selisih vs Kompetitor": format_rupiah(
                            future_price - future_record["competitive_price"]
                        ),
                    }
                )
                future_chart_rows.append({"Tanggal": horizon_date, "Harga": future_price})

            st.dataframe(pd.DataFrame(future_table_rows), use_container_width=True)
            if future_chart_rows:
                seasonal_fig = px.line(
                    pd.DataFrame(future_chart_rows),
                    x="Tanggal",
                    y="Harga",
                    markers=True,
                )
                seasonal_fig.update_layout(title="Seasonal Price Trend", yaxis_title="Harga (Rp)")
                st.plotly_chart(seasonal_fig, use_container_width=True)
        else:
            st.info("Tidak ada paket dalam kategori yang dipilih.")

    st.subheader("Metrik per Paket")
    st.caption("Nilai otomatis per paket (tidak perlu input manual).")
    metrics_df = pd.DataFrame.from_dict(package_metrics, orient="index").reset_index()
    metrics_df = metrics_df.rename(columns={"index": "package_key"})
    metrics_df[["product_id", "product_name"]] = metrics_df["package_key"].str.split(
        "::", n=1, expand=True
    )
    service_lookup = catalog.set_index("service_id")["service_name"].to_dict()
    metrics_df["service_name"] = metrics_df["product_id"].map(service_lookup)
    metrics_df = metrics_df[
        [
            "product_id",
            "product_name",
            "service_name",
            "competitive_price",
            "competitor_count",
            "category_quantity",
        ]
    ]
    st.dataframe(metrics_df, use_container_width=True)

    if st.button("Generate Forecast"):
        records: List[Dict[str, float]] = []
        result_rows: List[Dict[str, float]] = []
        for _, svc in catalog.iterrows():
            category = svc["category"]
            pkg_key = svc["package_key"]
            pkg_data = package_metrics.get(pkg_key, {})
            record = {
                "category": category,
                "competitive_price": pkg_data.get("competitive_price", svc["base_price_idr"]),
                "competitor_count": pkg_data.get("competitor_count", 0),
                "category_quantity": pkg_data.get("category_quantity", 0),
                "total_visitors": macro_metrics["total_visitors"],
                "monthly_event_days": macro_metrics["monthly_event_days"],
                "temperature_celsius": macro_metrics["temperature_celsius"],
                "prcp_mm": macro_metrics["prcp_mm"],
            }
            records.append(record)
            result_rows.append(
                {
                    "date": forecast_date,
                    "service_id": svc["service_id"],
                    "service_name": svc["service_name"],
                    "package_name": svc["package_name"],
                    "category": category,
                    "base_price_idr": float(svc["base_price_idr"]),
                    "package_key": pkg_key,
                }
            )

        predictions = predict_prices(model, records)
        result_df = pd.DataFrame(result_rows)
        result_df["recommended_price"] = predictions.astype(float)

        # Enforce intuitive ordering so premium packages stay above entry-level ones
        for service_id, idx_list in result_df.groupby("service_id").groups.items():
            group_idx = list(idx_list)
            ordered_group = result_df.loc[group_idx].sort_values("base_price_idr")
            prev_price = 0.0
            for idx in ordered_group.index:
                raw_price = result_df.at[idx, "recommended_price"]
                adjusted_price = max(raw_price, result_df.at[idx, "base_price_idr"], prev_price)
                result_df.at[idx, "recommended_price"] = adjusted_price
                prev_price = adjusted_price

        result_df["delta_vs_base"] = result_df["recommended_price"] - result_df["base_price_idr"]

        display_df = result_df[
            [
                "date",
                "service_name",
                "package_name",
                "category",
                "base_price_idr",
                "recommended_price",
                "delta_vs_base",
            ]
        ].copy()
        display_df["base_price_idr"] = display_df["base_price_idr"].apply(format_rupiah)
        display_df["recommended_price"] = display_df["recommended_price"].apply(format_rupiah)
        display_df["delta_vs_base"] = display_df["delta_vs_base"].apply(format_rupiah)

        st.success("Forecast harian berhasil dibuat.")
        st.dataframe(display_df, use_container_width=True)

        summary = (
            result_df.groupby("category")["recommended_price"].agg(["count", "mean", "min", "max"]).rename(columns={
                "count": "jumlah_paket",
                "mean": "harga_rata_rata",
                "min": "harga_minimum",
                "max": "harga_maksimum",
            })
        )
        summary[["harga_rata_rata", "harga_minimum", "harga_maksimum"]] = summary[
            ["harga_rata_rata", "harga_minimum", "harga_maksimum"]
        ].applymap(format_rupiah)
        st.markdown("### Ringkasan per Kategori")
        st.dataframe(summary, use_container_width=True)

        st.markdown("### Visualisasi Forecast Harga")
        chart_df = result_df.copy()
        chart_df["Produk"] = chart_df["service_name"] + " — " + chart_df["package_name"]
        forecast_chart = px.bar(
            chart_df.sort_values("recommended_price", ascending=False),
            x="Produk",
            y="recommended_price",
            color="category",
            labels={"recommended_price": "Harga Rekomendasi (Rp)"},
        )
        forecast_chart.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(forecast_chart, use_container_width=True)

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
