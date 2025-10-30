"""Streamlit dashboard for dynamic pricing recommendations."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from dynamic_pricing import get_live_pricing_recommendation


@st.cache_data(show_spinner=False)
def _model_available() -> bool:
    """Return ``True`` if a serialized model is present on disk."""

    candidate_paths = (
        Path("artifacts/dynamic_pricing_model.joblib"),
        Path("dynamic_pricing_model.joblib"),
    )
    return any(path.exists() for path in candidate_paths)


def _render_recommendation(result: Dict[str, object]) -> None:
    """Render the pricing recommendation returned by the inference helper."""

    st.subheader("Recommended Price")
    current_price = result["current_base_price"]
    recommended_price = result["recommended_price"]
    delta_pct = result["delta_pct"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Base price", f"Rp {current_price:,.0f}")
    col2.metric(
        "Recommended", f"Rp {recommended_price:,.0f}", delta=f"{delta_pct:+.2f}%"
    )
    projected_profit = (recommended_price - current_price) * 10
    col3.metric("Projected uplift (10 units)", f"Rp {projected_profit:,.0f}")

    st.markdown("### Context snapshot")
    context = result["context"]
    st.dataframe(pd.DataFrame([context]))

    with st.expander("Show raw features"):
        st.json(result["raw_features"], expanded=False)


def main() -> None:
    st.set_page_config(
        page_title="Dynamic Pricing Dashboard",
        page_icon="💹",
        layout="wide",
    )
    st.title("💹 Dynamic Pricing Recommendation Dashboard")
    st.caption(
        "Real-time pricing suggestions powered by the trained gradient boosting pipeline."
    )

    if not _model_available():
        st.error(
            textwrap.dedent(
                """
                Model artifact not found.
                Ensure ``dynamic_pricing_model.joblib`` exists in the project root or
                under ``artifacts/`` before running the dashboard.
                """
            ).strip()
        )
        return

    with st.form("recommendation_form"):
        st.subheader("Service context")
        service_id = st.text_input("Service ID", value="VIP_TABLE")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=-6.2000, format="%.4f")
        with col2:
            longitude = st.number_input("Longitude", value=106.8167, format="%.4f")
        submitted = st.form_submit_button("Get recommendation")

    if submitted:
        with st.spinner("Computing recommendation..."):
            try:
                result = get_live_pricing_recommendation(
                    service_id=service_id,
                    lat=float(latitude),
                    lon=float(longitude),
                )
            except FileNotFoundError as exc:  # pragma: no cover - surfaced in UI
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.exception(exc)
            else:
                _render_recommendation(result)


if __name__ == "__main__":
    main()
