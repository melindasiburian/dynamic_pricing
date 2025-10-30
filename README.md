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

The app lets you enter a service identifier together with latitude and longitude
coordinates. It returns the recommended price along with the contextual metrics
used by the model.