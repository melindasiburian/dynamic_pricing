# Dynamic Pricing Toolkit

This repository collects experiments around demand forecasting, reinforcement
learning, and data integrations for travel and ecommerce scenarios.  The code
base is organized as small, focused modules that can be reused in notebooks or
within lightweight Streamlit dashboards.

## Passenger data for Ngurah Rai (DPS)

Airports rarely expose direct passenger-count feeds, but we can combine flight
status APIs with a passenger load heuristic to approximate how many travelers
are inside the terminal.  The new `airport_passenger_api.py` module wraps the
[aviationstack](https://aviationstack.com/) REST API so we can fetch
international arrivals for Ngurah Rai International Airport and derive an
estimated passenger count.

```python
from airport_passenger_api import NgurahRaiPassengerAPI

api = NgurahRaiPassengerAPI(aviationstack_key="YOUR_ACCESS_KEY")
snapshot = api.get_snapshot(limit=50)

print(snapshot.timestamp)
print(f"Flights tracked: {len(snapshot.flights)}")
print(f"Estimated passengers in terminal: {snapshot.estimated_in_terminal}")
```

### What the API returns

Each flight dictionary inside the snapshot contains:

- Airline and flight number
- The origin airport and country
- Scheduled and estimated arrival times
- Terminal, gate, and baggage carousel information when available
- Aircraft metadata that feeds the passenger estimator

### Passenger estimator

By default the estimator multiplies the aircraft seat capacity by an 82% load
factor (based on ICAO global averages).  You can override both the seat
capacity map and the default load factor if you have better operational data.

```python
api = NgurahRaiPassengerAPI(
    aviationstack_key="YOUR_ACCESS_KEY",
    seat_capacity_map={"B787-9": 320},
    default_load_factor=0.88,
)
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

You will also need a valid aviationstack API key.  Free plans work for light
usage; place the key in an environment variable or pass it directly into the
class constructor as shown above.
