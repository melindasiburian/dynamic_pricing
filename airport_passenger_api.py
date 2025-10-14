"""Utility functions for retrieving passenger data for Ngurah Rai (DPS) airport.

This module wraps public aviation APIs so that other parts of the project can
pull near real-time information about international arrivals and derive
estimated passenger counts that are currently inside the terminal.  It is
written with composability in mind so the caller can plug in their own
occupancy heuristics or feed the snapshot into a dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests


DEFAULT_SEAT_CAPACITY: Dict[str, int] = {
    "A320": 180,
    "A321": 210,
    "A330": 290,
    "A350": 325,
    "B737": 189,
    "B737-8": 189,
    "B737-9": 220,
    "B777": 368,
    "B787": 296,
}


@dataclass(frozen=True)
class PassengerSnapshot:
    """Aggregated passenger view for Ngurah Rai international arrivals."""

    timestamp: datetime
    flights: List[Dict[str, Any]]
    estimated_in_terminal: int


class NgurahRaiPassengerAPI:
    """Client that aggregates flight and passenger information for DPS.

    The class relies on the `aviationstack` REST API for near real-time
    arrival information.  Because passenger loads are not exposed by any
    public API, we use a deterministic estimator based on aircraft seat
    capacity and a configurable load factor.
    """

    AVIATIONSTACK_URL = "https://api.aviationstack.com/v1/flights"

    def __init__(
        self,
        aviationstack_key: str,
        seat_capacity_map: Optional[Dict[str, int]] = None,
        default_load_factor: float = 0.82,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not aviationstack_key:
            raise ValueError("An aviationstack API key is required")

        if not (0.0 < default_load_factor <= 1.0):
            raise ValueError("The load factor must be between 0 and 1")

        self._session = session or requests.Session()
        self._aviationstack_key = aviationstack_key
        self._seat_capacity = {
            **DEFAULT_SEAT_CAPACITY,
            **(seat_capacity_map or {}),
        }
        self._default_load_factor = default_load_factor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_international_arrivals(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        """Return structured international arrival data for Ngurah Rai.

        Parameters
        ----------
        limit:
            Maximum number of flights to request from aviationstack.
        """

        params = {
            "access_key": self._aviationstack_key,
            "arr_iata": "DPS",
            "flight_status": "active",
            "limit": limit,
            "sort": "estimated_arrival",
        }
        response = self._session.get(self.AVIATIONSTACK_URL, params=params, timeout=15)
        response.raise_for_status()

        flights = response.json().get("data", [])
        international_arrivals: List[Dict[str, Any]] = []
        for flight in flights:
            departure_country = (flight.get("departure") or {}).get("country")
            if not departure_country or departure_country.lower() == "indonesia":
                continue

            normalized = self._normalize_flight(flight)
            if normalized:
                international_arrivals.append(normalized)

        return international_arrivals

    def estimate_passenger_load(self, flights: Iterable[Dict[str, Any]]) -> int:
        """Estimate the number of passengers in-terminal based on flights.

        The estimator uses aircraft seat capacity heuristics multiplied by the
        configured load factor.  You can subclass or wrap this method if you
        have better data (e.g., from airport operations).
        """

        total = 0
        for flight in flights:
            aircraft_iata = flight.get("aircraft", {}).get("iata")
            aircraft_icao = flight.get("aircraft", {}).get("icao")
            aircraft_key = aircraft_iata or aircraft_icao
            if aircraft_key and aircraft_key in self._seat_capacity:
                capacity = self._seat_capacity[aircraft_key]
            else:
                capacity = self._seat_capacity.get("A320", 180)
            total += int(round(capacity * self._default_load_factor))
        return total

    def get_snapshot(self, *, limit: int = 100) -> PassengerSnapshot:
        """Fetch the latest arrivals and derive an in-terminal passenger count."""

        flights = self.fetch_international_arrivals(limit=limit)
        estimated = self.estimate_passenger_load(flights)
        return PassengerSnapshot(
            timestamp=datetime.now(timezone.utc),
            flights=flights,
            estimated_in_terminal=estimated,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_flight(self, flight: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reduce the aviationstack payload to the fields we care about."""

        arrival_info = flight.get("arrival") or {}
        if not arrival_info.get("airport"):
            return None

        return {
            "airline": (flight.get("airline") or {}).get("name"),
            "flight_iata": flight.get("flight", {}).get("iata") or flight.get("flight", {}).get("number"),
            "aircraft": flight.get("aircraft") or {},
            "departure_airport": (flight.get("departure") or {}).get("airport"),
            "departure_country": (flight.get("departure") or {}).get("country"),
            "scheduled_arrival": arrival_info.get("scheduled"),
            "estimated_arrival": arrival_info.get("estimated"),
            "terminal": arrival_info.get("terminal"),
            "gate": arrival_info.get("gate"),
            "baggage": arrival_info.get("baggage"),
            "status": flight.get("flight_status"),
        }
