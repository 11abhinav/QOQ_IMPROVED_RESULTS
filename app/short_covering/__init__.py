"""
Short-Covering Early-Ignition Scanner Package.
Contains 2-layer architecture:
1. Layer 1: EOD Positioning Engine (Bhavcopy + Open Interest buildup analysis -> candidate watchlist)
2. Layer 2: Intraday 5m Ignition Engine (Real-time futures price/OI ignition + multi-timeframe confirmation -> early alerts)
"""

__all__ = [
    "fno_universe",
    "fno_contract_resolver",
    "oi_data_service",
    "short_position_detector",
    "short_covering_scanner",
    "short_covering_schema",
    "short_covering_backtester",
]
