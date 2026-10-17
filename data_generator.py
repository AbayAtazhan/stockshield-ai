"""
StockShield AI - Enterprise Retail & Customer Intelligence Dataset Generator
AMD Developer Hackathon MVP

Generates realistic enterprise retail sales history and current inventory snapshots.
Pre-configures realistic retail archetypes:
 - Critical High Stockout Risk items (days_until_stockout < lead_time_days)
 - Overstock / Dead Capital items (excessive inventory, tied-up working capital)
 - High Return Anomaly items (elevated return rate, degraded customer sentiment)
 - Healthy baseline items
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StockShield.DataGenerator")


@dataclass(frozen=True)
class SKUArchetypeConfig:
    """Pre-defined SKU configuration archetype."""
    sku_id: str
    product_name: str
    category: str
    current_stock: int
    target_daily_mean: float
    lead_time_days: int
    unit_cost: float
    unit_price: float
    return_rate_pct: float
    customer_sentiment_score: float
    sales_growth_trend: float = 0.0  # Daily trend coefficient
    archetype_label: str = "HEALTHY"


# Comprehensive curated catalog of 25 SKUs tailored for Retail Tech / AMD Ecosystem
CURATED_SKU_CATALOG: List[SKUArchetypeConfig] = [
    # -------------------------------------------------------------
    # Archetype 1: 5 Critical High Stockout Risk Items (Fast velocity, critically low stock)
    # -------------------------------------------------------------
    SKUArchetypeConfig(
        sku_id="SKU-1001",
        product_name="AMD Ryzen 9 7950X3D Desktop Gaming Processor",
        category="Processors & Computing",
        current_stock=24,
        target_daily_mean=36.5,
        lead_time_days=12,
        unit_cost=430.00,
        unit_price=599.00,
        return_rate_pct=1.8,
        customer_sentiment_score=4.85,
        sales_growth_trend=0.08,
        archetype_label="HIGH_STOCKOUT_RISK",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-1002",
        product_name="AMD Radeon RX 7900 XTX 24GB Flagship GPU",
        category="Graphics & Accelerators",
        current_stock=18,
        target_daily_mean=27.0,
        lead_time_days=14,
        unit_cost=710.00,
        unit_price=979.99,
        return_rate_pct=2.1,
        customer_sentiment_score=4.78,
        sales_growth_trend=0.06,
        archetype_label="HIGH_STOCKOUT_RISK",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-1003",
        product_name="StockShield Low-Latency AI PCIe Accelerator Card",
        category="Processors & Computing",
        current_stock=15,
        target_daily_mean=21.5,
        lead_time_days=10,
        unit_cost=840.00,
        unit_price=1249.00,
        return_rate_pct=1.4,
        customer_sentiment_score=4.92,
        sales_growth_trend=0.10,
        archetype_label="HIGH_STOCKOUT_RISK",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-1004",
        product_name="Ultra-Wide 49-inch Curved QD-OLED 240Hz Monitor",
        category="Displays & Audio",
        current_stock=14,
        target_daily_mean=18.0,
        lead_time_days=9,
        unit_cost=670.00,
        unit_price=949.99,
        return_rate_pct=2.9,
        customer_sentiment_score=4.62,
        sales_growth_trend=0.04,
        archetype_label="HIGH_STOCKOUT_RISK",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-1005",
        product_name="Pro Rapid-Trigger Magnetic Hall-Effect Keyboard",
        category="Peripherals & Accessories",
        current_stock=32,
        target_daily_mean=42.0,
        lead_time_days=8,
        unit_cost=92.00,
        unit_price=169.99,
        return_rate_pct=2.4,
        customer_sentiment_score=4.75,
        sales_growth_trend=0.07,
        archetype_label="HIGH_STOCKOUT_RISK",
    ),

    # -------------------------------------------------------------
    # Archetype 2: 4 Overstock / Dead Capital Items (Massive stock, stagnant sales)
    # -------------------------------------------------------------
    SKUArchetypeConfig(
        sku_id="SKU-2001",
        product_name="Legacy DVI-D to VGA Active Converter Cable (Pack of 5)",
        category="Cables & Power",
        current_stock=920,
        target_daily_mean=0.9,
        lead_time_days=5,
        unit_cost=16.50,
        unit_price=29.99,
        return_rate_pct=4.1,
        customer_sentiment_score=3.10,
        sales_growth_trend=-0.02,
        archetype_label="OVERSTOCK_DEAD_CAPITAL",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-2002",
        product_name="Passive PCIe Dual Aluminum Heat Sink Bracket v1",
        category="Processors & Computing",
        current_stock=1150,
        target_daily_mean=0.6,
        lead_time_days=6,
        unit_cost=12.00,
        unit_price=22.50,
        return_rate_pct=3.7,
        customer_sentiment_score=3.25,
        sales_growth_trend=-0.03,
        archetype_label="OVERSTOCK_DEAD_CAPITAL",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-2003",
        product_name="Wired USB Ergonomic Stationary Trackball Mouse",
        category="Peripherals & Accessories",
        current_stock=640,
        target_daily_mean=1.1,
        lead_time_days=7,
        unit_cost=27.00,
        unit_price=47.99,
        return_rate_pct=4.4,
        customer_sentiment_score=3.35,
        sales_growth_trend=-0.01,
        archetype_label="OVERSTOCK_DEAD_CAPITAL",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-2004",
        product_name="Cat5e Unshielded Industrial Bulk Reel (305m Spool)",
        category="Cables & Power",
        current_stock=480,
        target_daily_mean=0.8,
        lead_time_days=9,
        unit_cost=54.00,
        unit_price=84.99,
        return_rate_pct=2.8,
        customer_sentiment_score=3.40,
        sales_growth_trend=-0.01,
        archetype_label="OVERSTOCK_DEAD_CAPITAL",
    ),

    # -------------------------------------------------------------
    # Archetype 3: 3 High Return Anomaly Items (Severe return spikes, customer churn)
    # -------------------------------------------------------------
    SKUArchetypeConfig(
        sku_id="SKU-3001",
        product_name="Gen4 M.2 NVMe SSD 4TB (Batch-X Thermal Throttling)",
        category="Processors & Computing",
        current_stock=260,
        target_daily_mean=32.0,
        lead_time_days=10,
        unit_cost=175.00,
        unit_price=269.99,
        return_rate_pct=32.4,  # Anomaly: >30% returns!
        customer_sentiment_score=1.85,  # Anomaly: critical drop in reviews
        sales_growth_trend=-0.05,
        archetype_label="HIGH_RETURN_ANOMALY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-3002",
        product_name="Wireless Spatial ANC Headset (BT Drop Firmware Bug)",
        category="Peripherals & Accessories",
        current_stock=210,
        target_daily_mean=25.5,
        lead_time_days=8,
        unit_cost=88.00,
        unit_price=149.99,
        return_rate_pct=28.7,  # Anomaly: 28.7% returns
        customer_sentiment_score=2.08,
        sales_growth_trend=-0.04,
        archetype_label="HIGH_RETURN_ANOMALY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-3003",
        product_name="1000W SFX-L Titanium PSU (Coil Whine Defect)",
        category="Cables & Power",
        current_stock=175,
        target_daily_mean=19.0,
        lead_time_days=11,
        unit_cost=115.00,
        unit_price=194.99,
        return_rate_pct=25.1,  # Anomaly: 25.1% returns
        customer_sentiment_score=2.25,
        sales_growth_trend=-0.03,
        archetype_label="HIGH_RETURN_ANOMALY",
    ),

    # -------------------------------------------------------------
    # Archetype 4: 13 Balanced / Healthy Baseline Retail Items
    # -------------------------------------------------------------
    SKUArchetypeConfig(
        sku_id="SKU-4001",
        product_name="AMD Ryzen 7 7800X3D Gaming Processor",
        category="Processors & Computing",
        current_stock=380,
        target_daily_mean=22.0,
        lead_time_days=9,
        unit_cost=290.00,
        unit_price=399.00,
        return_rate_pct=2.2,
        customer_sentiment_score=4.82,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4002",
        product_name="AMD Radeon RX 7800 XT 16GB Gaming GPU",
        category="Graphics & Accelerators",
        current_stock=290,
        target_daily_mean=16.5,
        lead_time_days=12,
        unit_cost=360.00,
        unit_price=499.99,
        return_rate_pct=2.5,
        customer_sentiment_score=4.65,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4003",
        product_name="27-inch QHD Fast IPS 180Hz Gaming Monitor",
        category="Displays & Audio",
        current_stock=310,
        target_daily_mean=14.0,
        lead_time_days=8,
        unit_cost=185.00,
        unit_price=279.99,
        return_rate_pct=3.1,
        customer_sentiment_score=4.48,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4004",
        product_name="Ultralight Wireless Esports Mouse 4000Hz Polling",
        category="Peripherals & Accessories",
        current_stock=450,
        target_daily_mean=24.0,
        lead_time_days=6,
        unit_cost=52.00,
        unit_price=99.99,
        return_rate_pct=2.7,
        customer_sentiment_score=4.55,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4005",
        product_name="Double-Shot PBT Low-Profile Keycap Set (ANSI)",
        category="Peripherals & Accessories",
        current_stock=280,
        target_daily_mean=11.5,
        lead_time_days=7,
        unit_cost=21.00,
        unit_price=44.99,
        return_rate_pct=2.0,
        customer_sentiment_score=4.38,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4006",
        product_name="Thunderbolt 4 Dual 4K 60Hz 10-in-1 Docking Station",
        category="Cables & Power",
        current_stock=240,
        target_daily_mean=9.5,
        lead_time_days=10,
        unit_cost=110.00,
        unit_price=179.99,
        return_rate_pct=3.4,
        customer_sentiment_score=4.32,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4007",
        product_name="850W ATX 3.0 PCIe 5.0 Gold Fully Modular PSU",
        category="Cables & Power",
        current_stock=330,
        target_daily_mean=13.0,
        lead_time_days=8,
        unit_cost=78.00,
        unit_price=129.99,
        return_rate_pct=2.6,
        customer_sentiment_score=4.45,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4008",
        product_name="360mm ARGB AIO Liquid CPU Cooler Triple Fan",
        category="Processors & Computing",
        current_stock=260,
        target_daily_mean=10.5,
        lead_time_days=9,
        unit_cost=75.00,
        unit_price=124.99,
        return_rate_pct=3.2,
        customer_sentiment_score=4.40,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4009",
        product_name="32GB (2x16GB) DDR5-6000MHz CL30 EXPO Memory Kit",
        category="Processors & Computing",
        current_stock=410,
        target_daily_mean=18.5,
        lead_time_days=7,
        unit_cost=68.00,
        unit_price=109.99,
        return_rate_pct=2.1,
        customer_sentiment_score=4.60,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4010",
        product_name="Braided Sleeved Modular PSU Extension Cable Kit",
        category="Cables & Power",
        current_stock=340,
        target_daily_mean=12.0,
        lead_time_days=6,
        unit_cost=15.00,
        unit_price=32.99,
        return_rate_pct=2.3,
        customer_sentiment_score=4.35,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4011",
        product_name="Monitor ScreenBar Eye-Care LED Lamp Auto-Dimming",
        category="Displays & Audio",
        current_stock=295,
        target_daily_mean=11.0,
        lead_time_days=8,
        unit_cost=34.00,
        unit_price=64.99,
        return_rate_pct=2.8,
        customer_sentiment_score=4.42,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4012",
        product_name="Wi-Fi 7 Tri-Band PCIe Expansion Card with Bluetooth 5.4",
        category="Cables & Power",
        current_stock=230,
        target_daily_mean=8.5,
        lead_time_days=9,
        unit_cost=42.00,
        unit_price=74.99,
        return_rate_pct=2.9,
        customer_sentiment_score=4.28,
        archetype_label="HEALTHY",
    ),
    SKUArchetypeConfig(
        sku_id="SKU-4013",
        product_name="Studio Dynamic Broadcast USB-C / XLR Microphone",
        category="Displays & Audio",
        current_stock=270,
        target_daily_mean=10.0,
        lead_time_days=8,
        unit_cost=58.00,
        unit_price=109.99,
        return_rate_pct=2.4,
        customer_sentiment_score=4.51,
        archetype_label="HEALTHY",
    ),
]


def generate_retail_dataset(
    num_skus: int = 25,
    days: int = 90,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic daily historical retail transaction data and a current
    inventory snapshot DataFrame.

    Parameters:
        num_skus (int): Number of SKUs to generate (default 25, clamped to available catalog).
        days (int): Number of historical days to simulate (default 90).
        random_seed (int): Seed for deterministic, reproducible simulation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_historical: Daily time-series dataset over `days` for each SKU.
            - df_snapshot: Current inventory snapshot with SKU attributes, performance stats,
                           and last 30 days sales trend arrays.
    """
    rng = np.random.default_rng(seed=random_seed)

    # Validate and select SKUs from curated catalog
    selected_configs = CURATED_SKU_CATALOG[:min(num_skus, len(CURATED_SKU_CATALOG))]
    actual_num_skus = len(selected_configs)

    # Reference timeline: ending yesterday relative to today
    end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, periods=days, freq="D")

    # Day of week multiplier (Retail seasonality: Saturday/Sunday peak, Tuesday trough)
    # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
    dow_multipliers = {
        0: 0.95,  # Mon
        1: 0.90,  # Tue
        2: 0.96,  # Wed
        3: 1.02,  # Thu
        4: 1.15,  # Fri
        5: 1.28,  # Sat (peak retail traffic)
        6: 1.22,  # Sun
    }

    history_records: List[Dict[str, Any]] = []

    logger.info(
        f"Simulating retail history: {actual_num_skus} SKUs across {days} days "
        f"({start_date.strftime('%Y-%m-%d')} to {date_range[-1].strftime('%Y-%m-%d')})..."
    )

    for sku_cfg in selected_configs:
        # Base daily mean sales
        base_mean = sku_cfg.target_daily_mean

        for day_idx, current_date in enumerate(date_range):
            dow = current_date.dayofweek
            seasonality = dow_multipliers[dow]

            # Linear trend component (e.g. accelerating demand for high-stockout items)
            trend_factor = 1.0 + (sku_cfg.sales_growth_trend * (day_idx / days))

            # Expected demand for this day
            expected_demand = base_mean * seasonality * trend_factor

            # Stochastic sales realization (Poisson for count data + overdispersion)
            units_sold = int(rng.poisson(lam=max(expected_demand, 0.1)))

            # Unit price may experience micro-promotions (-5% to +2%)
            price_variance = 1.0 + rng.uniform(-0.03, 0.01)
            daily_price = round(sku_cfg.unit_price * price_variance, 2)

            # Daily customer returns (binomial based on return_rate_pct)
            return_prob = sku_cfg.return_rate_pct / 100.0
            daily_returns = int(rng.binomial(n=max(units_sold, 1), p=min(return_prob, 0.99)))

            # Customer sentiment score with small Gaussian noise, bounded [1.0, 5.0]
            sentiment_noise = rng.normal(loc=0.0, scale=0.15)
            daily_sentiment = round(
                float(np.clip(sku_cfg.customer_sentiment_score + sentiment_noise, 1.0, 5.0)),
                2
            )

            # Stockout occurrence flag in history (simulated historical stockout event)
            # Highly trending items occasionally experienced 0 sales days in history due to past supply gaps
            is_historical_stockout = False
            if sku_cfg.archetype_label == "HIGH_STOCKOUT_RISK" and day_idx in (15, 38, 62):
                if rng.random() < 0.4:
                    units_sold = 0
                    is_historical_stockout = True

            history_records.append({
                "date": current_date,
                "sku_id": sku_cfg.sku_id,
                "product_name": sku_cfg.product_name,
                "category": sku_cfg.category,
                "units_sold": units_sold,
                "unit_price": daily_price,
                "unit_cost": sku_cfg.unit_cost,
                "daily_returns": daily_returns,
                "customer_sentiment_score": daily_sentiment,
                "out_of_stock": is_historical_stockout,
            })

    df_historical = pd.DataFrame.from_records(history_records)
    df_historical["date"] = pd.to_datetime(df_historical["date"])
    df_historical = df_historical.sort_values(by=["sku_id", "date"]).reset_index(drop=True)

    # -------------------------------------------------------------
    # Build Snapshot DataFrame
    # -------------------------------------------------------------
    snapshot_records: List[Dict[str, Any]] = []

    for sku_cfg in selected_configs:
        sku_hist = df_historical[df_historical["sku_id"] == sku_cfg.sku_id].sort_values("date")

        # Extract last 30 days sales sequence
        last_30d_series = sku_hist.tail(30)["units_sold"].tolist()

        # Compute empirical daily sales mean over full history
        empirical_mean = round(float(sku_hist["units_sold"].mean()), 2)

        # Empirical average sentiment and return rate for snapshot consistency
        empirical_sentiment = round(float(sku_hist["customer_sentiment_score"].mean()), 2)
        total_sold = sku_hist["units_sold"].sum()
        total_returned = sku_hist["daily_returns"].sum()
        empirical_return_rate = round(float((total_returned / max(total_sold, 1)) * 100), 2)

        # Allow snapshot to preserve the configured target profile
        snapshot_records.append({
            "sku_id": sku_cfg.sku_id,
            "product_name": sku_cfg.product_name,
            "category": sku_cfg.category,
            "current_stock": int(sku_cfg.current_stock),
            "daily_sales_mean": empirical_mean,
            "lead_time_days": int(sku_cfg.lead_time_days),
            "unit_cost": float(sku_cfg.unit_cost),
            "unit_price": float(sku_cfg.unit_price),
            "return_rate_pct": float(sku_cfg.return_rate_pct),
            "customer_sentiment_score": float(sku_cfg.customer_sentiment_score),
            "last_30d_sales_trend": last_30d_series,
            "archetype_label": sku_cfg.archetype_label,
        })

    df_snapshot = pd.DataFrame.from_records(snapshot_records)

    logger.info(
        f"Dataset generation complete: {len(df_historical)} historical rows, "
        f"{len(df_snapshot)} SKU inventory snapshots."
    )

    return df_historical, df_snapshot


if __name__ == "__main__":
    # Test generation and display diagnostic summary
    df_history, df_snap = generate_retail_dataset(num_skus=25, days=90)
    print("--- Snapshot Sample (First 5 SKUs) ---")
    print(df_snap[["sku_id", "product_name", "category", "current_stock", "daily_sales_mean", "return_rate_pct", "customer_sentiment_score"]].head())
    print("\n--- Archetype Distribution ---")
    print(df_snap["archetype_label"].value_counts())
    print("\n--- Trend Array Sample (SKU-1001) ---")
    print("Length of last_30d_sales_trend:", len(df_snap.iloc[0]["last_30d_sales_trend"]))
    print("Trend values:", df_snap.iloc[0]["last_30d_sales_trend"][:10], "...")
