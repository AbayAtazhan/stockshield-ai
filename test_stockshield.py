"""
StockShield AI - Automated Test Suite
AMD Developer Hackathon MVP

Validates data generation, scikit-learn forecasting, IsolationForest anomaly detection,
and deterministic supply chain mathematical calculations.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from data_generator import generate_retail_dataset, CURATED_SKU_CATALOG
from ml_engine import StockShieldMLEngine, SKUIntelligenceRecord, analyze_retail_inventory


def test_data_generator_dimensions_and_schema():
    """Verify historical dataset and snapshot DataFrame dimensions and required columns."""
    days = 90
    num_skus = 25
    df_history, df_snapshot = generate_retail_dataset(num_skus=num_skus, days=days, random_seed=42)

    assert len(df_snapshot) == num_skus, f"Expected {num_skus} snapshot rows, got {len(df_snapshot)}"
    assert len(df_history) == num_skus * days, f"Expected {num_skus * days} history rows, got {len(df_history)}"

    required_snapshot_cols = [
        "sku_id",
        "product_name",
        "category",
        "current_stock",
        "daily_sales_mean",
        "lead_time_days",
        "unit_cost",
        "unit_price",
        "return_rate_pct",
        "customer_sentiment_score",
        "last_30d_sales_trend",
    ]

    for col in required_snapshot_cols:
        assert col in df_snapshot.columns, f"Missing required snapshot column: {col}"

    # Verify last_30d_sales_trend length
    for _, row in df_snapshot.iterrows():
        trend = row["last_30d_sales_trend"]
        assert isinstance(trend, list), f"Expected list for last_30d_sales_trend in {row['sku_id']}"
        assert len(trend) == 30, f"Expected 30 items in trend array, got {len(trend)}"

    # Verify archetype distribution
    archetypes = df_snapshot["archetype_label"].value_counts().to_dict()
    assert archetypes.get("HIGH_STOCKOUT_RISK", 0) >= 4, "Expected at least 4 High Stockout Risk items"
    assert archetypes.get("OVERSTOCK_DEAD_CAPITAL", 0) >= 3, "Expected at least 3 Overstock items"
    assert archetypes.get("HIGH_RETURN_ANOMALY", 0) >= 2, "Expected at least 2 High Return Anomaly items"


def test_ml_forecaster_training_and_predictions():
    """Verify HistGradientBoostingRegressor training, high R^2, and positive forecast outputs."""
    df_history, df_snapshot = generate_retail_dataset(num_skus=25, days=90, random_seed=42)
    engine = StockShieldMLEngine(random_state=42)

    forecaster = engine.train_demand_forecaster(df_history)
    assert forecaster is not None
    assert engine.evaluation_report is not None
    assert engine.evaluation_report.r2_score > 0.90, f"R^2 too low: {engine.evaluation_report.r2_score}"

    forecasts = engine.predict_14d_demand(df_snapshot, df_history)
    assert len(forecasts) == len(df_snapshot)
    assert np.all(forecasts >= 0.0), "Forecasts must be non-negative"


def test_isolation_forest_anomaly_detection():
    """Verify IsolationForest correctly identifies the high return anomaly SKUs."""
    df_history, df_snapshot = generate_retail_dataset(num_skus=25, days=90, random_seed=42)
    engine = StockShieldMLEngine(random_state=42)

    engine.train_anomaly_detector(df_snapshot)
    is_anomaly, anomaly_scores = engine.detect_anomalies(df_snapshot)

    assert len(is_anomaly) == len(df_snapshot)
    assert len(anomaly_scores) == len(df_snapshot)
    assert np.all((anomaly_scores >= 0.0) & (anomaly_scores <= 1.0)), "Anomaly scores must be in [0, 1]"

    # Anomalies should flag the designated high-return SKUs
    anomaly_skus = df_snapshot.loc[is_anomaly, "sku_id"].tolist()
    assert "SKU-3001" in anomaly_skus, "SKU-3001 (SSD thermal throttle) must be flagged as anomaly"
    assert "SKU-3002" in anomaly_skus, "SKU-3002 (Headset firmware bug) must be flagged as anomaly"


def test_deterministic_business_metrics_math():
    """Verify exact deterministic math formulas for days_until_stockout, risk levels, and revenue at risk."""
    df_history, df_snapshot = generate_retail_dataset(num_skus=25, days=90, random_seed=42)
    engine = StockShieldMLEngine(random_state=42)

    df_results = engine.run_full_pipeline(df_history, df_snapshot)

    for _, row in df_results.iterrows():
        # 1. days_until_stockout = current_stock / max(predicted_daily_demand, 0.1)
        expected_burn = max(row["predicted_daily_demand"], 0.1)
        expected_days_until_stockout = round(row["current_stock"] / expected_burn, 1)
        assert math.isclose(row["days_until_stockout"], expected_days_until_stockout, abs_tol=0.2), \
            f"days_until_stockout mismatch for {row['sku_id']}"

        # 2. Risk level assertions
        lead_time = row["lead_time_days"]
        days_stockout = row["days_until_stockout"]
        if days_stockout < lead_time:
            assert row["stockout_risk_level"] == "CRITICAL", f"Expected CRITICAL for {row['sku_id']}"
        elif days_stockout >= 60.0:
            assert row["stockout_risk_level"] == "OVERSTOCK", f"Expected OVERSTOCK for {row['sku_id']}"

        # 3. revenue_at_risk formula
        if days_stockout < lead_time:
            stockout_window = round(lead_time - days_stockout, 1)
            expected_rev_at_risk = round(stockout_window * row["predicted_daily_demand"] * row["unit_price"], 2)
            assert math.isclose(row["revenue_at_risk"], expected_rev_at_risk, abs_tol=1.0)
        else:
            assert row["revenue_at_risk"] == 0.0, f"Expected 0 revenue at risk for {row['sku_id']}"

        # 4. recommended_reorder_qty formula: cover lead time + 14-day safety buffer
        target_cycle = lead_time + 14.0
        needed_units = target_cycle * row["predicted_daily_demand"]
        expected_reorder = max(0, math.ceil(needed_units - row["current_stock"]))
        assert row["recommended_reorder_qty"] == expected_reorder, f"Reorder mismatch for {row['sku_id']}"


def test_pydantic_schema_integrity():
    """Verify all enriched output rows satisfy strict Pydantic validation."""
    df_history, df_snapshot = generate_retail_dataset(num_skus=25, days=90, random_seed=42)
    df_results = analyze_retail_inventory(df_history, df_snapshot, random_state=42)

    for record in df_results.to_dict(orient="records"):
        # Strip non-schema metadata if any
        model_obj = SKUIntelligenceRecord(**record)
        assert model_obj.sku_id.startswith("SKU-")
        assert model_obj.stockout_risk_level in {"CRITICAL", "WARNING", "HEALTHY", "OVERSTOCK"}
        assert model_obj.revenue_at_risk >= 0.0
        assert model_obj.recommended_reorder_qty >= 0


def test_ai_agent_executive_brief():
    """Verify executive brief generation and dynamic financial figures."""
    from ai_agent import StockShieldAIAgent
    df_history, df_snapshot = generate_retail_dataset(num_skus=25, days=90, random_seed=42)
    df_results = analyze_retail_inventory(df_history, df_snapshot, random_state=42)
    critical_df = df_results[df_results["stockout_risk_level"] == "CRITICAL"]

    agent = StockShieldAIAgent()
    brief = agent.generate_executive_brief(critical_df)

    assert brief.critical_sku_count == len(critical_df)
    assert brief.total_revenue_at_risk_usd > 100000.0
    assert "Executive Operations Action Brief" in brief.summary_memo or "CRISIS SUMMARY" in brief.summary_memo
    assert brief.telemetry.action_type == "executive_brief"
    assert brief.telemetry.latency_ms > 0.0
    assert brief.telemetry.inference_cost_usd > 0.0
    assert brief.telemetry.business_roi_multiplier > 1000.0


def test_ai_agent_po_email():
    """Verify supplier PO email generation with mathematical specifications."""
    from ai_agent import StockShieldAIAgent
    df_history, df_snapshot = generate_retail_dataset(num_skus=25, days=90, random_seed=42)
    df_results = analyze_retail_inventory(df_history, df_snapshot, random_state=42)
    top_critical = df_results[df_results["stockout_risk_level"] == "CRITICAL"].iloc[0]

    agent = StockShieldAIAgent()
    po = agent.generate_supplier_po_email(top_critical)

    assert po.sku_id == top_critical["sku_id"]
    assert po.recommended_reorder_qty == top_critical["recommended_reorder_qty"]
    assert po.total_po_value_usd > 0.0
    assert "Purchase Order" in po.email_body or "PO" in po.email_body
    assert po.telemetry.action_type == "po_email"


def test_ai_agent_vibe_marketing_and_quality_gate():
    """Verify overstock vibe marketing copy and automated Quality Gate grading."""
    from ai_agent import StockShieldAIAgent
    df_history, df_snapshot = generate_retail_dataset(num_skus=25, days=90, random_seed=42)
    df_results = analyze_retail_inventory(df_history, df_snapshot, random_state=42)
    top_overstock = df_results[df_results["stockout_risk_level"] == "OVERSTOCK"].iloc[0]

    agent = StockShieldAIAgent()
    campaign = agent.generate_vibe_marketing_campaign(top_overstock, discount_pct=25.0)

    assert campaign.sku_id == top_overstock["sku_id"]
    assert campaign.discount_pct == 25.0
    assert campaign.discounted_price < campaign.original_price
    # Quality gate checks
    assert campaign.quality_gate.deterministic_margin_safe is True
    assert campaign.quality_gate.projected_margin_usd > 0.0
    assert campaign.quality_gate.total_quality_score >= 75
    assert campaign.quality_gate.passed_quality_gate is True
    assert len(campaign.campaign_headline) > 5
    assert len(campaign.promotional_copy) > 20


def test_telemetry_tracker_unit_economics():
    """Verify TelemetryTracker recording, costs, and summary aggregation."""
    from ai_agent import TelemetryTracker
    tracker = TelemetryTracker()

    tracker.record_call(
        action_type="test_action",
        provider="test-provider",
        model="test-model",
        endpoint="http://localhost:8000/v1",
        latency_ms=120.0,
        prompt_tokens=400,
        completion_tokens=600,
        revenue_at_risk_saved=50000.0,
        is_simulated=True
    )

    summary = tracker.get_summary()
    assert summary["total_calls"] == 1
    assert summary["total_tokens"] == 1000
    # 1000 tokens @ $0.0008 / 1K = $0.0008
    assert math.isclose(summary["total_cost_usd"], 0.0008, abs_tol=1e-5)
    assert summary["total_revenue_saved_usd"] == 50000.0
    # ROI = 50,000 / 0.0008 = 62,500,000x
    assert math.isclose(summary["overall_roi_multiplier"], 62500000.0, abs_tol=100.0)


if __name__ == "__main__":
    test_data_generator_dimensions_and_schema()
    test_ml_forecaster_training_and_predictions()
    test_isolation_forest_anomaly_detection()
    test_deterministic_business_metrics_math()
    test_pydantic_schema_integrity()
    test_ai_agent_executive_brief()
    test_ai_agent_po_email()
    test_ai_agent_vibe_marketing_and_quality_gate()
    test_telemetry_tracker_unit_economics()
    print("ALL TESTS PASSED SUCCESSFULLY!")
