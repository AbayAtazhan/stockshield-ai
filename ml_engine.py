"""
StockShield AI - Enterprise Retail & Customer Intelligence ML Engine
AMD Developer Hackathon MVP

Mathematical ML & Inventory Risk Pipeline:
 1. Demand Forecasting: HistGradientBoostingRegressor trained on historical lag features
    (7-day moving average, day-of-week, price, sentiment, return rate) to predict 14-day forward demand.
 2. Anomaly Detection: IsolationForest detecting discrepancies between sales velocity and return rates.
 3. Deterministic Business Metrics:
    - days_until_stockout = current_stock / max(predicted_daily_demand, 0.1)
    - stockout_risk_level: CRITICAL, WARNING, HEALTHY, or OVERSTOCK
    - revenue_at_risk: exact dollar amount lost during stockout window if not reordered today
    - recommended_reorder_qty: exact units needed to cover lead time + 14-day safety buffer
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StockShield.MLEngine")


class SKUIntelligenceRecord(BaseModel):
    """Structured Pydantic validation schema for single-SKU analysis results."""
    sku_id: str
    product_name: str
    category: str
    current_stock: int
    lead_time_days: int
    unit_cost: float
    unit_price: float
    return_rate_pct: float
    customer_sentiment_score: float
    predicted_daily_demand: float
    forecasted_14d_demand: float
    days_until_stockout: float
    stockout_risk_level: str
    revenue_at_risk: float
    recommended_reorder_qty: int
    is_anomaly: bool
    anomaly_score: float
    stockout_window_days: float
    tied_up_capital: float
    gross_margin_pct: float


@dataclass
class ModelEvaluationReport:
    """Diagnostic metrics for ML model evaluation."""
    r2_score: float
    mae: float
    training_samples: int
    features_used: List[str]
    num_anomalies_detected: int


class StockShieldMLEngine:
    """
    Enterprise Machine Learning and Inventory Risk Engine.
    Employs scikit-learn regressors and anomaly detectors alongside deterministic
    supply chain replenishment algorithms.
    """

    FEATURE_COLUMNS = [
        "sales_7d_ma",
        "day_of_week",
        "unit_price",
        "customer_sentiment_score",
        "return_rate_pct",
    ]

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.forecaster: Optional[HistGradientBoostingRegressor] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.evaluation_report: Optional[ModelEvaluationReport] = None

    def prepare_training_data(
        self,
        df_historical: pd.DataFrame,
        horizon_days: int = 14
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts temporal lag features and computes the forward 14-day cumulative demand
        target without future lookahead leakage.

        Parameters:
            df_historical (pd.DataFrame): Daily transaction records sorted by SKU and date.
            horizon_days (int): Forecast horizon in days (default 14).

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y.
        """
        df = df_historical.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["sku_id", "date"]).reset_index(drop=True)

        # 1. Day of week feature (0 = Mon, 6 = Sun)
        df["day_of_week"] = df["date"].dt.dayofweek

        # 2. 7-day rolling moving average of sales (shifted by 1 day to strictly avoid leakage)
        df["sales_7d_ma"] = (
            df.groupby("sku_id")["units_sold"]
            .transform(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean().bfill())
        )

        # 3. Daily return rate percentage
        df["return_rate_pct"] = (
            (df["daily_returns"] / np.maximum(df["units_sold"], 1.0)) * 100.0
        ).clip(0.0, 100.0)

        # 4. Target computation: Forward 14-day cumulative sales
        # Sum of units_sold for t+1 through t+horizon_days
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_days)
        df["target_14d_demand"] = (
            df.groupby("sku_id")["units_sold"]
            .transform(lambda s: s.shift(-1).rolling(window=indexer, min_periods=horizon_days).sum())
        )

        # Drop training records without full forward 14-day history
        valid_mask = df["target_14d_demand"].notna()
        df_train = df[valid_mask].copy()

        X = df_train[self.FEATURE_COLUMNS]
        y = df_train["target_14d_demand"]

        logger.info(
            f"Prepared ML training dataset: {len(X)} samples across {len(self.FEATURE_COLUMNS)} features."
        )
        return X, y

    def train_demand_forecaster(
        self,
        df_historical: pd.DataFrame
    ) -> HistGradientBoostingRegressor:
        """
        Trains HistGradientBoostingRegressor on historical retail lag features.

        Parameters:
            df_historical (pd.DataFrame): Historical daily sales dataset.

        Returns:
            HistGradientBoostingRegressor: The fitted regression model.
        """
        X, y = self.prepare_training_data(df_historical, horizon_days=14)

        # Initialize scikit-learn HistGradientBoostingRegressor with tuned parameters
        self.forecaster = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=0.1,
            random_state=self.random_state,
        )

        self.forecaster.fit(X, y)

        # Assess in-sample training quality
        y_pred = self.forecaster.predict(X)
        r2 = float(r2_score(y, y_pred))
        mae = float(mean_absolute_error(y, y_pred))

        logger.info(
            f"HistGradientBoostingRegressor training completed | R^2 Score: {r2:.4f} | MAE: {mae:.2f} units."
        )

        self.evaluation_report = ModelEvaluationReport(
            r2_score=r2,
            mae=mae,
            training_samples=len(X),
            features_used=self.FEATURE_COLUMNS,
            num_anomalies_detected=0,
        )

        return self.forecaster

    def predict_14d_demand(
        self,
        df_snapshot: pd.DataFrame,
        df_historical: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Predicts the 14-day forward demand for each SKU in the inventory snapshot.

        Parameters:
            df_snapshot (pd.DataFrame): Current inventory snapshot DataFrame.
            df_historical (pd.DataFrame, optional): Historical data to compute recent lag if needed.

        Returns:
            np.ndarray: Forecasted 14-day cumulative demand array.
        """
        if self.forecaster is None:
            raise RuntimeError("Forecaster has not been fitted. Call train_demand_forecaster first.")

        inference_rows: List[Dict[str, float]] = []

        for _, row in df_snapshot.iterrows():
            sku_id = row["sku_id"]

            # Compute recent 7-day moving average
            if "last_30d_sales_trend" in row and isinstance(row["last_30d_sales_trend"], list):
                trend_array = row["last_30d_sales_trend"]
                sales_7d_ma = float(np.mean(trend_array[-7:])) if len(trend_array) >= 7 else float(np.mean(trend_array))
            elif df_historical is not None:
                sku_hist = df_historical[df_historical["sku_id"] == sku_id].sort_values("date")
                sales_7d_ma = float(sku_hist.tail(7)["units_sold"].mean())
            else:
                sales_7d_ma = float(row.get("daily_sales_mean", 10.0))

            unit_price = float(row["unit_price"])
            sentiment = float(row["customer_sentiment_score"])
            return_rate = float(row["return_rate_pct"])

            # To avoid day-of-week bias across a 14-day horizon, evaluate expected prediction
            # across the 7-day weekly cycle and average
            dow_predictions: List[float] = []
            for dow in range(7):
                feat_dict = {
                    "sales_7d_ma": sales_7d_ma,
                    "day_of_week": dow,
                    "unit_price": unit_price,
                    "customer_sentiment_score": sentiment,
                    "return_rate_pct": return_rate,
                }
                dow_predictions.append(feat_dict)

            df_dow = pd.DataFrame(dow_predictions)[self.FEATURE_COLUMNS]
            dow_forecasts = self.forecaster.predict(df_dow)
            expected_14d_forecast = float(np.mean(dow_forecasts))

            # Demand cannot be negative
            clamped_forecast = max(0.0, round(expected_14d_forecast, 1))
            inference_rows.append({"forecasted_14d_demand": clamped_forecast})

        forecasts = np.array([r["forecasted_14d_demand"] for r in inference_rows], dtype=float)
        return forecasts

    def train_anomaly_detector(
        self,
        df_snapshot: pd.DataFrame
    ) -> IsolationForest:
        """
        Trains IsolationForest on sales velocity vs. return rates to flag anomalous SKUs.

        Parameters:
            df_snapshot (pd.DataFrame): Inventory snapshot containing daily_sales_mean and return_rate_pct.

        Returns:
            IsolationForest: Fitted anomaly detection model.
        """
        feature_matrix = df_snapshot[["daily_sales_mean", "return_rate_pct"]].to_numpy()

        # IsolationForest with contamination tuned to isolate severe return/defect anomalies
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.12,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.anomaly_detector.fit(feature_matrix)
        logger.info("IsolationForest anomaly detector fitted on sales velocity vs return rates.")
        return self.anomaly_detector

    def detect_anomalies(
        self,
        df_snapshot: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies IsolationForest to detect retail anomalies.

        Parameters:
            df_snapshot (pd.DataFrame): Current snapshot DataFrame.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - is_anomaly (boolean array, True if anomaly)
                - anomaly_score (float array, normalized [0.0, 1.0] severity score)
        """
        if self.anomaly_detector is None:
            self.train_anomaly_detector(df_snapshot)

        feature_matrix = df_snapshot[["daily_sales_mean", "return_rate_pct"]].to_numpy()

        # -1 = anomaly, 1 = normal
        raw_preds = self.anomaly_detector.predict(feature_matrix)
        is_anomaly = (raw_preds == -1)

        # Scikit-learn score_samples: more negative = more anomalous
        scores = self.anomaly_detector.score_samples(feature_matrix)

        # Calibrate to intuitive 0.0 - 1.0 severity score (1.0 = highest anomaly severity)
        min_s = float(np.min(scores))
        max_s = float(np.max(scores))
        score_range = max_s - min_s if max_s > min_s else 1.0
        normalized_scores = np.round((max_s - scores) / score_range, 3)

        num_anomalies = int(np.sum(is_anomaly))
        if self.evaluation_report is not None:
            self.evaluation_report.num_anomalies_detected = num_anomalies

        logger.info(f"Anomaly detection complete: {num_anomalies} anomalies detected.")
        return is_anomaly, normalized_scores

    def calculate_deterministic_metrics(
        self,
        df_snapshot: pd.DataFrame,
        forecasted_14d_demand: np.ndarray,
        is_anomaly: np.ndarray,
        anomaly_scores: np.ndarray,
    ) -> pd.DataFrame:
        """
        Calculates supply chain risk and replenishment business metrics:
          - predicted_daily_demand = forecasted_14d_demand / 14.0
          - days_until_stockout = current_stock / max(predicted_daily_demand, 0.1)
          - stockout_risk_level: CRITICAL (< lead_time_days), WARNING, HEALTHY, or OVERSTOCK (>= 60 days)
          - revenue_at_risk: dollar loss during stockout window if not reordered today
          - recommended_reorder_qty: units needed to cover lead time + 14-day safety buffer

        Parameters:
            df_snapshot (pd.DataFrame): Inventory snapshot.
            forecasted_14d_demand (np.ndarray): Predicted 14-day demand.
            is_anomaly (np.ndarray): Boolean anomaly flags.
            anomaly_scores (np.ndarray): Anomaly severity scores.

        Returns:
            pd.DataFrame: Enriched DataFrame containing all ML forecasts and business metrics.
        """
        df_enriched = df_snapshot.copy()

        # 1. Forecasted metrics
        df_enriched["forecasted_14d_demand"] = np.round(forecasted_14d_demand, 1)
        raw_daily_demand = np.maximum(forecasted_14d_demand / 14.0, 0.0)
        df_enriched["predicted_daily_demand"] = np.round(raw_daily_demand, 2)

        # 2. Days until stockout = current_stock / max(predicted_daily_demand, 0.1)
        effective_burn_rate = np.maximum(df_enriched["predicted_daily_demand"].to_numpy(dtype=float), 0.1)
        df_enriched["days_until_stockout"] = np.round(
            df_enriched["current_stock"].to_numpy(dtype=float) / effective_burn_rate, 1
        )

        # 3. Stockout Risk Level Categorization
        # - CRITICAL: days_until_stockout < lead_time_days
        # - WARNING: days_until_stockout < (lead_time_days * 1.5)
        # - OVERSTOCK: days_until_stockout >= 60.0
        # - HEALTHY: otherwise
        risk_levels: List[str] = []
        for _, row in df_enriched.iterrows():
            d_stockout = row["days_until_stockout"]
            lead_time = row["lead_time_days"]

            if d_stockout < lead_time:
                risk_levels.append("CRITICAL")
            elif d_stockout < (lead_time * 1.5):
                risk_levels.append("WARNING")
            elif d_stockout >= 60.0:
                risk_levels.append("OVERSTOCK")
            else:
                risk_levels.append("HEALTHY")

        df_enriched["stockout_risk_level"] = risk_levels

        # 4. Revenue at Risk: exact dollar amount lost during the stockout window if not reordered today
        # Stockout window (days): max(0, lead_time_days - days_until_stockout)
        stockout_windows = np.maximum(
            0.0,
            df_enriched["lead_time_days"].to_numpy(dtype=float) - df_enriched["days_until_stockout"].to_numpy(dtype=float)
        )
        df_enriched["stockout_window_days"] = np.round(stockout_windows, 1)

        units_lost_in_window = df_enriched["stockout_window_days"] * df_enriched["predicted_daily_demand"]
        revenue_at_risk = units_lost_in_window * df_enriched["unit_price"]
        df_enriched["revenue_at_risk"] = np.round(revenue_at_risk, 2)

        # 5. Recommended Reorder Qty: units needed to cover lead time + 14-day safety buffer
        # target_cycle_days = lead_time_days + 14
        # total_required_units = target_cycle_days * predicted_daily_demand
        # recommended_reorder_qty = max(0, ceil(total_required_units - current_stock))
        target_cycle_days = df_enriched["lead_time_days"].to_numpy(dtype=float) + 14.0
        total_required_units = target_cycle_days * df_enriched["predicted_daily_demand"].to_numpy(dtype=float)
        net_reorder_needed = total_required_units - df_enriched["current_stock"].to_numpy(dtype=float)
        reorder_quantities = np.maximum(0, np.ceil(net_reorder_needed)).astype(int)
        df_enriched["recommended_reorder_qty"] = reorder_quantities

        # 6. Anomaly fields
        df_enriched["is_anomaly"] = is_anomaly.astype(bool)
        df_enriched["anomaly_score"] = anomaly_scores.astype(float)

        # 7. Additional Financial & Executive Metrics
        df_enriched["tied_up_capital"] = np.round(
            df_enriched["current_stock"] * df_enriched["unit_cost"], 2
        )
        df_enriched["gross_margin_pct"] = np.round(
            ((df_enriched["unit_price"] - df_enriched["unit_cost"]) / df_enriched["unit_price"]) * 100.0, 2
        )

        return df_enriched

    def run_full_pipeline(
        self,
        df_historical: pd.DataFrame,
        df_snapshot: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Executes the full StockShield ML intelligence pipeline.

        Parameters:
            df_historical (pd.DataFrame): Daily transaction history.
            df_snapshot (pd.DataFrame): Current inventory snapshot.

        Returns:
            pd.DataFrame: Enriched retail intelligence dataset with verified Pydantic schema compatibility.
        """
        logger.info("Initiating StockShield AI Machine Learning Pipeline...")

        # Step 1: Train Demand Forecaster
        self.train_demand_forecaster(df_historical)

        # Step 2: Forecast 14-day Demand
        forecasts = self.predict_14d_demand(df_snapshot, df_historical)

        # Step 3: Train and Detect Anomalies
        self.train_anomaly_detector(df_snapshot)
        is_anomaly, anomaly_scores = self.detect_anomalies(df_snapshot)

        # Step 4: Calculate Deterministic Supply Chain Business Metrics
        df_result = self.calculate_deterministic_metrics(
            df_snapshot=df_snapshot,
            forecasted_14d_demand=forecasts,
            is_anomaly=is_anomaly,
            anomaly_scores=anomaly_scores,
        )

        # Step 5: Validate records against Pydantic schema
        validated_records: List[Dict[str, Any]] = []
        for record_dict in df_result.to_dict(orient="records"):
            validated_obj = SKUIntelligenceRecord(**record_dict)
            validated_records.append(validated_obj.model_dump())

        df_validated = pd.DataFrame.from_records(validated_records)
        # Preserve original trend list if present
        if "last_30d_sales_trend" in df_snapshot.columns:
            df_validated["last_30d_sales_trend"] = df_snapshot["last_30d_sales_trend"].values

        logger.info(
            f"Pipeline executed successfully for {len(df_validated)} SKUs. "
            f"Critical risks: {(df_validated['stockout_risk_level'] == 'CRITICAL').sum()} | "
            f"Overstock items: {(df_validated['stockout_risk_level'] == 'OVERSTOCK').sum()} | "
            f"Anomalies: {df_validated['is_anomaly'].sum()}"
        )
        return df_validated

    def generate_14d_daily_projections(
        self,
        df_analysis: pd.DataFrame,
        start_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Dict[str, Union[List[str], List[float]]]]:
        """
        Generates realistic day-by-day projected units sold for the upcoming 14 days,
        preserving the total forecasted_14d_demand while honoring day-of-week retail seasonality.

        Returns:
            Dict mapping sku_id -> {'dates': List[str], 'projected_sales': List[float]}
        """
        if start_date is None:
            start_date = pd.Timestamp.now().normalize()

        future_dates = pd.date_range(start=start_date, periods=14, freq="D")
        dow_weights = np.array([
            0.95, 0.90, 0.96, 1.02, 1.15, 1.28, 1.22,
            0.95, 0.90, 0.96, 1.02, 1.15, 1.28, 1.22
        ])
        norm_weights = dow_weights / dow_weights.sum()

        projections: Dict[str, Dict[str, Union[List[str], List[float]]]] = {}
        for _, row in df_analysis.iterrows():
            total_14d = float(row["forecasted_14d_demand"])
            daily_vals = [round(float(v), 2) for v in (total_14d * norm_weights)]
            projections[row["sku_id"]] = {
                "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
                "projected_sales": daily_vals,
            }

        return projections


def analyze_retail_inventory(
    df_historical: pd.DataFrame,
    df_snapshot: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Convenience functional API to analyze retail inventory using the StockShield ML Engine.

    Parameters:
        df_historical (pd.DataFrame): Daily historical transactions.
        df_snapshot (pd.DataFrame): Current inventory snapshot.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Enriched analysis DataFrame.
    """
    engine = StockShieldMLEngine(random_state=random_state)
    return engine.run_full_pipeline(df_historical, df_snapshot)


if __name__ == "__main__":
    from data_generator import generate_retail_dataset

    logger.info("Running standalone ML Engine demonstration...")
    hist_df, snap_df = generate_retail_dataset(num_skus=25, days=90)

    engine = StockShieldMLEngine()
    result_df = engine.run_full_pipeline(hist_df, snap_df)

    print("\n" + "=" * 95)
    print("STOCKSHIELD AI - ENTERPRISE RETAIL & CUSTOMER INTELLIGENCE MVP SUMMARY")
    print("=" * 95)

    display_cols = [
        "sku_id",
        "product_name",
        "current_stock",
        "lead_time_days",
        "predicted_daily_demand",
        "forecasted_14d_demand",
        "days_until_stockout",
        "stockout_risk_level",
        "revenue_at_risk",
        "recommended_reorder_qty",
        "is_anomaly",
        "anomaly_score",
    ]

    print("\n--- CRITICAL STOCKOUT RISKS (Lead time > Days until stockout) ---")
    critical_items = result_df[result_df["stockout_risk_level"] == "CRITICAL"][display_cols]
    print(critical_items.to_string(index=False))

    print("\n--- OVERSTOCK / DEAD CAPITAL ITEMS ---")
    overstock_items = result_df[result_df["stockout_risk_level"] == "OVERSTOCK"][
        ["sku_id", "product_name", "current_stock", "days_until_stockout", "tied_up_capital", "recommended_reorder_qty"]
    ]
    print(overstock_items.to_string(index=False))

    print("\n--- HIGH RETURN ANOMALIES (Detected via IsolationForest) ---")
    anomaly_items = result_df[result_df["is_anomaly"]][
        ["sku_id", "product_name", "return_rate_pct", "customer_sentiment_score", "predicted_daily_demand", "anomaly_score"]
    ]
    print(anomaly_items.to_string(index=False))

    print("\n" + "=" * 95)
    total_rev_at_risk = result_df["revenue_at_risk"].sum()
    total_dead_capital = result_df[result_df["stockout_risk_level"] == "OVERSTOCK"]["tied_up_capital"].sum()
    print(f"Total Revenue At Risk (Immediate reorder required): ${total_rev_at_risk:,.2f}")
    print(f"Total Dead Working Capital (Overstock):               ${total_dead_capital:,.2f}")
    print("=" * 95)
