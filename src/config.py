from __future__ import annotations

PREDICTION_HORIZON_CYCLES = 30
ROLLING_WINDOWS_CYCLES = (5, 15, 30)
COLD_START_MAX_CYCLES = 5
MAX_ALERT_RATE = 0.05
TARGET_EVENT_RECALL = 0.80
TARGET_POSITIVE_RATE_RANGE = (0.018, 0.022)
DEFAULT_CALIBRATION_METHOD = "sigmoid"
SCORING_BATCH_RULE = "UTC calendar date of snapshot_time"
RANDOM_SEED = 20261112

SNAPSHOT_KEY = "snapshot_id"
COMPONENT_KEY = "component_id"
AIRCRAFT_KEY = "aircraft_id"
INSTALLATION_KEY = "installation_id"
TIME_KEY = "flight_date"
CYCLE_KEY = "component_total_cycles"
BATCH_KEY = "scoring_batch_id"
TARGET_KEY = "unscheduled_removal_within_30_cycles"

SENSOR_COLUMNS = [
    "vibration_rms",
    "egt_margin",
    "sensor_temperature",
    "sensor_pressure",
    "sensor_current",
]

CATEGORICAL_COLUMNS = [
    "aircraft_model",
    "component_family",
    "part_number",
    "manufacture_batch",
    "operating_environment",
]

OUTPUT_COLUMNS = [
    "snapshot_id",
    "installation_id",
    "aircraft_id",
    "component_id",
    "scoring_batch_id",
    "raw_score",
    "raw_score_type",
    "risk_score",
    "alert_before_suppression",
    "alert_after_suppression",
    "final_alert",
    "capacity_deferred",
    "data_quality_status",
    "history_coverage_status",
    "scoring_method",
    "reason_codes",
    "model_version",
    "feature_version",
    "calibrator_version",
    "policy_version",
    "data_version",
]
