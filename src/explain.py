from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models import transform_lightgbm_features

FEATURE_PRESENTATION = {
    "vibration_rms": ("振动有效值", "检查振动、轴承及转动部件，并复核传感器通道。", "VIBRATION"),
    "egt_margin": ("排气温度裕度", "检查发动机热端、EGT裕度趋势及相关传感器。", "EGT_MARGIN"),
    "sensor_temperature": ("部件温度", "检查冷却、热负荷和温度传感器读数。", "TEMPERATURE"),
    "sensor_pressure": ("部件压力", "检查压力异常、泄漏和压力传感器。", "PRESSURE"),
    "sensor_current": ("部件电流", "检查电气负载、供电和电流传感器漂移。", "CURRENT"),
    "fault_code_count": ("近期故障代码频次", "核对重复故障代码、排故记录和故障确认结果。", "RECENT_FAULTS"),
    "maintenance_count": ("近期维修频次", "复核近期维修、重复故障和维修后状态。", "RECENT_MAINTENANCE"),
    "cycles_since_install": ("本次装机循环", "结合装机寿命和厂家限制安排检查。", "CYCLES_SINCE_INSTALL"),
    "component_total_cycles": ("部件全寿命循环", "结合全寿命使用记录和寿命限制安排检查。", "LIFETIME_CYCLES"),
    "cso": ("距大修循环", "结合大修间隔和使用循环安排检查。", "CSO"),
    "csn": ("部件累计循环", "结合累计使用记录安排检查。", "CSN"),
    "component_flight_hours": ("部件累计飞行小时", "结合飞行小时和循环共同评估使用负荷。", "FLIGHT_HOURS"),
    "operating_environment": ("运营环境", "复核高温、高湿或沙尘运行暴露及维护记录。", "HARSH_ENVIRONMENT"),
    "part_number": ("部件型号", "核对同型号可靠性和适用维修方案。", "PART_NUMBER"),
    "component_family": ("部件类别", "按对应部件族维修手册安排检查。", "COMPONENT_FAMILY"),
    "sensor_quality": ("传感器质量状态", "先确认数据质量，再执行相应部件检查。", "SENSOR_QUALITY"),
}


def _presentation(feature: str) -> tuple[str, str, str]:
    for prefix, (label, action, code) in FEATURE_PRESENTATION.items():
        if feature == prefix or feature.startswith(prefix + "_"):
            suffix = ""
            if "_mean_last_" in feature:
                suffix = "，窗口均值"
            elif "_std_last_" in feature:
                suffix = "，窗口波动"
            elif "_slope_per_cycle" in feature:
                suffix = "，窗口趋势"
            elif "_is_missing" in feature:
                suffix = "，缺失标志"
            return label + suffix, action, code
    return feature, "检查与该因素相关的部件和记录，由维修工程人员确认。", feature.upper()


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def lightgbm_raw_explanations(
    model: Any,
    background: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict[str, Any]:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    values = explainer.shap_values(samples)
    if isinstance(values, list):
        values = values[-1]
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = np.asarray(expected).reshape(-1)[-1]
    raw_prediction = model.predict(samples, raw_score=True)
    reconstructed = float(expected) + values.sum(axis=1)
    if not np.allclose(reconstructed, raw_prediction, atol=1e-5):
        raise ValueError("SHAP raw-score additivity check failed")
    return {
        "explainer": explainer,
        "values": values,
        "expected_raw_score": float(expected),
        "raw_score_type": "lightgbm_log_odds",
        "background_mode": "tree_path_dependent_node_cover",
        "supplied_background_rows": len(background),
        "additivity_max_error": float(np.max(np.abs(reconstructed - raw_prediction))),
    }


def save_global_shap_summary(values: np.ndarray, samples: pd.DataFrame, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, samples, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def local_reason_card(
    sample: pd.Series,
    shap_values: np.ndarray,
    feature_names: list[str],
    calibrated_risk: float,
    threshold: float,
    component_id: str,
    top_n: int = 3,
    exclude_features: set[str] | None = None,
) -> dict[str, Any]:
    excluded = exclude_features or {"snapshot_cycle", "aircraft_total_cycles", "reading_cycle"}
    contributions = pd.Series(shap_values, index=feature_names)
    contributions = contributions.loc[~contributions.index.isin(excluded)]
    if "component_family" in sample.index and str(sample["component_family"]) != "ENGINE":
        contributions = contributions.loc[~contributions.index.str.startswith("egt_margin")]
    records = []
    for feature, contribution in contributions.items():
        label, action, code = _presentation(feature)
        records.append(
            {
                "feature": feature,
                "feature_label": label,
                "suggested_action": action,
                "group_code": code,
                "contribution": float(contribution),
                "abs_contribution": abs(float(contribution)),
            }
        )
    table = pd.DataFrame(records)
    group_contributions = table.groupby("group_code")["contribution"].sum()
    ordered_groups = group_contributions.reindex(
        group_contributions.abs().sort_values(ascending=False).index
    ).head(top_n)
    reasons = []
    for code, contribution in ordered_groups.items():
        members = table.loc[table["group_code"].eq(code)]
        representative = members.loc[members["abs_contribution"].idxmax()]
        feature = representative["feature"]
        reasons.append(
            {
                "feature": feature,
                "feature_label": representative["feature_label"].split("，")[0] + "（相关特征合并）",
                "value": _json_value(sample[feature]),
                "direction": "INCREASES_RISK" if contribution > 0 else "DECREASES_RISK",
                "raw_score_contribution": float(contribution),
                "reason_code": f"{'HIGH' if contribution > 0 else 'LOW'}_{code}",
                "suggested_action": representative["suggested_action"],
                "member_features": members["feature"].tolist(),
            }
        )
    return {
        "component_id": component_id,
        "risk_score": float(calibrated_risk),
        "threshold": float(threshold),
        "raw_score_semantics": "Contributions apply to model log-odds, not calibrated probability",
        "selection_scope": "Top physical/operational features after excluding pure clock columns",
        "top_reasons": reasons,
        "suggested_mro_action": "安排针对性检查并核实传感器、故障和维修证据；不得仅凭评分直接拆换部件。",
    }


def attach_alert_reason_codes(
    decisions: pd.DataFrame,
    bundle: dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    result = decisions.copy()
    result["reason_codes"] = ""
    alerts = result.loc[result["final_alert"].astype(bool)]
    if alerts.empty:
        return result
    if bundle["model_family"] == "logistic":
        pipeline = bundle["model"]
        inputs = alerts[bundle["numeric_columns"] + bundle["categorical_columns"]]
        transformed = pipeline["preprocess"].transform(inputs)
        contributions = np.asarray(transformed * pipeline["model"].coef_[0]) if isinstance(
            transformed, np.ndarray
        ) else transformed.multiply(pipeline["model"].coef_[0]).toarray()
        names = pipeline["preprocess"].get_feature_names_out()
        raw = pipeline.decision_function(inputs)
        if not np.allclose(contributions.sum(axis=1) + pipeline["model"].intercept_[0], raw):
            raise ValueError("Logistic explanation additivity check failed")
        for position, index in enumerate(alerts.index):
            order = np.argsort(np.abs(contributions[position]))[::-1][:3]
            result.loc[index, "reason_codes"] = ";".join(
                f"{'HIGH' if contributions[position, j] > 0 else 'LOW'}_LINEAR_{names[j]}"
                for j in order
            )
        return result
    if bundle["model_family"] != "lightgbm":
        raise ValueError(f"Unsupported model family: {bundle['model_family']}")
    model_input = transform_lightgbm_features(
        alerts,
        bundle["numeric_columns"],
        bundle["categorical_columns"],
        bundle["category_maps"],
    )
    explanation = lightgbm_raw_explanations(bundle["model"], model_input, model_input)
    for position, (index, row) in enumerate(alerts.iterrows()):
        card = local_reason_card(
            model_input.iloc[position],
            explanation["values"][position],
            model_input.columns.tolist(),
            row["risk_score"],
            threshold,
            row["component_id"],
        )
        result.loc[index, "reason_codes"] = ";".join(
            reason["reason_code"] for reason in card["top_reasons"]
        )
    return result
