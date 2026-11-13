from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.explain import (
    attach_alert_reason_codes,
    lightgbm_raw_explanations,
    local_reason_card,
    save_global_shap_summary,
)
from src.models import transform_lightgbm_features
from src.training import TARGET, write_prediction_contract


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def _restore_decisions(
    features: pd.DataFrame,
    splits: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    test_features = features.merge(
        splits.loc[splits["final_use"].eq("test"), ["snapshot_id", "entity_group_id"]],
        on="snapshot_id",
        validate="one_to_one",
    )
    prediction_by_id = predictions.set_index("snapshot_id")
    for column in prediction_by_id.columns:
        test_features[column] = test_features["snapshot_id"].map(prediction_by_id[column])
    return test_features


def _model_input(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    if bundle["model_family"] == "logistic":
        original = frame[bundle["numeric_columns"] + bundle["categorical_columns"]]
        preprocessor = bundle["model"]["preprocess"]
        transformed = preprocessor.transform(original)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out(), index=frame.index)
    return transform_lightgbm_features(
        frame,
        bundle["numeric_columns"],
        bundle["categorical_columns"],
        bundle["category_maps"],
    )


def _explanation(model_input: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    if bundle["model_family"] == "lightgbm":
        return lightgbm_raw_explanations(bundle["model"], model_input, model_input)
    model = bundle["model"]["model"]
    contributions = model_input.to_numpy() * model.coef_[0]
    raw = model.decision_function(model_input.to_numpy())
    reconstructed = float(model.intercept_[0]) + contributions.sum(axis=1)
    error = float(np.max(np.abs(raw - reconstructed)))
    if error > 1e-5:
        raise ValueError("Logistic explanation additivity check failed")
    return {"values": contributions, "additivity_max_error": error}


def _card(
    row: pd.Series,
    model_input: pd.DataFrame,
    values: np.ndarray,
    position: int,
    threshold: float,
) -> dict[str, Any]:
    card = local_reason_card(
        model_input.iloc[position],
        values[position],
        model_input.columns.tolist(),
        row["risk_score"],
        threshold,
        row["component_id"],
    )
    card.update(
        {
            "snapshot_id": row["snapshot_id"],
            "aircraft_id": row["aircraft_id"],
            "component_family": row["component_family"],
            "part_number": row["part_number"],
            "scoring_batch_id": row["scoring_batch_id"],
            "removal_event_id": (
                str(row["removal_event_id"]) if pd.notna(row["removal_event_id"]) else None
            ),
            "target": int(row[TARGET]) if pd.notna(row[TARGET]) else None,
            "scoring_method": row["scoring_method"],
            "final_alert": bool(row["final_alert"]),
        }
    )
    return card


def _cards_markdown(cards: list[dict[str, Any]]) -> str:
    lines = [
        "# 高风险部件 SHAP 解释卡",
        "",
        "贡献对应模型原始 log-odds；风险分数为校准及冷启动混合后的概率。LightGBM 使用 SHAP；逻辑回归使用变换特征相对零参考值的线性贡献。",
        "",
    ]
    for index, card in enumerate(cards, start=1):
        lines.extend(
            [
                f"## 案例 {index}：{card['component_id']}",
                "",
                f"- 飞机：{card['aircraft_id']}；部件族：{card['component_family']}；型号：{card['part_number']}",
                f"- 快照：{card['snapshot_id']}；批次：{card['scoring_batch_id']}",
                f"- 风险分数：{card['risk_score']:.6f}；冻结阈值：{card['threshold']:.6f}",
                f"- 实际标签：{card['target']}；关联拆换事件：{card['removal_event_id']}",
                "",
                "| 排名 | 主要因素 | 当前值 | 方向 | 原始分数贡献 | 建议检查 |",
                "| ---: | --- | ---: | --- | ---: | --- |",
            ]
        )
        for rank, reason in enumerate(card["top_reasons"], start=1):
            direction = "推高风险" if reason["direction"] == "INCREASES_RISK" else "降低风险"
            value = reason["value"]
            display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(
                f"| {rank} | {reason['feature_label']} | {display_value} | {direction} | {reason['raw_score_contribution']:+.5f} | {reason['suggested_action']} |"
            )
        lines.extend(["", f"**MRO 建议：** {card['suggested_mro_action']}", ""])
    return "\n".join(lines)


def generate_explanations(
    experiment_dir: str | Path,
    features_path: str | Path,
    split_path: str | Path,
    test_dir: str | Path,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    experiment = Path(experiment_dir)
    test = Path(test_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    with (experiment / "freeze_manifest.json").open(encoding="utf-8") as handle:
        freeze = json.load(handle)
    bundle = joblib.load(experiment / "frozen_model.joblib")
    features = pd.read_parquet(features_path)
    splits = pd.read_parquet(split_path)
    predictions = pd.read_parquet(test / "test_predictions.parquet")
    decisions = _restore_decisions(features, splits, predictions)
    decisions = attach_alert_reason_codes(decisions, bundle, freeze["threshold"])
    write_prediction_contract(decisions, reports / "test_predictions_with_explanations.parquet")

    labeled = decisions.loc[decisions[TARGET].notna()]
    positive = labeled.loc[labeled[TARGET].eq(1)]
    negative = labeled.loc[labeled[TARGET].eq(0)]
    positive_sample = positive.sample(n=min(1000, len(positive)), random_state=20261112)
    negative_sample = negative.sample(
        n=min(2000 - len(positive_sample), len(negative)), random_state=20261112
    )
    summary_frame = pd.concat([positive_sample, negative_sample]).sample(
        frac=1, random_state=20261112
    )
    summary_input = _model_input(summary_frame, bundle)
    summary_explanation = _explanation(summary_input, bundle)
    save_global_shap_summary(
        summary_explanation["values"], summary_input, reports / "figures" / "shap_summary.png"
    )
    importance = pd.DataFrame(
        {
            "feature": summary_input.columns,
            "mean_abs_shap_log_odds": np.abs(summary_explanation["values"]).mean(axis=0),
        }
    ).sort_values("mean_abs_shap_log_odds", ascending=False)
    importance.to_csv(reports / "shap_feature_importance.csv", index=False)

    alerts = decisions.loc[decisions["final_alert"].astype(bool)].copy()
    true_alerts = alerts.loc[alerts[TARGET].eq(1)].sort_values("risk_score", ascending=False)
    selected = true_alerts.drop_duplicates("component_id").head(2)
    selected_input = _model_input(selected, bundle)
    selected_explanation = _explanation(selected_input, bundle)
    cards = [
        _card(row, selected_input, selected_explanation["values"], position, freeze["threshold"])
        for position, (_, row) in enumerate(selected.iterrows())
    ]
    with (reports / "high_risk_cards.json").open("w", encoding="utf-8") as handle:
        json.dump(cards, handle, indent=2, default=_json_default, allow_nan=False)
    (reports / "high_risk_cards.md").write_text(_cards_markdown(cards), encoding="utf-8")

    false_positive = alerts.loc[alerts[TARGET].eq(0)].sort_values("risk_score", ascending=False).head(1)
    false_positive_card = None
    if len(false_positive):
        false_input = _model_input(false_positive, bundle)
        false_explanation = _explanation(false_input, bundle)
        false_positive_card = _card(
            false_positive.iloc[0], false_input, false_explanation["values"], 0, freeze["threshold"]
        )
        with (reports / "false_positive_card.json").open("w", encoding="utf-8") as handle:
            json.dump(false_positive_card, handle, indent=2, default=_json_default, allow_nan=False)

    manifest = {
        "source_test_predictions_sha256": _sha256(test / "test_predictions.parquet"),
        "freeze_manifest_sha256": _sha256(experiment / "freeze_manifest.json"),
        "summary_rows": len(summary_input),
        "explanation_method": "tree_shap_raw" if bundle["model_family"] == "lightgbm" else "linear_transformed_zero_reference",
        "summary_additivity_max_error": summary_explanation["additivity_max_error"],
        "alert_additivity_max_error": selected_explanation["additivity_max_error"],
        "high_risk_card_count": len(cards),
        "false_positive_card": false_positive_card is not None,
        "false_negative_card": None,
        "false_negative_note": "Event misses are listed in the frozen acceptance report.",
        "reason_coded_final_alerts": int(decisions.loc[decisions["final_alert"], "reason_codes"].ne("").sum()),
        "final_alerts": int(decisions["final_alert"].sum()),
    }
    with (reports / "explanation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate frozen-model SHAP deliverables")
    parser.add_argument("--experiment", default="artifacts/experiments/validation_v1")
    parser.add_argument("--features", default="artifacts/features/mock_v0_2_0/features.parquet")
    parser.add_argument("--split", default="artifacts/splits/mock_v0_2_0/split_manifest.parquet")
    parser.add_argument("--test-dir", default="artifacts/frozen_test/validation_v1")
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    result = generate_explanations(
        args.experiment, args.features, args.split, args.test_dir, args.reports
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
