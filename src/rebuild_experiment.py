from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.provenance import sha256


def run_module(module: str, *arguments: str) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], check=True)


def rebuild(run_id: str, protocol_path: str | Path) -> Path:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id):
        raise ValueError("Run ID must contain letters, digits, underscores or dashes")
    root = Path("artifacts/runs") / run_id
    if root.exists():
        raise FileExistsError(f"Experiment already exists: {root}")
    root.mkdir(parents=True)
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config_path = Path("configs/mock_data_acceptance_2027.yaml")
    if sha256(config_path) != protocol["dataset_config_sha256"]:
        raise ValueError("Acceptance config differs from the registered protocol")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    (root / "test_results.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    tests.check_returncode()
    print(tests.stdout, flush=True)
    training_data = Path("data/generated/mock_v0_2_0")
    if not (training_data / "manifest.json").exists():
        subprocess.run([sys.executable, "data/generate_mock_data.py", "--output", str(training_data),
                        "--combined-output", str(training_data / "combined.csv")], check=True)
    run_module("src.validate_data", "--data-dir", str(training_data))
    features = root / "training_features/features.parquet"
    split = root / "training_split/split_manifest.parquet"
    run_module("src.build_features", "--data-dir", str(training_data), "--output", str(features))
    run_module("src.split_data", "--data-dir", str(training_data), "--output", str(split.parent))
    experiment = root / "selection"
    run_module("src.train_models", "--features", str(features), "--split", str(split),
               "--output", str(experiment), "--acceptance-protocol", str(protocol_path))
    # The independent dataset is first generated after selection has written its frozen manifest.
    acceptance_data = root / "acceptance_data"
    subprocess.run([sys.executable, "data/generate_mock_data.py", "--config", str(config_path),
                    "--output", str(acceptance_data), "--combined-output", str(acceptance_data / "combined.csv")], check=True)
    acceptance_features = root / "acceptance_features/features.parquet"
    acceptance_split = root / "acceptance_split/split_manifest.parquet"
    run_module("src.build_features", "--data-dir", str(acceptance_data), "--output", str(acceptance_features))
    run_module("src.acceptance", "split", "--features", str(acceptance_features), "--output", str(acceptance_split))
    acceptance = root / "acceptance"
    run_module("src.evaluate_frozen", "--experiment", str(experiment), "--features", str(acceptance_features),
               "--split", str(acceptance_split), "--data-dir", str(acceptance_data), "--output", str(acceptance),
               "--acceptance-protocol", str(protocol_path), "--bootstrap-iterations", str(protocol["bootstrap_iterations"]))
    run_module("src.generate_explanations", "--experiment", str(experiment), "--features", str(acceptance_features),
               "--split", str(acceptance_split), "--test-dir", str(acceptance), "--reports", str(root / "explanations"))

    report = Path("reports") / run_id
    report.mkdir(parents=True, exist_ok=False)
    for source in [acceptance / "test_metrics.json", acceptance / "test_slice_metrics.csv",
                   experiment / "validation_candidate_metrics.csv", experiment / "freeze_manifest.json",
                   root / "test_results.txt", protocol_path, acceptance_data / "manifest.json"]:
        shutil.copy2(source, report / source.name)
    for name in ["high_risk_cards.md", "high_risk_cards.json", "false_positive_card.json",
                 "explanation_manifest.json", "shap_feature_importance.csv"]:
        source = root / "explanations" / name
        if source.exists():
            shutil.copy2(source, report / name)
    shutil.copytree(root / "explanations/figures", report / "figures")
    predictions = pd.read_parquet(acceptance / "test_predictions.parquet")
    positives = predictions.loc[predictions["unscheduled_removal_within_30_cycles"].eq(1)]
    events = positives.groupby("removal_event_id").agg(
        detected=("final_alert", "max"), component_id=("component_id", "first"),
        aircraft_id=("aircraft_id", "first"), part_number=("part_number", "first"),
    )
    events.to_csv(report / "event_detection.csv")
    metrics = json.loads((acceptance / "test_metrics.json").read_text(encoding="utf-8"))
    freeze = json.loads((experiment / "freeze_manifest.json").read_text(encoding="utf-8"))
    evidence = metrics["acceptance_evidence"]
    outcome = "通过" if metrics["sla_met"] else "未通过"
    summary = f"""修复后实验与独立区间验收 · {run_id}

**验收结果：{outcome}。** 本次使用新随机种子生成的模拟机队，实际评分区间为 {evidence['first_snapshot']} 至 {evidence['last_snapshot']}。

| 指标 | 结果 | 预登记门槛 |
| --- | ---: | ---: |
| 事件召回率 | {metrics['event_recall']:.2%}（{metrics['detected_events']}/{metrics['actual_unscheduled_events']}） | ≥80% |
| 每 100 条部件评分记录的警报数 | {metrics['alerts_per_100']:.4f} | ≤5 |
| 超额批次 | {metrics['over_capacity_batches']}/{metrics['batches']} | 0 |
| 事件数量 | {metrics['actual_unscheduled_events']} | ≥{protocol['minimum_events']} |
| PR-AUC | {metrics['pr_auc']:.6f} | 描述性指标 |
| 已知结局警报精确率 | {metrics['precision']:.2%} | 描述性指标 |
| 首次警报提前量中位数 | {metrics['median_first_alert_lead_cycles']} 循环 | 描述性指标 |
| 结局待定警报 | {metrics['pending_outcome_alerts']} | 计入警报负担 |

飞机 {evidence['aircraft']} 架，部件 {evidence['components']} 个，评分记录 {evidence['rows']:,} 条。

模型：`{metrics['candidate_id']}`；冻结阈值：`{metrics['threshold']}`；重复警报抑制：30 循环；逐批容量：5%。

独立性证据：旧数据最晚评分时点为 {evidence['previous_latest_snapshot']}；新数据种子为 {evidence['seed']}；飞机、部件、装机、快照、拆换事件与旧数据的交集均为 0。生成配置、区间、门槛和 {protocol['bootstrap_iterations']} 次重采样已登记于冻结清单；模型冻结于 {evidence['frozen_at_utc']}，新数据生成于 {evidence['generated_at_utc']}。

策略选择保留 {freeze['policy_labeled_records']} 条可用标签，将 {freeze['policy_pending_records']} 条结局未成熟记录保留在评分人口中并隐藏其标签。冻结时点为 {freeze['policy_selection_cutoff']}；最晚使用标签为 {freeze['policy_latest_used_label']}。

聚类重采样 95% 区间：事件召回率 {metrics['bootstrap_95_percent_interval']['event_recall']}；警报比例 {metrics['bootstrap_95_percent_interval']['alert_rate']}。独立事件假设下的 Wilson 参考区间为 {metrics.get('event_recall_wilson_95_iid_reference')}，用于补充小样本说明。

本次修复包括当前读数的可用时间过滤、策略标签成熟时间控制、特征允许清单、逻辑回归解释、重采样权重、缺失评分兜底、常量 PSI 和冻结产物核验。测试输出见 `test_results.txt`；候选比较、分组结果、逐事件检出和解释卡保存在本目录。

验收阶段仅评分一次，参数与阈值沿用冻结产物。结果适用于当前合成机制；真实机队数据的验证仍需单独开展。

复现命令：`uv run python -m src.rebuild_experiment --run-id 新实验名称`。完整模型、预测和生成数据位于 `artifacts/runs/{run_id}/`。
"""
    (report / "summary.md").write_text(summary, encoding="utf-8")
    print(f"Acceptance {outcome}: {report / 'summary.md'}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild corrected models and run preregistered independent acceptance")
    parser.add_argument("--run-id", default="corrected_v2")
    parser.add_argument("--protocol", default="configs/acceptance_2027.json")
    args = parser.parse_args()
    rebuild(args.run_id, args.protocol)


if __name__ == "__main__":
    main()
