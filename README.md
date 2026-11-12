# 航空部件非计划拆换预测 PoC

预测在役部件是否会在未来 30 个飞行循环内发生非计划拆换。业务门槛：事件级召回率至少 80%，每 100 个参评部件最多 5 次最终警报；采用逐轮硬容量时，每轮均不得超限。

## 当前状态

已实现：

- `data/ + src/ + notebooks/ + tests/` 工程结构和 `.cursorrules` 防泄漏规则。
- Mock Dataset 0.2.0：八张规范化表、合并表、CSV/Parquet、UTC 时间及文件哈希。
- 装机区间、CSN/CSO、飞机/部件累计循环、记录 `available_at` 和标签 `label_available_at`。
- 飞机—部件连通组、全局时间截点、标签成熟清洗及持久化划分清单。
- 因果 5/15/30 循环滚动特征、缺失指示和模型输入排除清单。
- 逻辑回归与 LightGBM 模型封装、原生类别映射及冻结模型 sigmoid/isotonic 校准。
- 事件召回、每轮 `floor(0.05 × N)` 硬容量、30 循环重复警报抑制和容量延后留痕。
- LightGBM 原始 log-odds SHAP、高风险原因卡、冷启动混合评分和 PSI。
- 12 项自动化测试。

**尚未完成完整数据规模下的双模型训练、验证模型选择、冻结测试和最终业务指标。当前结果不能解释为模型已经达标。**

## 环境

- Python 3.12，由 `uv` 管理。
- 主要依赖：pandas、PyArrow、scikit-learn、LightGBM、SHAP、Matplotlib。
- 依赖和开发测试工具锁定于 `uv.lock`。

```bash
uv sync
```

## 数据生成、验证与划分

```bash
uv run python data/generate_mock_data.py \
  --config configs/mock_data.yaml \
  --output data/generated/mock_v0_2_0 \
  --combined-output data/mro_dataset.csv

uv run python -m src.validate_data --data-dir data/generated/mock_v0_2_0

uv run python -m src.split_data \
  --data-dir data/generated/mock_v0_2_0 \
  --output artifacts/splits/mock_v0_2_0
```

默认时间边界：

- 训练：早于 `2024-09-01`，且标签在该点前可用。
- 校准：`2024-09-01` 至 `2025-03-01`，且标签在校准截止前可用。
- 警报策略：`2025-03-01` 至 `2025-09-01`。
- 测试：`2025-09-01` 及以后，仅用于冻结后的最终评估。

外层训练、验证、测试的飞机和部件互斥；校准与策略属于同一外层验证分区。

## 测试与展示

```bash
uv run pytest -q
```

`notebooks/poc_evaluation.ipynb` 用于 EDA、划分证据及后续冻结实验报告。Notebook 不承担核心算法实现。

## 文档

- [需求、架构与技术设计](mro-requirements-and-concept-design.md)
- [实施与验收清单](mro-TODO.md)
- [Mock Dataset 数据字典](DATASET.md)

公开参考包括 NGAFID、NASA C-MAPSS/N-CMAPSS、FAA SDR、MaintNet 及相关开源项目。其标签与本题不同，不与 Mock 主指标混合。

## 代码仓库

提交仓库：<https://github.com/wizounovziki/ML-solution-for-aircraft>

项目代码与提交记录以该仓库为准；公开提交前应检查密钥、个人信息、数据下载链接和最终复现结果。

## 安全与局限

合成数据只用于流程验证。风险评分仅辅助维修检查排序，不自动作出拆换、放行或适航决策。真实部署需要独立真实数据验证、标签成熟管理、业务审批、模型回滚及与 AMOS/Maximo 等系统的授权集成。
