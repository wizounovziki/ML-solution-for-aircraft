# 项目内容与产物分类

| 目录或文件 | 用途 |
| --- | --- |
| `README.md` | 项目入口、主要结果图、提交材料与复现命令 |
| `reports/final_report.md` | 对照五项输出要求的提交主报告，含 7 张结果图 |
| `reports/figures/` | 当前报告直接引用的 PNG 图表 |
| `reports/analysis/` | 图表数据、解释案例及输入/输出哈希清单 |
| `reports/corrected_v2/` | 已完成独立验收的紧凑证据：指标、候选比较、冻结配置、逐事件结果、原解释卡及测试记录 |
| `notebooks/independent_acceptance_2027.ipynb` | 当前已执行的验收 Notebook |
| `src/`、`tests/`、`configs/` | 冻结实验使用的实现、测试与配置 |
| `tools/build_submission_report.py` | 从已保存预测生成图文报告；位于冻结训练实现之外 |
| `data/mro_dataset.csv`、`DATASET.md` | 随仓库提交的开发数据与字典 |
| `docs/design/` | 早期需求与技术选型材料；其中的旧实验数值属于历史设计上下文 |
| `docs/reviews/` | 修复前的问题审阅记录 |
| `docs/implementation-checklist.md` | 开发过程清单 |
| `archive/v1/` | 早期报告、图表、预测输出和 Notebook |
| `artifacts/runs/corrected_v2/` | 本地完整模型、原始数据、特征、划分和预测，保留原位置供哈希复核 |
| `artifacts/archive/v1/` | 本地旧实验、旧特征、旧测试和打包文件 |
| `artifacts/archive/intermediate/` | 重建前的临时特征与划分副本 |
| `data/generated/` | 本地生成的开发数据表 |

本次整理将原先散落在仓库根目录的设计与开发文档移入 `docs/`，将 `reports/` 根目录的旧版结果与旧 Notebook 移入 `archive/v1/`。旧 PR 图从 Notebook 的已保存输出中提取，和原 SHAP 图一起保留在历史材料中。

`artifacts/`、`data/generated/`、虚拟环境和缓存由 `.gitignore` 排除。当前验收运行位置保持为 `artifacts/runs/corrected_v2/`，对应冻结清单中保存的路径。

生成新的完整实验：

```sh
uv sync --locked
uv run python -m src.rebuild_experiment --run-id reproduction_v2
uv run python tools/build_submission_report.py --run-id reproduction_v2 --output reports/reproduction_v2/final_report.md
```

使用已有 `corrected_v2` 预测重建当前图文报告：

```sh
uv run python tools/build_submission_report.py
```
