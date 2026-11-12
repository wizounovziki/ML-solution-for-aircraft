# MRO Mock Dataset 数据字典

版本：`0.2.0`  
生成种子：`20261112`  
时区：UTC  
预测窗口：快照后 `(0, 30]` 个飞行循环  
评分间隔：每 5 个飞行循环  
评分批次：快照 UTC 日期对应的统一机队批次

## 1. 数据性质与规模

数据由 `src/mock_data.py` 完全合成，入口为 `data/generate_mock_data.py`。不包含真实航空公司、飞机、部件或人员信息，不代表真实物理模型、拆换率或航空安全水平。

| 内容 | 数量 |
| --- | ---: |
| 飞机 | 60 |
| 部件/安装记录 | 723 |
| 快照、传感器及合并样本 | 114,989 |
| 故障事件 | 3,683 |
| 维修事件 | 2,133 |
| 拆换事件 | 484 |
| 其中非计划拆换 | 404 |
| 成熟标签 | 113,090 |
| 正例快照 | 2,424（2.143%） |
| 删失快照 | 1,899 |

当前每个部件序列号只对应一次装机；`installation_id` 和划分器已支持同一部件多装机及跨飞机流转，实际流转压力场景将在后续数据版本补充。

## 2. 文件

八张规范化表和一张合并表均提供 CSV 与 Parquet：

- `aircraft`
- `components`
- `component_snapshots`
- `sensor_readings`
- `fault_events`
- `maintenance_events`
- `removal_events`
- `labels`
- `mro_dataset`

版本目录还包含 `generation_config.yaml` 和 `manifest.json`，后者记录规模及 CSV SHA-256。根目录 `data/mro_dataset.csv` 是便于交换的合并文件。

`removal_events`、`labels`、`label_available_at` 和 `removal_event_id` 属于目标/评估信息，不得作为模型输入。`components.remove_cycle` 是未来元数据，同样禁止进入特征。

## 3. 通用时间契约

- `event_time`、`reading_time`：事件实际发生时间。
- `available_at`：系统可读取时间，可能晚于实际发生时间。
- `label_available_at`：标签可以合法用于拟合的时间。

历史记录必须同时满足：

```text
event_time <= snapshot_time
available_at <= snapshot_time
```

正例标签在实际拆换登记可用时成熟；负例在完整观察 30 循环后成熟；删失记录的 `label_available_at` 为空。训练和校准拟合只能使用其截止点前已经成熟的标签。

## 4. 表结构

### 4.1 `aircraft`

| 字段 | 含义 |
| --- | --- |
| `aircraft_id` | 飞机唯一编号 |
| `aircraft_model` | 飞机型号，含 `B737-CLASSIC` 老旧机型 |
| `entry_service_date` | 投入服务日期 |
| `age_cycles_at_simulation_start` | 模拟开始前的累计循环 |
| `operating_environment` | 温带、高温、寒冷、高湿或高温沙尘 |
| `is_legacy_aircraft` | 是否应用额外结构性缺失 |
| `fleet_group` | 模拟机队分组 |

### 4.2 `components`

| 字段 | 含义 |
| --- | --- |
| `component_serial` | 部件全寿命唯一序列号 |
| `installation_id` | 一次装机至拆下的唯一区间 |
| `component_family` | 部件大类 |
| `component_model` | 部件型号，后期可能出现 `*-NEXT` |
| `manufacture_batch` / `manufacture_date` | 制造批次和日期 |
| `install_aircraft_id` / `install_cycle` | 装机飞机及飞机循环 |
| `cycles_before_install` | 本次装机前的部件全寿命循环；当前为 0 |
| `installation_number` | 该模拟位置的安装序号 |
| `remove_cycle` | 已观察拆换循环；属于目标元数据 |

### 4.3 `component_snapshots`

| 字段 | 含义 |
| --- | --- |
| `snapshot_id` | 快照唯一编号 |
| `aircraft_id` / `component_serial` / `installation_id` | 飞机、部件和装机区间键 |
| `component_family` / `component_model` | 部件属性 |
| `snapshot_cycle` | 评分时飞机累计循环 |
| `snapshot_time` / `available_at` | 快照时间和系统可用时间 |
| `aircraft_total_cycles` | 飞机累计循环，当前与 `snapshot_cycle` 相同 |
| `aircraft_age_cycles` | 包含模拟开始前任期的飞机年龄循环 |
| `component_total_cycles` | 部件全寿命累计循环 |
| `component_flight_hours` | 部件累计飞行小时 |
| `cycles_since_install` | 本次装机后的使用循环 |
| `operating_environment` / `ambient_temperature` | 运营环境和当次环境温度 |

### 4.4 `sensor_readings`

| 字段 | 含义 |
| --- | --- |
| `snapshot_id` / `component_serial` | 对应快照和部件 |
| `reading_cycle` / `reading_time` | 实际读数循环和时间 |
| `available_at` | 读数可用时间，少量记录迟报 |
| `vibration_rms` | 振动；退化末期上升 |
| `egt_margin` | 发动机排气温度裕度；退化末期下降，非发动机结构性为空 |
| `sensor_temperature` | 合成温度 |
| `sensor_pressure` | 合成压力 |
| `sensor_current` | 合成电流，后期含漂移 |
| `sensor_quality` | `VALID` 或 `DEGRADED` |

包含约 5% MCAR 缺失、老旧飞机额外结构性缺失和传感器尖峰。真实极端退化信号不得仅因数值较大而直接删除。

### 4.5 `fault_events`

字段包括 `fault_event_id`、快照/飞机/部件/装机键、`event_cycle`、`event_time`、`available_cycle`、`available_at`、`fault_code` 和 `severity`。

`available_cycle` 以部件全寿命循环表示，用于近期窗口计数；`available_at` 是实际系统时间。故障代码与部件族相关，急性退化期故障概率提高。

### 4.6 `maintenance_events`

字段包括 `maintenance_event_id`、关联键、实际及可用循环/时间、`event_type` 和 `action_code`。事件类型为 `INSPECTION` 或 `REPAIR`，动作为 `ROUTINE_CHECK` 或 `CORRECTIVE_ACTION`。

### 4.7 `removal_events`

| 字段 | 含义 |
| --- | --- |
| `removal_event_id` | 唯一拆换事件，事件级召回按此计数 |
| `component_serial` / `installation_id` / `aircraft_id` | 关联键 |
| `removal_cycle` / `removal_time` | 实际拆换循环和时间 |
| `available_at` | 拆换登记可用时间 |
| `removal_type` | `UNSCHEDULED` 或 `SCHEDULED` |
| `reason_code` | `DEGRADATION` 或 `LIFE_LIMIT` |
| `confirmed_failure` | 是否确认故障；含少量未确认故障 |

### 4.8 `labels`

| 字段 | 含义 |
| --- | --- |
| `snapshot_id` | 一快照一条标签 |
| `target_unscheduled_removal` | `1`：30 循环内非计划拆换；`0`：成熟负例；空：删失 |
| `is_censored` | 观察不足或窗口内计划拆换竞争事件 |
| `removal_event_id` | 仅正例关联拆换事件 |
| `label_available_at` | 标签合法可用时间 |

正例条件：

```text
0 < removal_cycle - snapshot_cycle <= 30
AND removal_type == "UNSCHEDULED"
```

### 4.9 `mro_dataset`

合并快照、传感器、标签和飞机属性，并增加：

- 框架别名：`component_id`、`part_number`、`flight_date`。
- `csn = component_total_cycles`，`cso = cycles_since_install`。
- `unscheduled_removal_within_30_cycles`：目标别名。
- `scoring_batch_id`：UTC 日期评分批次。
- `fault_code_count_last_10_cycles`：当时已可见的最近 10 部件循环故障数。
- `maintenance_count_last_30_cycles`：当时已可见的最近 30 部件循环维修数。

合并表仍包含目标元数据，建模必须通过 `src/features.py` 的排除清单和共享时点构建器处理，不能直接把全部列作为输入。

## 5. 合成机制与限制

- 寿命按部件族的对数正态分布生成，高温、高湿和沙尘环境提高风险。
- 拆换前 20–35 循环存在指数式急性退化。
- 后期出现新型号和电流漂移，用于冷启动与漂移测试。
- 当前不包含真实跨飞机流转，也没有单独的大修循环重置。
- 合成机制会让部分信号比真实数据更易学习，模型结果不能外推为真实 MRO 效果。

## 6. 复现与验证

```bash
uv sync
uv run python data/generate_mock_data.py \
  --config configs/mock_data.yaml \
  --output data/generated/mock_v0_2_0 \
  --combined-output data/mro_dataset.csv

uv run python -m src.validate_data --data-dir data/generated/mock_v0_2_0
uv run python -m src.split_data --data-dir data/generated/mock_v0_2_0
```

验证覆盖主外键、装机循环核算、事件实际/可用时间、标签成熟、30 循环标签、删失、拆换后在役记录、非计划事件覆盖、1.8%–2.2% 正例率、缺失和合并表一致性。
