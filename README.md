# π–Hypertriton Femto Analysis

面向新版 compact AO2D 的 π–H3L 离线分析框架：独立候选 BDT 训练、Hyper 重建质量拟合选取工作点、同一工作点逐 pair 应用、分中心度四电荷道关联函数、质量纯度与 sideband 修正、系统误差、MC QA，以及可复跑的阶段产物和 HTML 报告。**物理分析图由 PyROOT 绘制，ML QA 使用 HIPE4ML 官方 plot_utils；training/WP 分别保留数值 ROOT 文件，关联结果汇总至 `report/analysis.root`。** 图形文件只导出 PDF，不生成 PNG。

**当前输入约定：Femto 不需要 hypertriton ID，不重建候选指纹。** BDT 在独立 `Hyper` / `MCHyper` 候选树上训练，在 Femto 每条 pair 的候选特征上推理。测试配置保留 `0–10%`、`10–50%` 和全部四个电荷道。

## 快速运行

Farm 单 period 的安装、预检和运行见 [Farm 部署](docs/farm.md)。安装完成后只需修改 [单 period 配置](configs/farm.single_period.example.yaml) 的副本：

```bash
bash scripts/install_environment.sh
conda activate pih3l-femto
cp configs/farm.single_period.example.yaml configs/farm.yaml
# Edit configs/farm.yaml: input roots, output, selections, training/WP/systematics.
pih3l doctor -c configs/farm.yaml
pih3l run -c configs/farm.yaml
```

依赖版本由 [environment.yml](environment.yml) 与 package 声明管理，Linux CI 使用同一环境。`pih3l` 为已安装入口，不需要硬编码本机 Python 或 `PYTHONPATH`。

如果当前 Conda 环境已经安装了项目和依赖，可直接运行：

```bash
conda activate pih3l-femto
pih3l inspect -c configs/testset.yaml
pih3l run -c configs/testset.yaml
```

该环境已提供 PyROOT 6.38.04。ROOT 是图形与 ROOT 文件输出的必要环境依赖，缺少时明确报错；不会自动退回其他绘图库。

TestSet 输入路径记录在 `configs/testset.yaml`，默认输出为本项目的 `outputs/testset/`。四类输入分别使用 `inputs.femto/background/signal/mc_femto.root`，通过 `--output /absolute/path` 覆盖输出。`--data-root` 仅兼容未指定独立根的旧配置。相对路径以配置文件所在目录为基准。

`pih3l` 和 `python -m pih3l` 是等价入口。Farm 安装脚本会把项目作为普通 Python package 安装，因此从其他目录、批作业或容器入口都不依赖源码路径。

需要进入目录时自动激活环境，可复制 [.envrc.example](.envrc.example) 并按 [direnv 使用说明](docs/direnv.md) 配置本机或服务器的 Conda 路径。批任务使用 `direnv exec . python -m pih3l ...`，依赖环境仍需预先准备。

## 输入与处理链

| 数据源 | 用途 |
| --- | --- |
| `Femto` | 主要 SE/ME pair tree，构建数据 CF |
| `Hyper` | 独立候选树，从质量 sideband 取训练背景 |
| `MCHyper` | 只选重建、真信号、two-body 候选作为训练信号 |
| `MCFemto` | 可选 pair truth 数据，用于 primary fraction、truth 与 k* resolution QA |

```text
MCHyper signal + Hyper sideband ──► 候选 BDT
                                      │
Hyper 全质量区 ──► raw/eff 扫描 + pol0 ──► 最大 εc0/√(c0+B) 的 WP
                                      │ 同一 WP
                                      ▼
Femto ──► schema / 运动学 ──► 逐 pair 推理 ──► 0–10、10–50 × 四道
                                                │
                  ┌─────────────────────────────┼────────────────┐
                  ▼                             ▼                ▼
                raw CF                 pair-weighted 质量纯度   sideband CF
                  └─────────────────────────────┴────────────────┘
                                                ▼
                                    满足输入条件时计算 corrected CF
                                                │
                         系统变化 + MCFemto QA ──┴──► 结果与报告
```

| Channel | Pair |
| --- | --- |
| 0 | H3L + π⁺ |
| 1 | H3L + π⁻ |
| 2 | anti-H3L + π⁻ |
| 3 | anti-H3L + π⁺ |

中心度始终取超氚一侧事件。质量、ct、cosPA、衰变半径与 k* 由输入运动学重建；默认 k* 使用名义 H3L 质量，与当前 producer 一致。

## 阶段命令

每个下游命令会检查并执行其缺少或过期的依赖。缓存按阶段消费的配置、输入、代码、环境与产物 hash 验证；改变 WP 不重训模型；最低存储 score 未改变时可复用已有 pair 分数，超出保存范围时重新从 ROOT 推理。`--force` 会同时重建依赖阶段。大数据、断点续跑、分 pT 模型和实测效率见 [执行与性能](docs/performance.md)。

| 命令 | 工作 | 主要目录 |
| --- | --- | --- |
| `inspect` | 发现文件、DF 树和字段 | `input_inventory.json` |
| `doctor` | 环境、ROOT/PDF、配置/输入列及输出写入预检；不训练或扫描全量 pair | 控制台或 `--json` |
| `raw-qa` | BDT 前分中心度的四道 SE/ME k* 及独立 Hyper 质量 | `raw_qa/`（ROOT/PDF/JSON） |
| `prepare` | 默认登记输入与流式来源，不写数据表 | `prepared/`（JSON 元数据） |
| `train` | 候选训练与训练 QA 数据 | `training/` |
| `working-point` | Hyper 重建质量拟合扫描与最终 WP | `working_point/` |
| `apply` | WP 后分块推理，按系统预留 score 裁剪并保存精简列 | `scored/`（可按 pT 模型分目录） |
| `correlate` | 名义 8 组 histogram、CF、纯度与协方差 | `nominal/` |
| `systematics` | 重跑命名变化并按来源汇总 | `systematics/` |
| `mc-qa` | MC fraction / truth / resolution QA | `mc_qa/` |
| `report` | ROOT 图形、ROOT 对象文件与 HTML 报告 | `report/analysis.root`、`report/index.html` |
| `run` | 完整链条，最终生成报告 | 全部 |

例如只要更新名义结果，可运行：

```bash
pih3l correlate -c configs/testset.yaml
```

## 结果与数据限制

`nominal/correlations.csv` 提供每个 k* bin 的 SE、ME、raw/corrected CF、统计误差、有效标志与纯度状态；`results.json` 保存完整 histogram、质量拟合和状态；`nominal/covariances.npz` 保存统计协方差。`systematics/final_observables.csv` 汇总 raw/corrected 的 `stat_error`、`syst_error`、`total_error` 与有效标志，`systematics/covariances.npz` 保存对应系统及合计协方差。每个阶段有 `manifest.json` 记录配置、输入、环境、耗时和产物 checksum。

`report/analysis.root` 提供 `nominal/`、`final/`、`variations/`、`training/`、`working_point/`、`mixing_qa/`、`mc_qa/`、`metadata/` 和 `canvases/` 目录。无效 CF bin 保存 NaN 和 mask，作图 graph 只保留有效点。具体对象路径及重新打开 canvas 的例子见 [ROOT 输出说明](docs/root_output.md)。

这份汇总 ROOT 文件由 `report` / `run` 阶段生成；单独执行 `correlate` 时产生 CSV/JSON/NPZ 数值结果；`systematics` 还会在 `systematics/qa/` 生成各 variation、source 和总量的 ROOT/PDF QA。std 表示 nominal CF，variation 比值为 C_var/C_nom，相对系统误差为 sigma_syst/abs(C_nom)。随后运行 `report` 会链接系统 QA 并同步 ROOT 对象和 canvas。

当前 TestSet 已按修正后的保存位移重训并完整重跑：cosPA、ct、decay radius 直接使用 producer 已保存的 PV→SV 位移，训练使用五个实际输入，变量分布图另加仅供 QA 的 H3L mass。MC 严格取 2.95<mass<3.02，Hyper background 取区间外；score 为 HIPE4ML/XGBoost 的原始 margin，覆盖约 −11.4 到 10.3。training/WP、compact scored、关联、系统变化和 report 均使用新的模型与阈值。当前检查见 [margin/顶点软件回归](reports/validation/margin_vertex_software_checks.json)、[完整产物验收](reports/validation/testset_margin_vertex_validation.json)，producer 与真实 cosPA 对照见 [顶点审计](reports/validation/vertex_convention_audit.json)。

多 period 配置、独立训练/WP、计数合并与路径示例见 [多 period 工作流](docs/periods.md)。`configs/testset.periods.yaml` 以三个 id 重复 TestSet，输出 `outputs/testset_periods/`；各 period 的 compact scored 位于 `periods/<id>/scored/`，合并 nominal CF 位于顶层 `nominal/` 和 `report/`。三次重复输入仅是软件诊断，报告明确标注，不代表独立物理统计量。

701 个扫描点中 353 个有效，默认 pol0 区间内 313 点；测试 WP margin 为 0.4214570286160236、实际信号效率约 0.946016。pol0 χ²/ndf 约 101.533、MC DSCB χ²/ndf 约 92.127，常数归一化与 MC 质量形状的描述仍需物理研究；软件成功不证明拟合充分。八组 corrected CF 因 TestSet 旁带不足不可用，MCFemto 为空。此前 probability 分数的检查记录均为历史产物。未运行 50 GB full period 或正式物理 validation。

`training/score_efficiency_array.txt` 使用 HIPE4ML 从固定 held-out 两类样本产生密集 score/efficiency 映射。最终 WP 来自独立 Hyper DSCB 质量扫描：将动态 3σ fitted raw signal / requested efficiency 用 ROOT `pol0` 拟合为 `c0`，按 spectrum 的 `ε*c0/sqrt(c0+B3)` 选最大值；同一阈值原样应用到 Femto 与 MCFemto。完整算法、质量拟合差异和 QA 路径见 [工作点说明](docs/working_point.md)。`ct/fCt` 被配置验证明确禁止作为训练特征，避免 lifetime 分布 sculpting。

训练统一使用 HIPE4ML `ModelHandler + XGBoost`，`model_params` 配置树参数。两份示例配置都采用 H3l spectrum 的 XGBoost 起点；旧 backend/HGB 参数已删除，新模型保存为 `model_handler.pkl`。`use_optuna: true` 开启 training partition 内的分组 CV AUC 搜索，支持逐 pT 模型覆盖；最终 held-out 不参与调参，Hyper 质量拟合仍负责选择 WP。开关默认关闭，完整配置、依赖与优化 QA 产物见 [训练与 Optuna 说明](docs/training.md)。训练效率数组默认采用 spectrum 的 0.29–0.99 / 701 点网格；训练不按中心度筛选或分模型。默认不保存 prepared Parquet，仅保存 compact scored，并覆盖全部声明的 WP/systematic 变化，详见 [存储与独立 WP 配置](docs/scored_storage.md)。

本次 TestSet 的 Femto 质量范围约为 **2.980–3.000 GeV/c²**，没有配置所需的宽 sideband；`MCFemto` 目录为空。因此，测试可验证分中心度四道 raw CF 和离线系统变化，**不能由该输入给出可信的 sideband-corrected CF 或 primary fraction**。相关产物保留明确的 unavailable 状态，不用 P=1 或假 MC 数据补齐。输入证据见 [测试清单](reports/validation/testset_inventory.json) 和 [质量覆盖](reports/validation/femto_mass_support.json)。

统计误差包含 SE/ME 归一化引入的 bin 间相关，但采用独立 pair 的近似，未包含共享候选/事件的全部相关。质量拟合是 **SE pair-weighted purity**，不是唯一候选产额。测试配置中的系统变化是可运行的示范 ensemble，不代表已通过正式物理评审的完整系统预算。

## 工程结构

```text
configs/                 分析配置与系统变化
src/pih3l/               可导入的分析 package 与统一 CLI
  config.py              配置合并、物理范围及变化校验
  schema.py / io.py      AO2D 适配、分块 ROOT / Parquet、JSON
  dataset.py             可重复遍历的 ROOT 来源，投影、运动学和输入范围
  scored_storage.py      输出列、系统效率预留、pT 分区与范围检查
  validation_scores.py   held-out 标签/分数的紧凑 NPZ
  physics.py             四动量、质量、ct、cosPA、k*
  raw_qa.py               原始输入分布，只按中心度分类，不依赖训练
  raw_qa_output.py        原始 QA 的 ROOT 对象与 PDF
  training.py            候选 BDT、held-out 评估和密集效率映射
  efficiency.py          HIPE4ML 固定样本 score/efficiency 映射
  training_qa.py         官方 HIPE4ML ML QA PDF 与数值 ROOT
  training_settings.py   XGBoost 参数、ModelHandler factory 与严格校验
  training_optimization.py  可选 Optuna、内部 grouped CV 与搜索 QA
  models.py              分 pT 模型 registry、局部训练和推理路由
  working_point.py       流式 Hyper 质量谱、raw/eff pol0 与逐模型 WP
  wp_mass_fit.py         RooFit DSCB/Chebychev、MC tails 与物理计数协方差
  working_point_qa.py    独立 WP ROOT/PDF、完整扫描质量拟合
  execution.py           阶段配置/数据契约与缓存边界
  score_cache.py         逐模型分数断点与局部重训复用
  selection.py           共用切选、cutflow、ME 权重
  correlation.py         SE/ME histogram 与归一化协方差
  purity.py              质量纯度与 sideband 修正
  analysis.py            单个名义/变化分析与结果导出
  systematics.py         按来源组合系统误差
  systematics_qa.py      各 variation/source/total 的 ROOT QA
  multi_period.py       period 执行、独立缓存与原子汇总
  period_merge.py       可加统计量及 MC 的严格合并
  report_models.py      period/model QA 与 ROOT 命名空间
  benchmark.py           独立 RDF/Uproot、共享 variation 性能对比
  mc.py                  primary/species fraction、truth、resolution、closure
  root_io.py             ROOT 对象写入、sumw2/valid mask 与生命周期管理
  root_plot.py           共用 ROOT 布局、图例和 PDF 导出
  root_output.py         PyROOT 图形、ROOT 目录组织与 PDF 导出
  report.py              HTML 索引与状态汇总
  pipeline.py / cli.py   阶段依赖、缓存、锁和事务式输出
tests/                   运动学、统计、schema、训练和 I/O 回归测试
docs/                    架构、操作指南、数据与物理契约
reports/validation/      本次 TestSet 的独立验证证据
outputs/                 生成的中间数据、模型、结果与报告
```

运行自动化检查：

```bash
conda run --no-capture-output -n pih3l-femto python -m unittest discover -s tests -v
```

当前 DSCB/报告同步的[软件回归记录](reports/validation/dscb_wp_software_checks.json)覆盖合成完整 workflow、全局/分 pT、pol1/pol2、RooFit 物理协方差、独立 DSCB 积分、3σ argmax 和 ROOT/PDF。真实 TestSet 的[完整产物检查](reports/validation/testset_dscb_validation.json)核对所有阶段 SHA256/依赖链、全部质量扫描以及 report 四个 canvas 的精确内容一致。低统计 TestSet 不作为正式 BDT/WP 或物理结果 validation；输入摘要见 [TestSet inventory](reports/validation/testset_inventory.md)。GitHub Actions 使用与 Farm 相同的 Conda ROOT 环境；Linux 运行状态以远端 Actions 结果为准。

WP 质量拟合已改为直接计算归一化函数的平滑曲线，data 保留带误差 histogram；精确 bin 积分仍保存在 ROOT。此次绘图更新见 [曲线回归](reports/validation/wp_curves_software_checks.json)和[最新完整产物检查](reports/validation/testset_wp_curves_validation.json)，不改变扫描拟合和 WP 数值。

## 文档

- [工作点与独立 QA](docs/working_point.md)：HIPE4ML 数组、raw/eff pol0、扫描质量拟合及 report 外的 ROOT/PDF。
- [训练与 Optuna](docs/training.md)：ModelHandler 调用链、XGBoost 参数、grouped CV、分 pT 与官方 ML QA。
- [BDT 前输入检查](docs/raw_qa.md)：k* 与原始 H3L 质量的独立 QA、计数口径与运行方式。
- [操作指南](docs/user_guide.md)：配置、各阶段、结果读取、系统变化和常见状态。
- [大数据执行与性能](docs/performance.md)：精简/流式数据、缓存依赖、续跑、分 pT 与局部重训、附近 WP variation、RDF 实测比较。
- [生产配置示例](configs/production.example.yaml)：独立输入/输出、内容 hash 与可选 MCFemto；需要根据生产数据审定物理选择。
- [Farm 单 period](docs/farm.md)：依赖安装、唯一 YAML、非项目目录运行、批任务、bind 与预检。
- [架构与扩展](docs/architecture.md)：数据流、职责、复现方式和扩展边界。
- [ROOT 输出](docs/root_output.md)：对象路径、sumw2/valid mask、canvas 重画与环境依赖。
- [独立架构审查](docs/review_notes.md)：检查范围、已修复问题与验证边界。
- [新版 schema](docs/data_schema.md)：实际 ROOT 字段、树与 MC 输入约定。
- [物理与统计契约](docs/analysis_contract.md)：CF、纯度、误差、MC fraction 的准确含义。
- [原始 README 归档](docs/physics_workflow_reference.md)：历史物理流程参考；其 ID 和旧字段要求不适用于当前 compact 输入。
