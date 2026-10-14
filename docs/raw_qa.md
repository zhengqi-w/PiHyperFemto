# BDT 前的输入检查

原始 QA 独立于 preparation、训练、WP 和关联函数计算，直接分块读取 Femto 与 Hyper。它只使用 producer 已保存的条目和 `analysis.centrality_bins` 的左闭右开中心度区间，不施加分析质量窗、candidate/pair/event cuts、CPR、训练 sideband 或 BDT 分数。producer 在写树前已经施加的选择无法通过离线 QA 撤销。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pih3l raw-qa -c configs/testset.yaml
```

命令只读取 Femto/Hyper，无需 MCHyper/MCFemto、模型或 WP；完整 `run` 的 report 依赖这个阶段，并把相同的 QA 收录到 `report/analysis.root` 和 HTML 中。单独执行 `raw-qa` 不重建旧 report 或其他阶段。完整命令仍遵循缓存失效和依赖重建规则。

| 检查 | 分组与口径 | 用途 |
| --- | --- | --- |
| Femto k* | 中心度 × 四电荷道 × SE/ME，单位 pair 行计数 | 检查输入覆盖、SE/ME 形状、空道及截断 |
| Hyper H3L invariant mass | 中心度，独立重建候选，合并 matter/antimatter | 查看训练背景树在 BDT 前的实际质量覆盖 |

k* 由共用运动学模块计算，使用 `physics.pair_mass` 约定，默认名义 H3L 质量。Hyper 质量由 He3 和衰变 pion 四动量重建。不输出 Femto pair 的原始质量 QA：ME 的候选重复随 mixing depth 增加，该谱不提供独立的候选质量检查。没有追踪或去重任何 H3L ID。

显示范围由 `raw_qa.mass_range`、`mass_bins`（仅用于 Hyper 质量）和 `kstar_edges` 单独控制，默认质量 2.94–3.06 GeV/c²、120 bins，k* 0–1.2 GeV/c、60 bins。改变显示范围不会引入物理选择。SE/ME 不进行归一化，不调用 signal fit，也不估计 significance 或 CF。

输出位于 `<output>/raw_qa/`：

- `raw_kstar_cent0.pdf`、`raw_kstar_cent1.pdf`：每张四道 SE/ME 绝对计数，log y。
- `raw_hyper_mass.pdf`：两个中心度的独立 Hyper 质量，linear y。
- `raw_qa.root`：TH1D、Poisson sumw2、可重画 TCanvas 及口径元数据；不生成 PNG。
- `results.json`：全部 histogram 边界、计数与范围外/非有限值统计；`summary.json`：精简口径与 PDF 清单；`manifest.json`：输入、配置及产物 hash。

所有 histogram 使用 `[low, high)`。有限值低于/高于显示区间分别进入 ROOT underflow/overflow，非有限值只计入 JSON。每个分布满足 `rows = sum(counts) + underflow + overflow + nonfinite`；ROOT Entries 为有限行数（包含 flow）。积分不一定等于 Entries，不能把范围外条目误判为输入丢失。

代码职责：`raw_qa.py` 只负责读入和数值聚合；`raw_qa_output.py` 负责保存与绘图；`root_plot.py` 提供共用布局；`pipeline.py` 管理执行、缓存和事务；`report.py` 组织报告。新增 QA 应复用这些接口，避免复制 preparation 的隐式筛选或在 report 中重新读输入树。

软件检查覆盖中心度边界、四道与 SE/ME、chunk 不变性、完整计数、缺失训练数据/特征时仍能运行，以及重新打开 ROOT 文件读取计数与 canvas，并确认 Femto 质量 QA 不再输出。它们不构成 TestSet 物理 validation。
