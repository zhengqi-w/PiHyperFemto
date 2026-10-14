# 流式处理、独立 WP 与 compact scored

默认 workflow 为 ROOT 候选分块训练 → Hyper 分块 WP → Femto/MCFemto 分块推理和裁剪 → compact scored → nominal/systematics/MC → ROOT/PDF。原始 ROOT 始终是重建来源。取消 prepared 数据表会减少磁盘写入；训练或 WP 缓存失效时则重新读取候选 ROOT，不再依赖完整候选缓存。

## 配置入口

两份示例配置均已采用以下默认布局：

```yaml
io:
  chunk_rows: 100000
  root_projection: true
  persist_prepared: false
  cache_model_scores: false
  scored_partition_by_pt: true
  slim_scored: true
  scored_columns: [mass_h3l, kstar, pt_h3l]
  scored_optional_columns:
    - HypTrackOccupancy
    - PairCentFT0C
    - MixingDepth
    - IsClosePairRejected
  checkpoint: true
training:
  efficiency_min: 0.29
  efficiency_max: 0.99
  npoints_for_effi: 701
working_point:
  signal_efficiencies: null
  objective: pol0_expected_significance
  yield_eff_range: [0.5, 0.95]
  min_pol0_points: 3
  max_fit_chi2_ndf: 10
  save_score_fit_frames: false
  centrality_bins: null
  candidate_cuts:
    cos_pa: {min: 0.99}
    ct: {min: 0, max: 40}
    NSigmaHe: {abs_max: 5}
  event_cuts:
    ZPrimVtx: {abs_max: 10}
  mass_range: [2.96, 3.04]
  sidebands: [[2.96, 2.98], [3.005, 3.04]]
  mass_fit:
    signal_model: dscb
    background: pol1  # pol1 / pol2
    mass_bins: 50
    mc_mass_bins: 80
    min_mass_entries: 80
    sigma_bounds: [0.0014, 0.002]
    mean_bounds: [2.985, 3.005]
    calibrate_mc_tails: true
    sideband_prefit: true
  storage:
    enabled: true
    reserve_signal_efficiency: 0.99
```

`scored_columns` 是用户明确要求的输出列，必须能从输入或派生计算获得；`scored_optional_columns` 是输入存在时才保存的 QA 列。示例中的 H3L mass、k*、pT 对应内部列 `mass_h3l`、`kstar`、`pt_h3l`。可加 `ct`、`cos_pa` 等后续需要的量。ITS 平均 cluster size 等派生输出会请求原始 packed 字段，解码后只保存派生结果。

仅保存这三个物理量无法重建四电荷道、中心度和 SE/ME，也无法改变 WP 或 PID cut。因此执行器自动补齐：

- `HypCentFT0C`、`ChargeChannel`、`IsMixed`、`bdt_score`。
- 分 pT 模式的 `bdt_model_id`，以及适用的路由状态。
- nominal 和全部已声明 systematic 的 candidate/event/pair cut 字段、所需 CPR 和 mixing depth。
- MCFemto 可用 truth QA 与 `KstarMC`；仅用于计算它的六个生成动量分量默认丢弃。

只用于 BDT 的拓扑变量、daughter 原始动量、来源 entry 字段不默认写出；若同时用于最终 cut 或显式输出，就必须保留。`dataset.json` 列出实际保存列，便于检查。schema 不一致明确失败，不静默补零拼接。

## training 和 WP 的职责

旧 `qa_signal_efficiencies` 仅生成固定参考效率 QA；`qa_centrality_bins` 仅生成中心度诊断，不定义中心度模型。两项已删除，旧配置残留会明确报未知字段。当前训练和 WP 默认不按 Femto 中心度限制候选，Femto 分析仍按中心度分组。

训练使用 HIPE4ML 在固定 held-out 两类 label/score 上产生密集效率数组，效率目标的生成规则与 spectrum 一致：`unique(round(linspace(min, max, npoints), 3))`。默认 0.29–0.99、701 个请求点。每个 pT bin 可以覆盖这三项设置。`score_efficiency_array.txt` 是 N×2，第一列 score、第二列 efficiency；`metrics.json` 还保存实际效率、数组定义和信号分母。小样本 ties 可能使多个效率对应相同 score。

WP 默认消费这组数组；显式 `signal_efficiencies` 可选扫描子集。在独立 Hyper 重建候选上逐阈值拟合 DSCB + pol1/pol2，将动态 3σ raw signal / requested efficiency 在 `yield_eff_range` 内做 ROOT pol0 得到 `c0`，按 `ε*c0/sqrt(c0+B3)` 选择 nominal。同一 score 或逐模型 score 映射用于 Femto/MCFemto。训练与 WP QA 在各自 `qa/` 保存 ROOT/PDF，report 同步四张当前 WP PDF 和相同 ROOT canvas。算法和质量检查见 [工作点说明](working_point.md)。

WP 的 `candidate_cuts`、`event_cuts`、`mass_range`、`signal_window`、`sidebands`、`purity` 可以单独设置；`null` 继承相应 analysis/purity 设置。sidebands 必须在 WP 自身质量范围内、位于信号窗左右。`centrality_bins: null` 表示不筛中心度。显式 event cuts 缺字段报错；兼容继承的可选 event cut 缺失会记录列名及缺失非空 batch 数，报告不会掩盖部分文件缺列。WP 实际设置保存在 `resolved_settings`。

## 裁剪的方向与系统覆盖

效率越高，score cut 越低。因此最大预留效率对应最低保留阈值，写入采用 **score >= threshold**。`reserve_signal_efficiency: 0.99` 提前允许在 nominal 周围向宽松方向移动，而无需为每次扫描保留全部背景。

每个模型的实际存储 floor 取以下阈值的最小值：预留效率、nominal、所有 `wp_efficiency_delta`、所有显式 systematic BDT thresholds。若 nominal 或变化更宽松，自动扩大存储范围；不是机械截在 0.99。需覆盖效率 1 时 floor 为 0，可能没有 score 裁剪收益。无效 score 与不在训练 pT 范围内的行不写出。

```yaml
systematics:
  variations:
    - name: bdt_looser
      group: bdt
      wp_efficiency_delta: 0.02
      overrides: {}
    - name: bdt_tighter
      group: bdt
      wp_efficiency_delta: -0.02
      overrides: {}
```

delta 是 held-out MC efficiency 的绝对变化。每个模型的 nominal actual efficiency + delta 必须在 `(0,1]`；超界报错。floor 写入 `working_point/summary.json` 和 role 的 `scored/.../dataset.json`。保留 ties 后实际效率可能略高于请求效率。

名义和已声明变化接受的 pair、SE/ME、sumw2 和 CF 协方差不受此裁剪影响。scored cutflow 的起始行数已是裁剪后分母，不能解释成原始输入效率；使用 scored summary 的读入/裁剪计数和独立 raw QA 查原始样本。

## 保存目录与内存

```text
prepared/
  inventory.json / summary.json / manifest.json
training/
  model_handler.pkl / metrics.json / validation_scores.npz / score_efficiency_array.txt
  qa/training.root / *.pdf
  models/<id>/...          # 分 pT 模式，包含自己的 qa/
working_point/
  summary.json / scan.csv / scan_fits.json / best_fit.json / ...
  qa/working_point.root / *.pdf
  models/<id>/...          # 分 pT 模式
scored/
  summary.json / manifest.json
  femto/
    dataset.json
    <id>/part-000000.parquet  # 分 pT；global 模式直接位于 femto 下
  mc_femto/...
.checkpoints/scored/...
```

实际 metadata 文件布局由阶段执行器生成。新流程不写 prepared/validation Parquet。`validation_scores.npz` 保存 held-out label、原始 margin score 和 score_domain；训练 QA 的其他统计量已在 metrics 中，不保留完整候选表。ROOT 读取、派生和推理按 chunk 执行，训练样本受 `max_per_class` 上限约束，WP 仅累计有限个质量 histogram。降低 `chunk_rows` 可减少批次内存，代价是更多 I/O 调度；不承诺某个固定 RSS。

全部后续 Parquet 读取使用递归发现，支持 pT 子目录。scored checkpoint 与提交产物用 hard link，不复制完整数据表；分片恢复先验证该 chunk 的全部 pT 文件，避免恢复一半。默认关闭逐模型 score 缓存；显式开启时用稀疏 NPZ，允许局部重训复用其他模型分数。

## 后续修改与历史清理

修改 WP 且实际 floor 不变时，scored 可复用；修改 floor、模型、输出列或保存范围时重新生成。minimum_threshold=null 表示所有有限 in-scope margin 都保留；旧或缺少 score-domain metadata 的数据必须重新 apply。直接用裁剪后文件计算超出 floor 的宽松选择会报错，不能返回少算的 CF。新增未保存字段、扩大质量/k*/中心度范围也需要原始 ROOT。

`persist_prepared: true` 是旧流程兼容开关，会重新持久化中间表；其 candidate 训练同样不按 Femto 中心度筛选。日常生产建议保持默认 false。

历史 `outputs` Parquet 清理清单见 [scored_storage_cleanup.json](../reports/validation/scored_storage_cleanup.json)。旧 held-out validation 表已迁移为 label/score NPZ，既有模型、JSON/CSV、ROOT/PDF 保留为历史产物。随后已按用户要求用当前 configs/testset.yaml 从原始 ROOT 重跑完整 TestSet，389/389 项产物检查通过。该次历史 scored 在 outputs/testset/scored/femto，保存 1,996,515 行、18 列；prepared 仅保留元数据。随后新增 HIPE4ML/pol0 QA 流程；模型迁移前的 TestSet 在默认 pol0 范围没有有效点，该次流程停在 WP，原 scored/report 保留为之前版本，不能当作新算法的成功结果。未运行 50 GB full period。检查记录见 [TestSet compact validation](../reports/validation/testset_compact_validation.json)，无用目录清理见 [cleanup](../reports/validation/testset_compact_cleanup.json)。

## 多 period 布局

每个 period 独立保存 `<output>/periods/<id>/scored/femto/` 和 `scored/mc_femto/`；分 pT 时再进入所属 model 子目录。顶层 `scored/dataset.json` 是引用 catalog，不能当作一个使用统一 BDT threshold 的表，也不能当作缺失 MC 的 role 数据集。汇总直接累加各 period 已筛选 histogram，详见 [period 工作流](periods.md)。
