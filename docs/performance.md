# 大数据执行、分 pT 模型与性能证据

## 已落地的执行模式

生产计算默认采用投影 Uproot → 分块运动学/BDT → compact scored Parquet → histogram → ROOT/PDF。训练和 WP 直接读取独立候选 ROOT，`prepared/` 只保存 JSON 来源元数据；训练用上限抽样，WP 累计 histogram。scored 保留观测量、pair 分类、全部已声明 cuts、模型路由和可用 MC truth QA，推理用的原始动量/额外特征在写入前丢弃。字段需求由配置自动汇总，不能在未保存的字段上默默运行。

```yaml
io:
  chunk_rows: 100000
  persist_prepared: false
  cache_model_scores: false
  scored_partition_by_pt: true
  scored_columns: [mass_h3l, kstar, pt_h3l]
  root_projection: true
  slim_scored: true
  checkpoint: true
  fingerprint: sha256
working_point:
  storage:
    enabled: true
    reserve_signal_efficiency: 0.99
```

写入 scored 前，按最高预留 MC efficiency 对应的最低 score 裁剪，并自动覆盖 nominal 与全部已声明系统变化；最终实际 floor 记录在 dataset.json。最高效率对应最宽松选择，使用 `>=` 保留 ties。压缩比例取决于模型分数，若必须覆盖效率 1 则可能无法按 score 压缩。输出列与 WP 配置详见 [紧凑存储](scored_storage.md)。

`chunk_rows` 控制 ROOT entry range 和推理 batch。现有 Parquet 若用了更大的 row group，读取一个小 batch 仍可能需要解码整个相交 row group；本框架新写的 shard 本身受 chunk 大小限制。运行保留 float64 运动学和 histogram 权重，不以降精度换吞吐。

WP 不再收集完整 Hyper 的 mass/score 数组。每批预测后累计各 score 阈值的整数质量 histogram，内存随阈值数和 mass bins 增长；只保留已受样本上限约束的 held-out 两类分数用于 HIPE4ML 效率插值。多个 pT 模型各自扫描 Hyper，内存有界，但扫描次数仍随模型数增长。

系统阶段为不同选择建立 `CorrelationAccumulator`，在一次 Parquet 读取中更新所有 accumulator；同一选择的不同拟合/归一化设置共享统计量。仅改变归一化或 mass background 的 variation 直接使用 nominal 的 histogram，无需重新读取 pair。修改 signal/sideband 窗口、选择、BDT WP 或 mass binning 时重新累计，避免从粗 mass histogram 近似恢复任意窗口。

`systematics/summary.json` 的 `execution` 记录实际数据扫描次数、重新累计的 histogram 组数和 nominal 复用数量。每个 variation 的结果目录、cutflow、mixing QA、协方差及按来源组合接口保留。

## 缓存依赖与断点

阶段 manifest 使用 `schema_version: 2`，记录该阶段消费的配置、源代码、环境和上游产物身份。新签名不要求不同阶段相同；依赖关系与 artifact checksum 才是 lineage。旧输出第一次进入新执行器会重建受影响阶段，因此请使用独立生产输出目录，不把历史产物当成新执行模式的结果。

| 修改 | 主要重新计算 | 可复用 |
| --- | --- | --- |
| WP 扫描点、已保存 cut 的阈值、最终 BDT override | WP 与相关结果 | 已训练模型、已有 pair 分数 |
| normalization、mass background 模型 | 结果、系统汇总、报告；background 模型也影响 WP fit | preparation、训练、pair 分数；对应 histogram |
| 已保存字段的最终分析 cut | WP（候选/event cut 时）、nominal、systematics、MC QA | preparation、训练、pair 分数 |
| 新增所需 cut 字段 | 受影响 preparation/compact 输出与结果 | 完成且契约不变的 role shard、模型分数向量 |
| 某 pT 模型 features 或 training candidate cuts | 该模型、WP、受影响推理与结果 | 其他模型、其已验证分数 |
| physics、输入、准备阶段保留范围 | 对应 preparation 与消费它的阶段 | 输入/契约未变的 role checkpoint |

训练只绑定独立候选 signal/background，Femto 输入变化不会单独使候选模型失效。`apply` 依赖 WP 的有效存储 floor；nominal 改变但 floor 未变时可复用 scored。分析阶段再应用最终阈值。

兼容 `persist_prepared: true` 时，`.checkpoints/prepared/<role>/<signature>/` 保存已完成 ROOT entry range，重启时先检查记录，已完成的范围跳过 basket 解码。`.checkpoints/scored/` 保存完成的推理 shard。checkpoint Parquet 和提交后的产物采用 hard link，避免整表复制；旧 generation 在新阶段成功提交后清理。失败不会提交半成品，checkpoint 保留以供重试。

默认不生成逐模型分数缓存。开启 `cache_model_scores: true` 时，`.checkpoints/model_scores/` 用压缩 NPZ 按源 shard、位置范围和模型身份保存稀疏位置/分数向量。局部模型变化时复用其他模型分数，但仍需读取并重写合并后的 compact scored shard，并重新计算下游结果。这里没有候选 ID 或跨事件 dedup。

默认验证仍逐字节检查产物 SHA256。生产配置 `fingerprint: sha256` 也会读取输入以验证内容，因此“复用”仍有完整性检查 I/O 成本，不等于完全不读磁盘。`fingerprint: stat` 可减少输入指纹成本，但只检查输入路径/大小/mtime；它不取消产物 checksum 验证。

`--force` 重算依赖阶段、训练所有模型和推理；通常无需使用。删除 checkpoint 会失去续跑能力。输出写锁阻止两个进程写同一输出目录。

## 三种训练范围

未配置范围时仍是一个 global 模型。限制一个共享模型的训练和应用范围：

```yaml
training:
  pt_range: [2, 8]
  pt_bins: []
  candidate_cuts:
    cos_pa: {min: 0.99}
```

独立分 pT 训练可覆盖全部或部分 pT 范围：

```yaml
training:
  pt_range: null
  pt_bins:
    - id: pt_2_4
      pt_range: [2, 4]
    - id: pt_4_8
      pt_range: [4, 8]
      candidate_cuts:
        cos_pa: {min: 0.995}
  retrain_bins: []
```

每个 bin 可指定 `features` 和 `candidate_cuts`；未指定时继承 training 默认值，显式指定的 cut 字典替换该模型的 training cuts。模型参数、Optuna 和效率网格均可按 bin 覆盖。训练 cut 与最终分析 cut 分开：前者定义训练样本，后者定义 WP fit/最终 pair 选择。ct/fCt 仍禁止作为训练 feature，ct 仅在 cuts 或显式输出需要时保存。

每个模型有独立 `model_handler.pkl`、held-out validation、ROC/AUC、Hyper significance scan 和 WP。`training/registry.json` 明确 pT 路由。区间均为 `[low, high)`、禁止重叠；未覆盖行在推理时标为 `bdt_in_scope=false` 和 NaN score，不写入 compact scored，不套用其他区间模型。`working_point/summary.json` 的 threshold 在分区模式中是 `{model_id: score}`，相同映射应用于 Femto 与 MCFemto。

只重训一个区间可设置 `retrain_bins: [pt_4_8]` 后运行 `train`/`run`，其余模型先验证后复用。完成后清空该列表，避免下一次运行继续要求重训。更改某模型的 training cuts/features 本身也会触发该模型重建。固定 seed 和相同输入可能产生相同模型；这时复用相同模型对应的分数是有效的。

## WP 周围的变化

```yaml
systematics:
  variations:
    - name: bdt_tighter
      group: bdt
      wp_efficiency_delta: -0.02
      overrides: {}
    # 可添加正 delta，前提是 nominal MC efficiency + delta <= 1。
```

delta 是 **held-out MC signal efficiency 的绝对改变量**，例如 0.80 → 0.78，而非 score 的改变量或相对百分比。每个模型分别从其 nominal WP 的实际 MC efficiency 推导目标，再由固定 held-out 分母求阈值；score ties 的实际效率可能不同于目标。目标必须在 `(0,1]`，超界明确报错，不静默截断。不能同时为同一 variation 配置绝对 `analysis.bdt_threshold`。

WP 阶段保存 `efficiency_variations.csv` 与 `variations/delta_±value.json`，包含目标/实际效率、threshold、候选数、质量谱、fit 状态及 S/B。ROOT 报告为每个模型保存独立 ROC、score histogram、WP scan、nominal mass fit 和 variation mass histogram/fit；物理 PDF 用 ROOT canvas，ML PDF 用 HIPE4ML 官方封装导出。

变化阈值由 MC 效率定义；其诊断 Hyper fit 失败不阻止该 cut variation 应用。每个 CF 的纯度修正仍由对应 channel 的 purity fit 状态决定。训练样本定义的系统变化需要修改 training 配置后用独立 output 跑出新 model → WP → apply → CF；本次不把多个重叠训练模型族自动塞进同一个 pT registry。

## 实测结果与边界

在当前机器、同一 Femto 文件的全部 19 个 TTree、15,645,511 原始 pair rows 上比较。准备选择后为 7,328,643 行；四道、两个中心度、SE/ME 所有 k* histogram 计数完全一致。每项两个独立进程运行，表中为耗时中位数，OS disk cache 未控制，ROOT 导入/JIT 计入耗时。

| ROOT 读取 + 运动学 + histogram | 秒 | 进程峰值 RSS（MiB，取两次最大值） |
| --- | ---: | ---: |
| Uproot 全 branch | 18.52 | 1198 |
| Uproot 必需列投影 | 11.50 | 1040 |
| RDataFrame 单线程 C++ | 21.38 | 756 |
| RDataFrame 四线程 C++ | 9.47 | 767 |

列投影在该计算段约快 1.61 倍；RDF 四线程较投影 Uproot 约快 1.21 倍。小子集上的 RDF 启动成本更明显。测试不包含 Parquet 写出、sklearn BDT、WP、fit 或完整报告；不能把这些倍数套到整个 workflow 或 50 GB 输入。生产后端保持 Uproot/Parquet，RDF 提供独立 benchmark，避免将全表 `AsNumpy` 导出后又回到 pandas 的模式。

现有 scored 7,328,643 行的 12 个 variation：独立扫描 5.42 秒，共享扫描 2.05 秒，约快 2.64 倍；两者每个 variation 的 histogram checksum 与 cutflow 完全一致。这两项都使用新的投影/聚合算法，因此是扫描方式的比较，未混入原先完整 fit/report 的时间。该测试为热缓存单次测量，不作正式生产吞吐保证。

前五个 scored shard、471,754 行的存储比较：63 列缩至 26 列（包含新模型路由字段），ZSTD 文件体积为原来的 69.4%，约减少 30.6%。训练/选择字段较多时精简幅度会不同；prepared 保留训练变量池，因此不会同步按这一比例缩小。

下述测量属于此前持久化 prepared 流程的历史性能证据；并非本轮新布局的吞吐或体积测量。本轮默认取消 prepared 与逐模型分数缓存，并进一步裁剪 scored；50 GB 输入仍需为 compact scored 和事务重建时的新旧 scored 产物留空间。新方案解决了 pair 数增长导致的全表内存和重复扫描问题，但尚未实际运行 50 GB full period，不声称已验证生产总耗时、最终 BDT/WP 或物理结果。该段只描述当时的历史测量；当前产物以对应运行 manifest 和最新检查记录为准。

性能产物在 `outputs/benchmarks/optimization/`：`root_backends_full/summary.json`、`ensemble.json`、`slim_storage.json`。复测入口：

```bash
python -m pih3l.benchmark \
  -c configs/testset.yaml --max-trees 0 --repeats 2 --threads 4 \
  --output outputs/benchmarks/my_root_comparison

python -m pih3l.benchmark \
  -c configs/testset.yaml --scored outputs/testset/scored/femto \
  --output outputs/benchmarks/my_ensemble.json
```

RDF benchmark 目前支持 schema 相同的 TTree；生产读取还支持当前适配契约的 RNTuple。backend benchmark 不训练模型；ensemble benchmark 只读取已有分数，不重建 TestSet 物理产物。

历史 TestSet/benchmark Parquet 已按用户要求清理；既有 JSON/CSV/ROOT/PDF 结果作为历史记录保留，不代表新配置已执行。删除清单见 [清理记录](../reports/validation/scored_storage_cleanup.json)。未来完整运行会从原始 ROOT 重建。
