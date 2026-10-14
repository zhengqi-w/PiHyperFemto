# 多 period 输入、训练与合并

四类 AO2D 输入通过独立的 `inputs.<role>.root` 设置，`patterns` 相对此根匹配。根可以是一个目录或多个目录的列表，单个文件也可以通过绝对 `patterns` 指定。所有相对根和输出目录以 YAML 文件所在目录为基准，支持 `~` 和环境变量。同一角色中的软链接/硬链接按物理文件去重。

| 配置角色 | 数据 | 用途 |
| --- | --- | --- |
| `femto` | Pi-Hyper Femto AO2D | 对 pair 中的 H3L 使用所属 period 的模型/WP，计算四道 SE/ME 与关联函数 |
| `background` | Hyper AO2D | 训练背景来自 2.95–3.02 GeV/c² 外；完整候选树用于 WP 质量扫描和 raw Hyper mass QA |
| `signal` | MCHyper AO2D | reconstructed true two-body signal、MC 支持范围、训练/效率标定和 DSCB tails |
| `mc_femto` | PiHyperFemtoMC AO2D | primary/species fractions、truth QA、k* resolution 与 closure；当前可以显式缺失 |

```yaml
inputs:
  femto: {root: /data/pi_hyper_femto, patterns: ["AO2D*.root"], required: true}
  background: {root: /data/hyper, patterns: ["AO2D*.root"], required: true}
  signal: {root: /data/hyper_mc, patterns: ["AO2D*.root"], required: true}
  mc_femto: {root: null, patterns: ["AO2D*.root"], required: false}
periods:
  - id: lhc25_period_a
    inputs:
      femto: {root: /data/pi_hyper_femto/LHC25PeriodA}
      background: {root: /data/hyper/LHC25PeriodA}
      signal: {root: /data/hyper_mc/LHC25PeriodA}
  - id: lhc25_period_b
    inputs:
      femto: {root: [/data/pi_hyper_femto/LHC25PeriodB/pass1, /data/pi_hyper_femto/LHC25PeriodB/pass2]}
      background: {root: /data/hyper/LHC25PeriodB}
      signal: {root: /data/hyper_mc/LHC25PeriodB}
    training:
      model_params: {n_estimators: 1000}
```

`periods` 为空或省略时，保留原来的单 period 工作流。每个 period 可覆盖 `inputs`、`training`、`working_point` 和 `seed`；未覆盖项继承公共配置。`physics`、`analysis`、`purity` 和 `systematics` 共同使用，保证中心度、质量、k* 网格与物理定义可合并。period/model id 使用小写字母、数字、`_`、`-`，period id 必须唯一。

没有显式 `root` 的旧输入仍按 `data_root + patterns` 解析。显式 `root: null` 关闭可选输入；必要输入为 null 会报错。`--data-root` 仅覆盖旧公共根，不覆盖四类显式独立根。根已指到 Femto/Hyper 子目录时，patterns 应为 `AO2D*.root`，不再重复目录名。

## 执行与目录

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -m pih3l run -c configs/testset.periods.yaml
```

`inspect`、`train`、`working-point`、`apply`、`correlate` 等现有命令自动识别 periods。`train` 逐 period 训练，`apply` 逐 period 推理与压缩，`correlate` 开始汇总计数；完整 `run` 保存子报告和合并报告。

```text
outputs/testset_periods/
  periods/test_period_1/
    training/model_handler.pkl
    training/qa/
    working_point/qa/
    scored/femto/*.parquet
    scored/mc_femto/
    nominal/
    systematics/qa/
    report/
  periods/test_period_2/...
  periods/test_period_3/...
  training/metrics.json           # period 模型索引，无伪全局模型
  working_point/summary.json      # thresholds_by_period，无平均 threshold
  scored/dataset.json             # 引用子数据集的 catalog，不是拼接表
  nominal/results.json           # 合并后的 standard/nominal CF
  systematics/qa/                 # 合并后重新评估的 variation/source/total QA
  mc_qa/summary.json
  report/index.html
  report/analysis.root
  report/correlations_cent0.pdf
  report/correlations_cent1.pdf
```

分 pT 模型继续位于各 period 的 `training/models/<id>/`、`working_point/models/<id>/`，scored 按既有配置保存到 `<period>/scored/femto/<model_id>/`。后续代码必须读取所属 period 的模型、效率数组、WP 和 scored 文件。父 scored catalog 不能当作单一模型的数据表或“缺失 MC”输入；直接 library 调用会给出明确错误。

## 合并计算

每个 period 独立执行 HIPE4ML ModelHandler 训练、dense score-efficiency mapping、Hyper WP 扫描和评分。原始 margin 不跨 period 比较，WP 效率变化按所属 period 的映射解析。

汇总阶段只累加可加量：`SE`、`ME`、`sumw²`、质量 histogram/二维质量 histogram 和切选计数，然后重新计算 normalization、完整统计协方差、纯度及 corrected CF。不会平均各 period 的 CF、alpha、拟合参数、AUC 或 WP。即使某 period 单独无法归一化，合并计数足够时仍能得到合法 CF。

同名 systematic variation 先合并所有 period 的该变化计数，再计算变化 CF，并与合并 nominal 比较，最后按 source 合成。任一 period 的变化失败，合并变化就记录 `failed_periods`；不使用其旧文件或其它 period 的局部样本补齐。系统 QA 的 std 仍为合并 nominal CF。

合并报告的系统误差总览按中心度和 raw/corrected 分图，四道 panel 同时展示 Total 黑色粗阶梯线和各 source 彩色阶梯线，纵轴为相对合并 nominal 的百分比。原 JSON/ROOT 数值仍为 fraction；绘图只将其乘 100，不重新组合来源或平均各 period 的相对误差。缺失 variation 的覆盖标记同样保留。

MC 合并原始分子、分母、resolution 计数和 residual moments，再计算 fraction、Wilson interval、bias/RMS 与 closure。部分 period 缺少 MC 时，数值、HTML 和 MC PDF 都标注 available/missing periods；全部缺少时保持 `skipped_missing_mc_femto`，不使用 MCHyper 代替 pair MC。

合并内存由 histogram 大小和 variation 数决定，不依赖全量 pair 行数。ROOT 数据按既有 chunk 投影读取，每个 period 只保存 compact scored，不额外复制原始 AO2D 或生成合并的大型 Parquet。

## 缓存与报告

每个子 period 保持独立的九阶段 manifest 和缓存。改变一个 period 的 seed、模型/WP 或输入，仅使该 period 受影响的阶段与父汇总失效；其它 period 的产物复用。父 manifest 绑定所有子 period 的实际 checksum，递归核查依赖和篡改，原子发布且对子目录加锁。

ROOT 保留 `nominal/`、`final/`、`variations/` 和 `systematics_qa/`，训练/WP 数值使用 `periods/<period>/training[/models/<id>]/`、`periods/<period>/working_point[/models/<id>]/`。canvas/PDF 后缀采用长度编码，避免带 `__` 的合法 period/model id 相互覆盖；图内显示可读的原名称。HTML 链接完整子 QA 并列出各模型的 AUC/WP，不伪造平均模型。

## 三次重复 TestSet 的检查

`configs/testset.yaml` 保持单 period；`configs/testset.periods.yaml` 用三个 id 引用同一套 TestSet 文件，并设置 `synthetic_repetition: true`。重复 Femto 文件在普通多 period 配置中会被拒绝，该开关只用于软件测试；MC 训练样本可以明确共享，输入重叠同时记录在 metadata 中。

本次三套独立模型的 AUC 均为 0.9935944534253403，WP margin 均为 0.4214570286160236。每个 period 的 nominal signal-window SE/ME 是 82,006/793,813，合并为 246,018/2,381,439；八组 raw CF 不变。公式中 covariance 缩小为 1/3 仅验证实现，不表示真实增加了三份独立统计量。八组 corrected 仍因旁带缺失不可用，MCFemto 仍缺失。

验收命令与证据：

```bash
python -B scripts/validate_periods.py \
  --config configs/testset.periods.yaml \
  --report reports/validation/testset_periods_validation.json
```

当前详细记录见 [三 period 验收](../reports/validation/testset_periods_validation.json)、[软件回归](../reports/validation/periods_full_software_checks.log) 和 [多 period native 集成](../reports/validation/periods_pipeline_checks.log)。三 period 验收逐项检查 nominal 与全部 17 个变化的子 period 计数合并、sumw2、CF/covariance、ROOT 对象、依赖与 checksum；[接口保护前后对照](../reports/validation/testset_periods_guard_comparison.json) 另保留模型/WP/评分和选中计数的精确对照。它没有保存全部旧版数值数组，因此不声称全量文件逐字节一致。这些记录验证软件与当前 TestSet，不代表 50 GB full period 吞吐或正式物理测量；百分比图形改版的验收以对应最新运行记录为准。
