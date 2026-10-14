# 使用指南

本文面向已经准备好 AO2D TestSet 的日常分析操作。入口、默认配置和完整数据接口分别见 [README](../README.md)、[testset.yaml](../configs/testset.yaml) 与 [data_schema.md](data_schema.md)。

50 GB 规模的列投影、精简 scored、阶段缓存、断点续跑，以及分 pT 模型、局部重训、WP 效率变化和性能证据详见 [大数据执行](performance.md)。`apply` 先完成 WP，再按预留系统变化范围保存 compact scored；后续分析从 WP 阶段读取标量或逐模型阈值。新流程和输出列配置详见 [紧凑存储](scored_storage.md)。

## 1. 环境与第一次运行

安装完成并激活项目环境后，从项目根目录运行：

```bash
pih3l inspect -c configs/testset.yaml
pih3l run -c configs/testset.yaml
```

训练、推理、模型保存和 ML QA 使用 HIPE4ML 的封装，底层固定 XGBoost。同一环境已经提供 PyROOT 6.38.04，物理分析图用 ROOT、ML QA 用 HIPE4ML plot_utils。ROOT 作为环境依赖，不由本项目的 pip dependencies 代为安装；缺少 PyROOT 时明确失败，不改用 Matplotlib。

可使用 [direnv 模板与步骤](direnv.md) 自动激活 Conda 环境并设置项目路径；服务器批任务用 `direnv exec . ...` 显式加载。该模板不安装依赖，不改变分析配置。

`inspect` 检查文件和支持的 ROOT 树，记录每个 DF 的 entries 与 branch 清单。它不等同于读取全部数据的内容检查；运动学、必要特征和切选字段在后续阶段验证。

完整运行的可读入口是 `outputs/testset/report/index.html`，主要 ROOT 产物是 `outputs/testset/report/analysis.root`。物理图从 ROOT canvas 导出 PDF；ML 图从官方 HIPE4ML 封装导出 PDF。HTML 链接这些文件，ROOT 文件保留数值对象和物理 canvas。中间日志会显示当前阶段以及读入、保留、打分的行数。`prepare` 和 `apply` 处理的是大量 pair rows，工作量应按 pair 数判断，而不是按超氚候选统计判断。

`analysis.root` 仅由 `report` 或完整 `run` 阶段生成。`correlate` 等上游阶段输出 CSV/JSON/NPZ，`systematics` 另生成独立 ROOT/PDF QA；之后执行 `python -m pih3l report -c configs/testset.yaml`，会检查依赖并生成汇总 ROOT 文件和所有主要图。

另一个样本应修改四类 `inputs.<role>.root`，并使用明确的输出目录：

```bash
python -m pih3l run \
  -c configs/testset.yaml \
  --output /absolute/path/to/analysis-output
```

数据目录中的输入文件保持只读。所有生成文件放在输出目录。不要把输出目录设成输入数据目录。

## 2. 配置由什么组成

YAML 覆盖 `src/pih3l/config.py` 的默认值。配置解析后写入各阶段 manifest，因此读结果时应检查 manifest 中的实际配置，不只看配置文件名。

| 配置区 | 主要选项 |
| --- | --- |
| `inputs` | 四类独立 `root`、相对 glob 与 required 标志 |
| `periods` | 各 period id、输入根与 training/WP/seed 覆盖 |
| `synthetic_repetition` | 重复 Femto 输入的显式软件测试声明；生产默认 false |
| `io` | `chunk_rows`、输入指纹、ROOT 列投影、`persist_prepared`、scored 输出列和 pT 子目录、checkpoint |
| `physics` | He3、pion、H3L 质量，He3 动量尺度，pair 质量约定 |
| `analysis` | 中心度、k* binning、归一化区、信号窗/sideband、切选、CPR、BDT 阈值 |
| `training` | 独立训练背景 sideband、特征、样本量、树参数与验证比例 |
| `working_point` | 独立 Hyper cuts、质量窗/sidebands、fit、效率扫描和 storage 预留范围 |
| `purity` | `fit` / `fixed` / `none`，背景模型与质量拟合边界 |
| `uncertainty` | 当前支持 `pair_poisson` |
| `systematics` | 命名 variations、来源 group 与 envelope / RMS 方法 |
| `mc` | MC QA 与受控的修正入口；不会自动应用 primary dilution |

输入默认匹配各目录的 `AO2D*.root`。若文件命名不同，只修改相关角色的 patterns，例如：

```yaml
inputs:
  femto:
    root: [/data/Femto/periodA, /data/Femto/periodB]
    patterns: ["*.root"]
    required: true
  mc_femto:
    root: null
    patterns: ["*.root"]
    required: false
```

相对 patterns 以所属角色的 `root` 为基准；根省略时，旧配置才回退到 `data_root`。相对根和 `output` 以 YAML 文件目录为基准。未知字段、非法数值、NaN 阈值、重叠窗口、错误电荷和缺失必要字段均不应静默忽略。

[production.example.yaml](../configs/production.example.yaml) 是独立生产的起点：使用更细中心度、内容 SHA-256 校验，MCFemto 当前为 null/optional，可在 pair MC 到位后设置独立根。它的路径是占位路径，切选与系统变化仍需分析者根据生产数据审定；文件名不代表物理选择已经通过验证。

多 period 输入、模型/WP 隔离、合并计数与三次 TestSet 示例详见 [period 工作流](periods.md)。`python -m pih3l run -c configs/testset.periods.yaml` 保存三个子 period 以及合并后的四道 CF；不平均 CF 或 score threshold。

## 3. 中心度、k* 和切选

默认中心度采用 `[0,10)`、`[10,50)`，因此中心度 10 只进入第二组。四个 channel 单独输出。默认 k* 范围为 `[0,0.6)` GeV/c、步长 0.02 GeV/c，归一化范围为 `[0.3,0.5)` GeV/c。归一化边界必须与 histogram edges 重合。

普通切选可用区间或命名字段表示：

```yaml
analysis:
  candidate_cuts:
    ct: {min: 0, max: 40}
    cos_pa: {min: 0.99}
    NSigmaHe: {abs_max: 5}
  pair_cuts:
    PairPiTrackPt: [0.15, 5]
    PairPiTrackEta: {abs_max: 0.8}
  event_cuts:
    ZPrimVtx: {abs_max: 10}
```

`min` 包含下界，`max` 不包含上界；`abs_max` 表示严格小于绝对值上限；`equals` 可用于布尔或离散数值。切选用内部 canonical 列名，不用 ROOT 的 `fHyp...` 前缀。配置引用缺失列时会失败，不会跳过该项。

`close_pair_reject_bits: 3` 拒绝 bit 1 和 2，即明确失败的 angular/distance 检查。bit 4、8 表示检查不可用；若生产定义要求不可用也剔除，可显式设为 15 并作为独立变化验证。默认混合权重是 `unit`。`inverse_depth` 是显式选项，要求所有被使用的 ME rows 具有有限、正的 mixing depth；需要独立物理验证。

## 4. BDT 的训练与应用

训练信号从 `MCHyper` 选择 reconstructed + signal + two-body entry；MC 严格取 `training.signal_mass_window` 内，Hyper background 严格取区间外，默认边界 [2.95,3.02]；两个端点均排除。此定义与 Femto 的 `analysis.sidebands` 独立，WP 则继续读完整 Hyper 质量区。模型及 QA 只用五个实际输入，score 统一为原始 margin，旧概率阈值不可沿用。

默认使用 He3/Pi DCA、He3 PID、cosPA、衰变半径、TPC crossed rows 等候选变量。`ct/fCt` 被明确禁止作为模型特征，避免对 lifetime 分布造成 BDT sculpting；质量、关联 pion 变量、k*、SE/ME、truth 标签也不允许作为模型特征。候选 tree 和 pair tree 必须都能提供选定特征；比如只在 Hyper 存在的 TOF 量不能自动应用到缺该字段的 compact Femto。

模型统一由 HIPE4ML `ModelHandler` 包装 XGBoost，`training.model_params` 配置其参数；`class_weight: balanced` 在拟合样本分区内计算权重，也可设为 `null`。样本通过可复现的随机优先级抽样限制每类上限。来源文件、树和 primary vertex 作为 surrogate group 划分训练与验证；没有把这一分组宣称为真实碰撞 ID。

`training.use_optuna: true` 开启仅在 training partition 内的分组交叉验证，搜索平均 AUC 最优参数后重新拟合；最终 held-out 保留给训练 QA 和 WP 效率映射。Optuna 默认关闭，可配置 trial/time budget、搜索域及逐 pT 模型覆盖。参数列表、优先级和完整示例见 [训练与 Optuna](training.md)。

默认 `working_point.signal_efficiencies: null` 消费训练生成的密集 score–efficiency 数组，也可指定显式效率子集。每个阈值随后应用到独立 `Hyper` 重建候选，在 2.96–3.04 GeV/c² 的 50-bin histogram 上用 RooFit DSCB + pol1/pol2 拟合。DSCB tails 从共同预选域 MCHyper 的 80-bin histogram 校准并固定，背景默认由 sidebands 预拟合后固定。按每点 fitted mean ± 3×core sigma 积分出 S3/B3，在效率区间内将 S3/eff 用 ROOT pol0 拟合为 `c0`。按 spectrum 指标 `ε*c0/sqrt(c0+B3)` 最大值对应的 score 写入 `working_point/summary.json`，再作为 Femto 和 MCFemto 的共同名义阈值。Femto pair tree 不参与 WP 优化。

`training.efficiency_min: 0.29`、`efficiency_max: 0.99`、`npoints_for_effi: 701` 定义密集效率数组；每个 pT 模型可覆盖。旧 `qa_signal_efficiencies`、`qa_centrality_bins` 删除，训练不分中心度。`working_point/scan.csv` 保存请求/实际效率、raw yield 及误差、raw/eff、背景、原始与 expected significance、mean/sigma、拟合质量和排除原因；`scan_fits.json` 保存所有点的质量计数、期望及完整拟合结果，`best_fit.json` 保存最佳点。pol0 默认范围 [0.5,0.95]，至少三个有效且不同 score 的 cut，质量拟合 χ²/ndf 上限 10；pol0 无效则明确失败，诊断保留在 `failures/working_point-<signature>/`。详见 [工作点说明](working_point.md)。WP 独立配置 `candidate_cuts`、`event_cuts`、`mass_range`、`sidebands` 和 `mass_fit`；nominal WP 使用动态 3σ，`signal_window` 不用于 WP 目标。Femto 最终选择及 pair purity 配置独立保留。

```bash
python -m pih3l train -c configs/testset.yaml
python -m pih3l working-point -c configs/testset.yaml
python -m pih3l apply -c configs/testset.yaml
```

重点查看：

- `training/metrics.json`：特征、样本量、阈值、效率分母说明、`score_efficiency_array`、AUC、KS、score–mass 相关与 XGBoost normalized gain；重要性 PDF 为官方 SHAP。
- `training/model_handler.pkl`：HIPE4ML 官方保存的模型和特征；metrics/manifest 保存范围、id 与来源。
- `training/validation_scores.npz`：仅 held-out 标签与 score；`score_efficiency_array.txt` 为 N×2 的 score/efficiency 数组。
- `training/optimization/`：启用 Optuna 时的最优参数、trial 表与内部 CV 隔离记录；分 pT 模型位于各自 `training/models/<id>/`。
- `training/qa/`：特征/质量、相关矩阵、train/test 分数和 ROC、efficiency/score、SHAP 重要性的九张官方 PDF 与数值 `training.root`，Optuna 启用时另有优化图。
- `working_point/summary.json`、`scan.csv`、`scan_fits.json`、`best_fit.json`：Hyper 数据 WP 选择及所有质量拟合结果。
- `working_point/qa/`：significance 对 score/efficiency、raw yield、raw/eff pol0、mean/sigma、fit stability、最佳质量拟合七张 PDF 与 `working_point.root`；完整 scan 和 MC tail 校准都保存在 ROOT。分 pT 时 QA 进入各自 `models/<id>/qa/`。report 同步导出四张 `working_point*.pdf`，并把相同 canvas 保存在 `analysis.root`。
- `scored/summary.json`：读入、推理、裁剪前后及无效分数计数；每个 role 的 `dataset.json` 记录保存列、最低阈值与数据范围。

AUC 只衡量分类能力，不证明数据/MC 一致性、质量形状无偏或 femto acceptance 无偏。

## 5. 关联函数与纯度

每组结果都保存信号窗、左右 sideband 和合并 sideband 的 SE/ME。关联函数为

```text
alpha = ME_norm / SE_norm
C_raw(k*) = alpha × SE(k*) / ME(k*)
```

协方差通过完整 Jacobian 传播，包含同一 histogram 归一化积分与各 bin 的相关。加权计数使用 sumw2。它仍然采用独立 pair 的抽样近似，不能包含 compact 输入未记录的共享事件全部相关。

为避免零计数被解释为零误差测量，没有 SE 或没有 ME 的 k* bin 会标记无效。没有归一化计数时整组归一化不可用。这些状态不会被填成 unity。

纯度拟合使用中心度、电荷道内选中 SE pairs 的质量 histogram。一个候选贡献多个 pair 时也贡献多次质量记录，因此结果是 **pair-weighted purity**。拟合模型为 Gaussian signal 加 constant/linear background，使用 binned Poisson likelihood；拟合边界、质量范围、曲率和拟合质量均需要检查。

当前配置项 `purity.min_mass_entries` 控制的是该 SE pair 质量 histogram 的最小 entries，而不是唯一候选计数；pair 数不能用来证明独立候选统计足够。

```text
C_corrected = [C_raw − (1 − P) × C_sideband] / P
```

只有接受的纯度估计和可用的 raw/sideband 结果才能给出修正。修正的误差包含 shared-P 项，但质量拟合与 pair histogram 之间的交叉协方差未由当前数据恢复。

TestSet 的 Femto 没有宽 sideband，当前 `purity.method: fit` 应报告 `insufficient_mass_sidebands`，对应 corrected CF 不可用。不能通过缩窄 sideband 直到“算出数字”就宣布得到背景修正：新的 sideband 需验证峰尾污染和背景代表性。

若有经过验证的外部 pair purity，可显式配置 `method: fixed`、`fixed`、`fixed_error`。这是用户提供的分析输入，不是框架从 TestSet 测得的纯度。

## 6. 系统误差

每个 variation 用同一条分析链重新进行切选、SE/ME、归一化、质量拟合和修正。下面是一个配置示例：

```yaml
systematics:
  method: envelope
  variations:
    - name: pion_pid_tight
      group: pion_pid
      overrides:
        analysis:
          pair_cuts:
            PairPiTrackTpcNSigmaPi: {abs_max: 2.5}
```

同一 `group` 的变化是同一来源的替代选择，先形成该来源的 envelope/RMS；不同 group 视为独立来源组合。envelope 使用逐 bin 最大带符号偏移的外积作为该来源协方差模型。RMS 仅使用在全部有效 nominal bins 中均完整的共同 variation ensemble，以保持半正定；缺少这样的 ensemble 会报告覆盖不足。每个变化有独立结果目录，总表在 `systematics/summary.json`。raw 与 corrected 分别汇总，因此即使当前 corrected 不可用，raw 的可用变化仍可检查。

测试配置包含 17 个变化，覆盖 BDT 阈值、归一化窗口、关联 pion pT 下限/上限、TPC crossed rows、PID/DCA、occupancy、质量窗、sideband、质量背景模型与 close-pair 标志；已移除 `cospa_tight`，nominal 的 cosPA selection 保留。新增 pT 下限为 0.20/0.25 GeV/c、上限为 2.0/2.25 GeV/c、TPC crossed rows 下限为 100/110。三组对应 `pion_pt_min`、`pion_pt_max`、`pion_tpc_rows`，每组两个替代变化；不同组独立合并的假设由 group 定义体现。TPC 使用配对 pion 的 `PairPiTrackTpcNClsCrossedRows`，不是超氚衰变 pion 的列。TestSet 的 loose/tight 使用效率 delta ±0.02；显式 BDT thresholds 属于原始 margin 输入，不会随模型自动重标定；需要围绕新 nominal 自动移动时，使用 `wp_efficiency_delta`，由当前模型的固定 held-out 分母和 HIPE4ML 重算 threshold。正式研究应按效率与拟合稳定性审定变化范围。

这个离线 variation engine 不会重训模型，也不能改变 producer 的 mixing pool。训练样本/特征/随机种子变化应做独立完整运行；mixing z/centrality pool 宽度变化需要重新生成 pair 输入。variation 的中心度和 k* binning 必须保持一致，不能扩展准备阶段已经丢弃的质量范围。

当前组合保留 variation 的统计涨落，没有实施 Barlow 筛选或统计去偏。缺失变化的覆盖会记录，不能将其解释为零系统误差。

`systematics` 命令会直接生成独立 ROOT QA，保存至 `systematics/qa/`，无需再读取 pair 行。每个 variation、每个 source 以及总系统误差均按中心度输出 PDF，四个电荷道在同一张图中。variation 上半部分对比 nominal/variation 的 raw CF 和各自统计误差、信号窗 SE/ME 计数，下半部分显示 raw/corrected 的 `C_variation/C_nominal`。这里的 **std 指 standard/nominal C(k*)**，不是统计标准差。比值不画统计误差条，因为共享 pair 的 nominal–variation 交叉协方差未知。

相对系统误差图采用 spectrum `c_sys_details_cen_30_50.pdf` 的百分比阶梯图样式，纵轴为 `100 × sigma_systematic/abs(C_nominal)`，单位 `%`。总览在同一 panel 叠加黑色粗阶梯线的 Total 和各来源的彩色阶梯线；每个中心度的 raw、corrected 各出一张四道图。独立 source 图也使用百分比阶梯线。阶梯边界对应实际 k* bin，无效 bin 留空，不跨缺失区间连线；完全不可用的 corrected 图仍保留明确状态。

百分比仅是图形显示单位。`results.json` 和 ROOT 中原有 `systematics_qa/.../values`、`graph` 保存无量纲 fraction，例如 `0.05` 表示 `5%`；不能对这些数值再次误用百分比定义。variation/nominal 图仍以比值 `1` 为基准，不乘 100。

raw 总览沿用 `systematic__total__cent<i>.pdf`，corrected 总览为 `systematic__total__corrected__cent<i>.pdf`；`<i>` 是配置中的中心度序号。ROOT 的 source/total leaf 下另有 `percent_steps/`，保存连续有效区间的百分比 TH1D 与区间索引，便于重画同一阶梯线。

零分母、非有限或无效 nominal/corrected bin 保持 null 和 validity mask，不补 0/1。失败的 variation 仍出现在 QA 索引、状态和空图中；source 保存预期、完成、失败列表及逐 bin coverage，RMS 另记录实际共同 ensemble 大小。部分可用变化计算出的 spread 不代表完整 envelope。`report` 链接所有系统 QA、展示最终相对系统误差，并将数值对象和 canvas 同步到 `report/analysis.root`，没有 PNG 或额外 pair 扫描。

`systematics/qa/index.json` 是 PDF/canvas 清单，`results.json` 保存 ratio、相对误差及 masks，`systematics.root` 保存可重画的原生 ROOT 对象。新增 cut 所需的输入列会自动保留在 compact scored；旧 scored 缺少 TPC 列时完整 workflow 会重新 apply。

系统阶段还导出最终汇总，便于绘图或传给下一步模型拟合：

| 文件 | 内容 |
| --- | --- |
| `systematics/final_observables.csv` | 每个中心度/电荷道、raw/corrected、k* bin 的 `value`、`stat_error`、`syst_error`、`total_error`、`valid` |
| `systematics/covariances.npz` | `<bin>__<observable>__systematic` 和 `<bin>__<observable>__total` 矩阵 |
| `nominal/covariances.npz` | `<bin>__raw_covariance` / `<bin>__corrected_covariance` 统计矩阵 |

其中 `<bin>` 例如 `cent_0_10__ch_0`，`<observable>` 为 `raw` 或 `corrected`。合计矩阵为 `V_total = V_stat + V_syst`，误差取对应对角元的平方根，使用统计与配置系统来源独立的组合假设。它仅包含本次实际配置且可计算的误差来源；缺失 corrected 值或系统覆盖时保留无效，不生成数值齐全的伪合计误差。

## 7. MCFemto 接口

填入 `MCFemto/AO2D*.root` 后，执行：

```bash
python -m pih3l mc-qa -c configs/testset.yaml
```

依赖阶段会根据新输入重建。fraction QA 使用同一模型、重建级切选和数据 signal mass window，区分 SE 与 ME，输出物种/primary/超氚信号/真 primary pair、两侧碰撞匹配 fraction，并检查有条件可得的 k* resolution。fraction 采用未加权 pair rows 与 Wilson 区间；即使 CF 选择 inverse-depth ME 权重，这些 fraction 仍是明确标注的未加权 QA。

primary pion fraction、true pion fraction、hypertriton truth fraction、true primary pair fraction 的分母和含义不同。要求 `SameMCCollision` 的 true-pair 标签在 ME 中通常为假，不应被解释成 ME 全部是污染。当前不会自动将这些量乘成 λ，也不会自动做 primary dilution correction；需要先确认残余相关和混合背景模型。

当前 TestSet 的 MCFemto 目录为空。`mc_qa/summary.json` 的 missing-input 状态是应有结果；synthetic fixture 或数据 raw CF 不能替代真实 MC 验证。

## 8. 产物、复跑与故障诊断

每个阶段先写临时目录，成功后替换对应正式目录；同一输出目录有写锁。已有完整结果不会因中途失败被替换成一半文件。

缓存检查按阶段消费的配置、输入 role、相关模块、依赖环境和 artifact checksum 进行；不同阶段的 signature 可以不同，manifest 记录消费的上游 lineage。只改报告代码不会使 preparation 或模型训练失效；只改 WP 不重新计算 pair 分数。preparation/scoring 有可验证的 shard checkpoint，局部 pT 重训还可复用其他模型的分数向量。`--force` 会重建目标阶段及所有依赖，并重训/重新推理。具体范围、SHA256 检查的 I/O 成本见 [执行说明](performance.md)。

| 状态或现象 | 含义与处理 |
| --- | --- |
| `missing_optional` | 可选输入不存在；当前 TestSet 的 MCFemto 属于这一类 |
| `skipped_missing_mc_femto` | MC QA 阶段没有 reconstructed MC pair 输入，未生成物理 fraction |
| `empty_after_preparation` | 输入存在，但没有行通过中心度、物理量或预备范围；检查 counts 与配置 |
| `insufficient_normalization` | 所设归一化区没有可用 SE/ME；查看原 histogram 与统计量 |
| `partial_empty_bins` | 部分 k* bin 不足以给出当前统计近似下的有效 CF |
| `insufficient_mass_sidebands` | 质量 sideband 无足够条目；检查上游输出质量范围 |
| `fit_at_boundary` / `singular_fit_covariance` / `poor_fit_quality` | 质量模型无法给出接受的纯度；不能只看 central value |
| `incomplete_coverage` | 系统变化在部分 bin 没有足够有效结果 |
| 必要 branch 不存在 | 检查新版 schema、特征与 cuts 的跨源可用性 |
| 输出目录被锁 | 另一个进程正在写同一输出；使用独立输出或等待现有任务结束 |

读取数值时同时读取 `valid` 和 `status`。JSON 将非有限数值写成标准 `null`，CSV/NumPy 保留 NaN 语义。不要在后处理中把缺失值自动填成 0。

ROOT 文件里的 `nominal/<bin>/g_cf_raw_stat` 等 graph 仅包含有效点；对应 TH1 保存全部 bin，并带 `valid_raw`/`valid_corrected`。SE/ME TH1 的误差来自原始 sumw2，协方差以 TH2 保存。`canvases/` 下的 CF、SE/ME spectra、WP、质量与 mixing canvas 可在新的 ROOT 进程中重画。完整 key 表和 PyROOT 读取例子见 [ROOT 输出说明](root_output.md)。

协方差/mask TH2 的 bin error 显式为零，表示没有再给“矩阵元素的误差”建模；不能让 ROOT 自动给 covariance 取平方根充当它的 error。质量–k* 与 MC resolution 的二维计数 histogram 则使用相应 Poisson variance。未加权计数对象可保存真实计数 Entries；加权 ME 和 derived summaries 的 Entries 设为 0，数值内容与 sumw2 仍完整。

开发检查使用现有环境中的 `unittest`：

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

也可使用 `make test PYTHON=python` 或 `make run PYTHON=python`。Makefile 只是同一入口的简写；本地验证不要求安装 pytest 或 ruff。

此前 probability-score DSCB/报告同步的[历史软件回归](../reports/validation/dscb_wp_software_checks.json)覆盖合成完整 workflow、HIPE4ML/Optuna、分 pT、独立 DSCB/Chebychev 积分、RooFit 物理协方差、3σ pol0/argmax、ROOT/PDF 和缓存。真实 TestSet 已完整重跑：[完整产物核对](../reports/validation/testset_dscb_validation.json)检查九阶段 SHA256/依赖链，确认 scored、关联、系统变化与 report 已消费新 WP，四张 report WP 图精确复用 stage canvas。701 点中 592 个有效，pol0 区间内 448 点，测试 score 约 0.52278。pol0 χ²/ndf 约 10.762、MC DSCB χ²/ndf 约 33.963，仍需物理建模研究；八组 corrected CF 因旁带不足不可用，MCFemto 为空。此前 Gaussian QA 为[历史检查](../reports/validation/testset_qa_refresh.json)。远端 CI、50 GB full period 和正式物理验证不属于本次本地结论。

真实 TestSet 的阶段统计、八组结果、ROOT 对象核对、缓存复跑和当前数据限制集中保留在 [独立验收报告](../reports/validation/acceptance_zh.md)。数值核对与运行签名以该报告和对应 JSON/manifest 为准。

ROOT I/O 的 API 用法参考 [Uproot 官方入门文档](https://uproot.readthedocs.io/en/stable/basic.html)，分类器参数与预测接口参考 [HIPE4ML ModelHandler 官方文档](https://hipe4ml.github.io/hipe4ml/model_handler.html)。实际依赖版本以当前运行 manifest 为准，不能用在线 stable 文档版本替代运行环境记录。

正式物理结果的最低证据包括：数据与 MC 选择一致性、质量/BDT/acceptance 检查、可用 sideband 或外部纯度、MC truth/primary 验证、统计相关性处理与完整系统变化。TestSet 用于确认代码和产物结构，不能单独完成这些物理验证。

最新 raw-margin 与顶点修正的全流程检查见 [完整验收](../reports/validation/testset_margin_vertex_validation.json)、[221 项软件回归](../reports/validation/margin_vertex_software_checks.json)和[保存位移审计](../reports/validation/vertex_convention_audit.json)。旧模型、旧 probability validation NPZ、缺少/不匹配 score_domain 的 scored 数据需重训或重新 apply，不能混用。
