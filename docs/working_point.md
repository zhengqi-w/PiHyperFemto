# 工作点、密集效率数组与独立 QA

workflow 顺序为 **training → Hyper WP → apply → correlations/systematics → report**。训练用 HIPE4ML plot_utils 输出 ML PDF、另外保存数值 ROOT；WP 用 PyROOT 输出 ROOT/PDF。report 链接 ML PDF，仅复用物理 canvas。关闭 report 不影响训练/WP QA 的生成。

## 1. 密集 score/efficiency 映射

`training.efficiency_min: 0.29`、`efficiency_max: 0.99`、`npoints_for_effi: 701` 定义 `unique(round(linspace(min,max,npoints),3))`，默认步长为 0.001。每个 pT bin 可独立覆盖。实际调用 [HIPE4ML score_from_efficiency_array](https://hipe4ml.github.io/hipe4ml/analysis_utils.html)，输入该模型的固定 outer held-out **两类** label/score；两类一起定义插值网格范围。输出 `score_efficiency_array.txt` 为 N×2：第一列 score，第二列请求 efficiency。

与 spectrum `BDTPreProcess.py` 使用同一个库 API，分数统一采用 `ModelHandler.predict(output_margin=True)` 的 raw margin，不做 sigmoid 或归一化。score 可为负或大于 1；训练、效率数组、WP、apply 和系统变化必须使用同一模型与分数定义。旧 probability 阈值不能原数沿用，应重训、重新扫描 WP，并用效率变化重算系统阈值。

HIPE4ML 的效率比较是 `score > threshold`，实际选择/保存采用 `score >= threshold`；有限样本和 score ties 会使 actual efficiency 与请求不同。`metrics.json` 和 WP summary 同时记录二者、固定分母、库版本与实现 SHA256。

某些离散效率平台会让 HIPE4ML spline 找不到 root。此时只将失败目标向零移动一个 float64 ULP，再调用原 API；保留原请求效率、实际评估目标及重试记录。不会改训练 split、替换样本或使用分位数兜底。仍无法插值时明确失败。效率 1 使用固定 held-out 的最小 score 保证该样本的 inclusive 接受率；它不是 margin 的全域下界。storage reserve=1 或关闭裁剪则使用 JSON `null` 边界，保留所有有限 in-scope 应用分数。

## 2. Hyper 扫描与 pol0

`working_point.signal_efficiencies: null` 直接消费训练数组；显式列表可以覆盖扫描网格。独立 Hyper 候选按 WP 的 cuts、pT 和质量范围分块读取，打分后累计各 cut 的质量 histogram，内存不随 full-period 候选数线性增长。相同 score 共享 histogram 和拟合计算，每个请求效率仍完整保留。

启用 MC 自动变量范围时，WP 还应用模型中冻结的 feature ranges，与训练/held-out 和 Femto/MCFemto 推理一致。summary 保存范围定义及其他切选后被范围排除的候选数。score efficiency 是该共同预选域内的条件效率；MC 接受率仍需另行测量，不能当作完整重建效率。范围学习与逐类 cutflow 见 [训练说明](training.md)。

WP 使用独立 `wp_mass_fit.py` 中的 RooFit extended binned likelihood：signal 为 DSCB（`RooCrystalBall`），背景只能选择 `pol1` 或 `pol2`（`RooChebychev`）。默认与 spectrum 使用相同质量范围 2.96–3.04 GeV/c²、data 50 bins、mean 2.985–3.005、sigma 0.0014–0.002。通过 bin 积分计算期望，不以 bin 中心 PDF 值近似计数。

先在共同预选域内的 MCHyper 真信号上拟合 DSCB tails，随后固定这四个 tail 参数用于所有 data score 点。MC 使用 80-bin histogram，流式累积，不保存全量候选；这与 spectrum 部分场景的 unbinned MC tail fit 有区别。MC 拟合失败会明确记录默认参数回退及原因，不能误认作成功校准。`mc_signal_fit.json` 和 ROOT `mc_signal_fit/` 保留 histogram、拟合和参数来源。

默认先在 sidebands 拟合背景，再固定其系数用于全区间拟合。背景函数必须在整个质量域连续为正，不对负值作截断。每点按 fitted mean ± 3×core sigma 积分得到 `S3_i` 与 `B3_i`，窗口裁剪到 mass range；raw signal 误差由实际 RooFit 浮动参数的物理计数协方差传播，包含 mean/sigma 和 moving window 的相关项。当前为固定 MC tails 和 sideband 背景形状下的条件误差；这些固定形状的不确定性需要单独的系统变化，不能由 yield 误差代替。Femto 的 pair-weighted purity fitter 仍是独立模块。

接受质量拟合状态 `ok`、有限 raw significance、正且有限 yield error、有限 χ²/ndf，并应用 `max_fit_chi2_ndf`。拒绝原因随每个点保存，失败点不填零。默认参数参考 spectrum 的当前 `general_config.json`：

```yaml
working_point:
  objective: pol0_expected_significance
  signal_efficiencies: null
  yield_eff_range: [0.5, 0.95]
  min_pol0_points: 3
  max_fit_chi2_ndf: 10.0
  save_score_fit_frames: false
  mass_range: [2.96, 3.04]
  sidebands: [[2.96, 2.98], [3.005, 3.04]]
  mass_fit:
    signal_model: dscb
    background: pol1          # 只能 pol1 / pol2
    mass_bins: 50
    mc_mass_bins: 80
    min_mass_entries: 80
    mean_bounds: [2.985, 3.005]
    sigma_bounds: [0.0014, 0.002]
    calibrate_mc_tails: true
    sideband_prefit: true
    alpha_bounds: [0.7, 5.0]
    n_bounds: [0.9, 15.0]
    tail_defaults:
      alpha_left: 1.5
      n_left: 2.0
      alpha_right: 1.5
      n_right: 2.0
```

对区间内有效点，以请求效率 `ε_i` 为横轴，`S3_i/ε_i` 为纵轴，误差为 `σ(S3_i)/ε_i`，调用 ROOT `TGraphErrors.Fit(TF1("pol0"))` 得到常数 `c0`。至少需要三个不同 score 的有效 cut；拟合必须有效，且 `c0` 和误差为正。重复 efficiency 点不会被 histogram bin 覆盖掉。

随后严格采用本机 spectrum `ProcessWP.C` 当前指标：

```text
S = c0 from pol0(raw signal within fitted 3σ / requested efficiency)
expected_significance_i = ε_i × S / sqrt(S + B3_i)
```

其中 `ε_i*S` 是选中后的预计信号，分母按参考实现使用未乘效率的 `S`。输出明确保存完整公式。质量拟合直接给出的 `raw_significance=S3_i/sqrt(S3_i+B3_i)` 保留为诊断曲线，当前不作为 WP 目标。扫描所有有效点找 expected significance 最大值，而非只在 pol0 区间内找最大值。

nominal score 写入 `summary.json`，同一 threshold 或逐 pT threshold 映射原样应用到 Femto/MCFemto。`analysis.bdt_threshold` 显式 override 时仍记录扫描最佳点及 override 来源。附近 `wp_efficiency_delta` 从 nominal **实际** held-out signal efficiency 出发，通过同一 HIPE4ML 样本求新 cut。储存 floor 覆盖 nominal、全部声明变化和预留最高效率。

pol0 点采用各质量拟合的边际误差加权；不同 cut 的嵌套样本有相关性，当前没有跨 WP 协方差，`c0_error` 不能视为完整物理系统误差。χ²/ndf 使用 histogram 计数误差与实际浮动参数数目，似然 deviance 另存；图形开关不改变拟合质量。默认 sideband 系数固定时 data fit 有四个浮动参数；关闭预拟合时 pol1/pol2 分别有五/六个。

### spectrum 风格的窗口 QA

QA 另存 `spectrum_qa`，对每个质量拟合计算随 fitted mean/sigma 变化的 2σ、3σ、4σ 窗口，边界裁剪到拟合 `mass_range`。三条曲线使用 nominal determination 的**同一个** pol0 常数：`Z_n = ε*c0/sqrt(c0+B_n)`；没有第二次独立归一化，也不将 2σ/4σ signal integral 另外乘进预计 S。图以 3σ 曲线为中心、2σ–4σ 构成填充带；此带代表积分窗口变化，不是统计置信区间。nominal 正是 3σ 曲线的最大值。

`s3overeff_vs_eff.pdf` 显示 yield 误差、pol0 常数/误差/χ²；`mean_sigma_vs_efficiency.pdf` 单独显示带误差的 mean 和 sigma；`best_fit.pdf` 使用单面板展示带计数误差的 data histogram，以及平滑 total fit、DSCB signal 和 polynomial background 曲线，不生成 Data−Fit 或 pull 图。曲线直接计算在完整质量域归一化的函数值，并乘 data bin width 转为 counts/bin；不是连接粗 histogram 的拟合期望。legend 和 text 显示 period、score、效率、expected 3σ significance、S3/B3、χ²/ndf、mean/sigma、背景阶数和拟合状态。最佳点直接复用扫描拟合，避免重拟合改变结果。图上星号始终对应实际 nominal threshold；显式 override 时使用其独立拟合及实际效率，并与扫描最佳点明确区分。

窗口 yield 误差微扰 mean/sigma 时同步更新窗口并执行裁剪；参考 spectrum 的 raw3 误差只传播总 signal yield。该区别及上述 MC binning/固定形状口径随 summary、ROOT 元数据和 `qa/index.json` 保存。pol0 不可用时不填造预计 S 或窗口带；低统计质量拟合以 missing 数值及拒绝原因保存。

## 3. 保存内容与位置

所有路径都相对于配置 output。全局训练放 `training/`，全局 WP 放 `working_point/`；按 pT 时分别放各阶段 `models/<id>/`，含独立 `qa/`。

| 阶段 | PDF | ROOT / 数值 |
| --- | --- | --- |
| training | `feature_distributions.pdf`，`correlations_signal.pdf`，`correlations_background.pdf` | `qa/training.root` 中的特征/质量直方图、Pearson 矩阵及有效标志 |
| training | `bdt_output.pdf`，`bdt_test_score.pdf`，`roc_train_test.pdf` | train/test 两类 score、两条 ROC/AUC；特征/相关 QA 来自经过选择和上限抽样的 train+held-out 候选，不能充当 full-period yield |
| training | `efficiency_vs_score.pdf`，`feature_importance.pdf`，`feature_importance_shap.pdf` | 请求/实际效率 graph；官方 SHAP bar/beeswarm PDF，ROOT 另存 native gain；数组另存 N×2 TXT |
| training（Optuna） | `optuna_training.pdf` | trials、best AUC、状态/参数；`optimization/` 保存 JSON/CSV/split 记录 |
| WP | `sig_vs_score.pdf`，`sig_vs_eff.pdf` | 3σ significance、2σ–4σ 填充带及 nominal 标记 |
| WP | `s3overeff_vs_eff.pdf`，`raw_signal_yield.pdf` | S3/eff 误差图与同一个 pol0 TF1；3σ raw yield |
| WP | `mean_sigma_vs_efficiency.pdf`，`fit_stability.pdf`，`best_fit.pdf` | mean/sigma 误差、χ² 稳定性、单面板最佳质量拟合、可重读 canvas |
| WP | 默认无逐点 PDF | `scan.csv` 保存全部点；`scan_fits.json` 与 ROOT `mass_scan/point_NNNN/` 保存所有质量 histogram、fit expectation、参数、协方差和拒绝原因 |

`save_score_fit_frames: true` 输出逐扫描点质量拟合 PDF，默认 false 避免 701 个文件；所有点始终在 ROOT 保存。未完成最佳 WP 时不生成 `best_fit.pdf`。training 和 WP 均输出 `qa/index.json`，列出内容及可用状态。

QA 都纳入 stage/model artifact checksum。缺文件、改动绘图代码、HIPE4ML editable 实现或 ROOT 环境会使相关缓存失效；只有文件存在不等于完整。report 直接链接官方 ML PDF，将当前 WP ROOT canvas 复制到 `analysis.root` 并导出四张同步 PDF：`working_point.pdf`（score）、`working_point_efficiency.pdf`、`working_point_yield.pdf`、`working_point_mass_fit.pdf`。分 pT 文件名加 `__<id>`。report 不重新计算拟合或 WP，不生成 `bdt_validation*.pdf`、`optuna_training*.pdf`。

## 4. 失败与历史产物

pol0 无有效点或 fit 失败时 workflow 停止，不输出伪造 threshold。已完成的 `scan.csv`、`scan_fits.json`、summary 和 QA 迁移到 `failures/working_point-<stage signature>/`；分 pT 保留其 models 子目录。失败记录包含永久路径，临时目录会清理，之前的成功阶段保持完整。旧成功报告可能仍在磁盘，必须结合 manifest 和失败记录识别，不表示新算法已成功。

`reports/validation/testset_qa_refresh.json` 是先前 Gaussian WP 的 training/WP 检查记录，不能作为当前 DSCB 的证据；其他旧失败记录同样保留其历史含义。当前实现需重新执行完整 `report` workflow，使 scored、关联、系统变化和 report 消费同一 WP lineage。TestSet 低统计运行用于核对软件产物，不作为正式物理 WP validation。

此前 probability score 的 TestSet DSCB workflow 记录位于 `reports/validation/testset_dscb_validation.json` 和 `dscb_wp_software_checks.json`，下列数值仅描述历史模型。701 点中 592 个有效、448 点进入默认 pol0 区间；最大点请求效率 0.961、threshold 0.5227796301211376、expected Z3=44.3771，data 质量拟合 χ²/ndf=1.5022。pol0 χ²/ndf=10.7620、MC tail fit χ²/ndf=33.963，表明常数假设和 MC 质量形状的描述仍需审定。MC tail fit 的 minimizer/covariance 状态成功不等于 goodness of fit 充分；当前不以人为放宽质量条件改善此结果。TestSet 的八组 corrected CF 不可用、MCFemto 空，不据此给出正式物理结果。

算法组织参考 H3l spectrum 项目的 `BDTPreProcess.py`、`ProcessWP.C` 与 `general_config.json`。

修正保存位移、五输入及 raw-margin 后的最新 TestSet 见 `reports/validation/testset_margin_vertex_validation.json` 与 `margin_vertex_software_checks.json`：353 个有效点、313 个 pol0 点，margin threshold 0.4214570286160236、实际效率约 0.946016；pol0 χ²/ndf≈101.533、MC DSCB≈92.127。测试配置使用效率 delta ±0.02，旧概率绝对阈值已移除。
