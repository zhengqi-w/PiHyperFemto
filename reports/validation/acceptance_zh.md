# TestSet 软件 Smoke Test（非物理 Validation）

## 结论

**本文件只记录代码链路 smoke test。当前训练集统计不足，不作为 BDT/WP 或物理结果 validation。**

- 完整流程现在是 `prepare → train → working_point → apply → correlate → systematics → mc_qa → report`。
- **74/74 项 unittest 通过**，耗时 17.104 s。日志见 `unittest_wp.log`。
- 完整 TestSet 执行了全部 **8 个阶段和 12 个系统变体**；独立验收器完成 **371/371 项检查**。
- BDT 只在独立 `MCHyper` truth signal 与 `Hyper` sideband 候选上训练；`ct/fCt` 不在训练特征中。
- 最终 WP 由独立 `Hyper` 重建候选的质量谱 signal+background fit 决定，Femto pair 不参与优化；同一个 score 随后应用到 Femto 和 MCFemto。
- 8 组 raw C(k*) 均有结果。Femto 质量范围缺少正式宽 sideband，因此 corrected C(k*) 明确不可用。
- MCFemto 为空，状态为 `skipped_missing_mc_femto`；没有用训练 MC 伪造 primary fraction、k* response 或 closure。

## 输入处理

| 角色 | ROOT 原始行数 | preparation 保留行数 | 状态 |
|---|---:|---:|---|
| femto | 15,645,511 | 7,328,643 | available |
| background / Hyper | 10,088 | 9,877 | available |
| signal / MCHyper | 13,183,618 | 291,100 | available |
| mc_femto | 0 | 0 | missing_optional |

MCHyper 先要求 `IsReco & IsSignal & IsTwoBodyDecay`，共有 590,085 条重建 signal；中心度范围再保留 291,100 条。compact Femto 使用每行已有的候选特征推理，没有构造或追踪 hypertriton ID。

## BDT 训练与 WP

训练特征为：`DcaV0Daug, DcaHe, DcaPi, NSigmaHe, cos_pa, decay_radius, NTPCCrossedRowsHe, NTPCCrossedRowsPi`。

- validation AUC：**0.990444**；train AUC：**0.995628**。
- 训练参考点为 80% held-out MC signal efficiency，对应 score **0.950935**；它仅是训练 QA，不是最终 WP。
- signal/background 的训练与验证 surrogate vertex groups 交集均为零。
- WP 扫描将配置的 MC signal-efficiency 分位数转换为 score，在 Hyper data 上逐点做质量拟合。
- 16 个 scan points 中有 6 个通过最小统计和 fit-quality 要求。
- 最佳 WP score：**0.430892**，对应 held-out MC signal efficiency **97%**。
- 最佳质量拟合：`S = 52.44`、`B = 6.76`、`S/sqrt(S+B) = 6.816`、fit p-value `0.2234`。
- 名义 Femto cutflow 中保存的 threshold 与 WP summary 完全一致。

TestSet 的 BDT loose/tight 变化使用 98%/95% MC signal-efficiency 分位数，即 score 0.273677/0.636877，包围名义拟合 WP。重训模型后必须重新计算这些变化。

## 分中心度和 charge channel 的覆盖

| 分析组 | signal SE pairs | signal ME pairs | raw 有效 k* bins | corrected 有效 bins |
|---|---:|---:|---:|---:|
| cent_0_10__ch_0 | 3,815 | 37,275 | 30/30 | 0/30 |
| cent_0_10__ch_1 | 3,807 | 37,860 | 30/30 | 0/30 |
| cent_0_10__ch_2 | 2,606 | 24,925 | 29/30 | 0/30 |
| cent_0_10__ch_3 | 2,508 | 24,493 | 30/30 | 0/30 |
| cent_10_50__ch_0 | 2,000 | 19,237 | 30/30 | 0/30 |
| cent_10_50__ch_1 | 2,016 | 19,004 | 29/30 | 0/30 |
| cent_10_50__ch_2 | 2,068 | 20,647 | 30/30 | 0/30 |
| cent_10_50__ch_3 | 2,070 | 20,412 | 30/30 | 0/30 |

所有 corrected 状态均为 `purity_insufficient_mass_sidebands`。无值 bin 在 JSON 中为 `null`，ROOT TH1/TH2 中保留 NaN，并有独立 valid mask；没有填零或 unity 冒充结果。

## ROOT QA 与持久化

主要 PNG/PDF 由原生 ROOT `TCanvas` 绘制。`outputs/testset/report/analysis.root` 使用 PyROOT 6.38.04 独立重开并验收：

- **8 张 canvas、4,411 个非目录对象**；包括 2,368 个 TH1D、1,368 个 TH2D、552 个 TGraphErrors、3 个 TGraph、112 个 TObjString 和 8 个 TCanvas。
- 新增 `working_point/` ROOT 目录，保存 significance-vs-score、efficiency-vs-score、最佳 Hyper 质量谱及拟合期望。
- `canvases/working_point` 可在新 ROOT 进程中重新 `Draw()` 和 `SaveAs()`。
- 两张关联图的全部 8 条 raw CF 曲线逐点与 JSON 数值和统计误差一致。
- stat/syst/total covariance、valid mask、12 个 variation 以及训练/mixing QA 均通过 ROOT 与机器可读输出的交叉检查。

## 运行证据

| 阶段 | 耗时 / s | 校验产物数 |
|---|---:|---:|
| prepared | 34.681 | 165 |
| training | 6.604 | 3 |
| working_point | 0.181 | 3 |
| scored | 32.626 | 88 |
| nominal | 2.486 | 5 |
| systematics | 30.215 | 63 |
| mc_qa | 0.004 | 1 |
| report | 4.520 | 19 |

8 阶段合计 **111.316 s**。source fingerprint 为 `583899a7c40aeda120472f41ece2ee804c3dfb72bc93b3724a0eb6975f7db76e`；pipeline signature 为 `876ae77df544dde5be5fc74fb42302ff15cbae3868137968e1c455ad93fd1520`。

完整运行日志：`testset_wp_run.log`；机器验收：`testset_wp_validation.json`；验收日志：`testset_wp_validation.log`；单元测试：`unittest_wp.log`。离线 wheel 构建也已通过。

## 物理解释边界

TestSet 的 Hyper WP 最佳点只有 90 个候选，适合验证算法链路，不足以替代正式 period-by-period 稳定性研究。大量 Femto rows 共享候选、pion 和事件；当前 pair-Poisson 与 pair-weighted mass likelihood 是明确标注的近似。正式结果仍需宽 sideband、MCFemto truth/resolution/primary QA、数据/MC控制样本和完整系统研究。
