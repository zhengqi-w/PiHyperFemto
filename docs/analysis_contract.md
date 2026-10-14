# 物理与数据契约

本文件说明框架必须保持的分析约定，以及结果进入正式物理分析前需要验证的假设。原始设计说明完整保存在 [physics_workflow_reference.md](physics_workflow_reference.md)，作为物理流程的背景资料。当前实现面向新版 compact `PIHYPFEMTO` schema；原文的字段、候选 ID 和去重要求不作为新版输入契约。

## 四种输入承担不同职责

| 输入目录 | 分析对象 | 用途 | 不能据此推断的量 |
| --- | --- | --- | --- |
| `Femto` | 重建的超氚候选与关联 pion 的 SE/ME pair | 数据关联函数、选取 QA | 没有配对的候选或全部碰撞数 |
| `Hyper` | 独立重建的超氚候选 | 质量 sideband 背景训练 | femto pair 的接受度 |
| `MCHyper` | MC 超氚候选 | 真信号训练、训练效率检查 | 数据的背景纯度 |
| `MCFemto` | MC SE/ME pair | primary fraction、物种纯度、真值匹配、动量分辨率、closure QA | 未输出的生成粒子分母或完整重建效率 |

默认训练是候选级分类：`Hyper` 严格质量区间外作背景，`MCHyper` 中满足重建和真信号且严格位于同一质量区间内的候选作信号，默认区间边界 [2.95,3.02]、两端点排除。这两种输入本身是独立候选树，每个 entry 作为一个候选。`Femto` 只做模型推理和 pair 分析，不作为训练背景来源。

## 新版 compact schema 与推理约定

遵循当前分析要求，框架不追踪或重建 `Femto` 的具体 hypertriton ID，不生成候选指纹。候选特征由每条 pair row 的 `Hyp*` 字段得到，在每条 row 上应用同一个 BDT；相同候选特征得到相同分数。这样无需额外的 candidate 表和 score join，也不会把重复 pair 送入训练。

推理中 `IsMixed`、关联 pion 的运动学或电荷、pair 编号和混合深度不进入 BDT 特征。独立候选训练树使用来源文件、树和 primary vertex 组成 surrogate group，再按 group 划分训练与验证。该分组减少共享顶点候选跨集合的机会，但不是真实碰撞 ID；必须保留这一限制。Femto 不建立相应身份，新版 pair tree 本身也不能支持严格事件级重采样。

缺失关键输入应导致明确失败。对可选 QA 字段，报告 `unavailable`，而不是赋零。字段缺失与字段值确实为零的物理含义不同。

## 单位与运动学

动量使用 GeV/c，质量使用 GeV/c²，位置和 `ct` 使用 cm，中心度使用百分数。He3 的 pT 在这里应为电荷修正后的真实动量；不能在不知道 producer 约定时再次乘 2。

超氚四动量由 He3 与衰变 pion 相加，质量由该四动量的 invariant mass 得到。`ct = L × m(H3L) / |p(H3L)|`，其中 `L` 是从主顶点到衰变顶点的三维距离；使用的质量常数必须记录在配置中。`cosPA` 是位移与候选动量方向的夹角余弦。零长度或零动量不能被悄悄变成正常候选。

对于 pair 的不变量 `s`，

```text
k* = sqrt([s − (m1 + m2)²] [s − (m1 − m2)²]) / (2 sqrt(s))
```

候选质量用于 pair 能量时，可选择重建质量或名义质量。该约定会影响 `k*`；重建与 MC 比较以及系统变化必须使用清楚记录的约定。

## 中心度与电荷道

中心度取超氚所在事件。新 schema 中如果只保留 `CentFT0C`，必须确认 producer 写入的是超氚一侧事件的值。不能用 mixed-event 关联 pion 一侧中心度替代它。

测试分析固定保留 `0–10%` 和 `10–50%` 两个区间，以及下面四个道。没有足够统计的组合仍必须有输出状态。

| 编码 | Pair | 符号关系 |
| --- | --- | --- |
| 0 | H3L + π⁺ | same sign |
| 1 | H3L + π⁻ | opposite sign |
| 2 | anti-H3L + π⁻ | same sign |
| 3 | anti-H3L + π⁺ | opposite sign |

如果输入没有 `ChargeChannel`，由 `IsMatter` 与关联 pion 的电荷推导，电荷不能用超氚衰变 pion 代替。区间边界约定必须统一，禁止中心度 10 同时进入两个区间。

## BDT 训练与选择

名义超氚 BDT 仅使用候选拓扑、daughter PID/track QA 等变量。质量不作为名义训练特征，否则会将 sideband 与质量峰的定义直接提供给分类器并造成质量形状偏差。关联 pion、`k*` 和 SE/ME 标签也不属于候选分类器。

MC 真值列只定义训练标签与 QA，不能成为输入特征。每个独立候选 entry 是一个训练样本，训练与验证使用上述 surrogate group 划分，数据来源与划分方法写入训练报告。当前验证集同时用于选择工作点，最终效率应使用另外的 MC 样本验证。工作点的分母是训练范围与 training cuts 内、特征有限的重建 MC 真信号验证候选；训练不按 Femto 中心度筛选或分模型，位于最终分析 cuts 之前；限量抽样和验证划分也属于该样本定义。该效率不等同于总重建效率、最终 pair 通过率或 primary fraction。

训练报告输出密集 `score_efficiency_array`，不再输出逐中心度训练效率表；score/efficiency 映射必须实际使用 HIPE4ML 和固定 held-out 两类样本。最终名义阈值由独立 Hyper 的 DSCB signal + pol1/pol2 background 扫描确定：MCHyper tails 校准后固定，在每点 fitted mean ± 3×core sigma 内积分 S3/B3，对 S3/requested efficiency 做 ROOT pol0 得 `c0`，按 spectrum 的 `ε*c0/sqrt(c0+B3)` 最大值选 WP，并把同一 score 应用到 Femto。raw `S3/sqrt(S3+B3)` 仅作诊断。MC 使用流式 binned tail fit，误差为固定 tails/sideband 形状下的条件协方差；与 spectrum 的 MC binning 和误差传播区别见 [工作点契约](working_point.md)。`ct/fCt` 不得作为 BDT 特征，Femto pair 也不得参与 WP 优化。

分 pT 模式对每个模型独立执行以上流程，并将相同的逐模型 threshold 映射用于 Femto/MCFemto；未被模型区间覆盖的行不参与 BDT 选择。`wp_efficiency_delta` 以该模型 nominal WP 的实际 held-out MC 效率为参考、固定分母求附近阈值；它不是 score 的相对缩放，也不是训练样本定义变化。详见 [模型与效率变化契约](performance.md)。

需要检查的训练证据包括 train/test score、ROC/AUC、样本计数、特征清单、选定阈值与 MC 效率。AUC 很高并不能替代数据与 MC 一致性、质量 sculpting 和 pair acceptance 检查。

## SE、ME、归一化与空 bin

对每个中心度、电荷道和质量区间独立构建 SE 与 ME。已有 pair tree 是上游混合的结果；离线可以选择现有 pair，但不能通过修改配置重新生成缺失的 mixing pool。

```text
alpha = sum(ME in normalization range) / sum(SE in normalization range)
C_raw(k*) = alpha × SE(k*) / ME(k*)
```

归一化区间和 histogram bin 必须具有一致的边界定义。若 SE 或 ME 的归一化积分为零，该组合的归一化无效；若某个 k* bin 的 ME 为零，该 bin 的比值无效。无效值应有 mask/状态与原因，不能填成 1、0 或拿相邻 bin 代替。

加权 histogram 的统计方差是 `sum(w²)`。producer 写入的 `MixingDepth` 不自动意味着每个 pair 应乘 `1/depth`。当前本机 producer 双向填充 mixed pairs，同一候选在不同时间参与混合；改变权重需要独立检查物理与接受度的一致性。

独立 pair 的 Poisson 误差不能完整描述共享候选与共享碰撞引入的相关性。alpha 本身也会在 k* bins 之间产生相关。正式拟合若要使用协方差，需使用经验证的协方差模型或另行提供的事件重采样产物；当前 compact pair tree 不重建事件或候选身份。

## Close-pair 标志

当前本机 producer 的 `IsClosePairRejected` 为位掩码，而非单纯 bool：

| bit | 含义 |
| --- | --- |
| 1 | angular 检查拒绝 |
| 2 | distance 检查拒绝 |
| 4 | angular 检查不可用 |
| 8 | distance 检查不可用 |

producer 会保留带标志的 pair。拒绝标志与不可用标志应分别配置并分别 QA；把所有非零值一概解释成“拒绝”会改变选择含义。历史文件是否包含该列，必须通过 schema 检查确认。

## 纯度与 sideband 修正

ME 修正接受度，不会自动扣除超氚组合背景。采用

```text
C_signal = [C_raw − (1 − P) × C_sideband] / P
```

时，必须同时满足：有效的 raw CF、有效的 sideband CF、`0 < P ≤ 1`、可信的质量拟合或已验证的外部纯度，以及 sideband 对信号窗口背景的代表性。

当前框架对每个中心度、电荷道的 SE pair 质量分布做拟合，得到 **pair-weighted purity**。同一个超氚候选与多个 pion 配对时会贡献多次，这与正在分析的 pair 样本一致；它不是唯一候选产额或唯一候选纯度。拟合的统计不确定度采用 pair-independent 近似，必须保留这一标签。中心度、电荷道积分的 pair purity 也不必然等于每个 k* bin 的纯度；粗 bin 纯度用于所有 k* 是明确的模型假设，应纳入系统研究。

统计不足或拟合失败时保留 raw 结果与失败状态；不能自动使用 P=1，也不能把 fallback 结果标成成功拟合。纯度及 sideband 的不确定度需要随修正传播。修正结果在有限统计下可能小于零，不能通过裁剪数值隐藏问题。

## MCFemto 的使用边界

primary fraction、true-pion fraction、真信号超氚比例、true-primary pair fraction 是不同的量。优先分别输出，再决定分析所需的乘积及其相关性。

`IsTruePrimaryHadHyperPair` 一类字段可能要求同一 MC 碰撞；这个标签在 ME 中为假是正常现象，不应被用作所有 ME pair 的“污染”判据。只有先明确分母与 truth 条件，primary fraction 才有可比较的意义。

MC QA 应至少记录 SE/ME 分开计数、选取后 fraction、truth self-correlation、reco/MC 碰撞匹配，以及有足够字段时的 k* resolution。缺少 truth 字段应报告不可用。只基于输出的重建 pair tree，不能计算包含未重建对象的总重建效率。

没有加入 femtoscopic FSI 的 generator 仍可能包含非 femto 相关、全局守恒或事件结构。`C≈1` 是一个需要结合 generator 和 mixing 的 closure 参照，不能凭“无 FSI”自动成为严格预言。

## 系统误差与结果等级

每个变化必须重跑它影响的候选选择、pair 选择、SE/ME 归一化、纯度及最终 observable。只移动已经归一化的最终 CF，不能代表 selection systematic。

同一来源的上下变化应先组成一个来源的不确定度，再按约定合并不同来源。RMS/envelope、对统计涨落的处理、有效 variation 数和被拒绝原因都应记录。无有效 variation 的 bin 是“未评估”，不是零系统误差。

`final_observables.csv` 的 total error 来自统计与系统协方差相加的对角元，依赖二者独立的组合假设；由于当前 variation spread 含统计涨落，该合计不是统计去偏后的完整物理预算。所有数值应与 `valid`、来源覆盖和修正状态共同使用。

`systematics/qa/` 的 std 严格指 nominal CF：variation ratio 为 C_var/C_nom，各来源及总相对不确定度为 sigma_syst/abs(C_nom)。零分母或无效 observable 留 null/mask；共享 pair 的 ratio cross-covariance 未计算，因此 ratio 不给统计误差条。每个 variation 的原始 CF 仍保存自身完整统计协方差。source 的 expected/completed/failed、逐 bin coverage 和 RMS 共同 ensemble 大小应与相对误差共同读取，partial spread 不是完整系统预算。

训练定义、特征或随机种子变化涉及模型重训；producer 的 mixing pool 宽度变化涉及上游再生产。能够在离线配置中表达一个名字，不等同于已经执行该类系统研究。

交付时区分四个等级：代码已实现、流程已执行、产物结构完整、物理结果已验证。小统计 TestSet 可以证明前几项，通常不能独立证明最终物理结论。

## 持久化契约

默认 `prepared/` 仅保留输入清单、来源描述和 summary，不保存候选或 pair Parquet。训练/WP 直接分块读取候选 ROOT；apply 分块读取 Femto/MCFemto，推理后仅写 compact scored。`mass_h3l`、`kstar`、`pt_h3l` 是 H3L mass、k*、pT 的统一内部名称。文件另外自动保存中心度、SE/ME、电荷道、score、模型路由及所有已声明 nominal/systematic 切选所需列，MC 保存可用 truth QA 列。原始 daughter 动量和仅用于 BDT 的特征默认不保存。

WP 先确定每个模型的最低保留 score，覆盖最高预留 MC efficiency、nominal 和全部已声明变化；写入时采用 `score >= minimum_threshold` 保留 ties。裁剪不改变这些选择接受的 pair，但改变 scored cutflow 的起始分母，原始计数应从 raw QA 和 scored summary 读取。降低阈值到保存范围之外、扩大已保存质量/k*/中心度范围或新增未保存切选列，需要从原始 ROOT 重新 apply。独立 WP cuts、fit 范围和存储配置见 [紧凑存储说明](scored_storage.md)。
