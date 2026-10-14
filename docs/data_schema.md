# 新版数据接口

框架面向 TestSet 使用的 compact pair tree。ROOT 中每个 `DF_*` 目录都可能有同名树，读取时必须遍历所有匹配树，不能只取第一个 DF。

## 目录与树

| 配置角色 | 默认目录 | ROOT tree | 每行含义 |
| --- | --- | --- | --- |
| `femto` | `Femto` | `O2pihypfemto` | 一条超氚–关联 pion pair，SE/ME 由标志区分 |
| `background` | `Hyper` | `O2hypcands` | 一条重建超氚候选 |
| `signal` | `MCHyper` | `O2mchypcands` | 一条 MC 候选/生成记录；训练必须选择重建真信号 |
| `mc_femto` | `MCFemto` | `O2pihypfemtomc` | 带真值扩展的一条 pair |

实际可接受的 tree 名称由 schema 模块定义；启动前运行输入检查，以实际输出为准。MCFemto 是可选数据源，空目录应产生明确的 missing-input QA 状态，不阻断数据 raw CF。

## 共同候选字段

训练树和 pair tree 的同一物理变量使用不同前缀，schema 层将其转成统一内部字段。

| 物理量 | Hyper / MCHyper | Femto / MCFemto |
| --- | --- | --- |
| Matter/antimatter | `fIsMatter` | `fHypIsMatter` |
| He3 运动学 | `fPtHe3`, `fEtaHe3`, `fPhiHe3` | `fHypPtHe3`, `fHypEtaHe3`, `fHypPhiHe3` |
| 衰变 pion 运动学 | `fPtPi`, `fEtaPi`, `fPhiPi` | `fHypPtPi`, `fHypEtaPi`, `fHypPhiPi` |
| daughter DCA | `fDcaV0Daug`, `fDcaHe`, `fDcaPi` | `fHypDcaV0Daug`, `fHypDcaHe`, `fHypDcaPi` |
| He3 TPC PID | `fNSigmaHe` | `fHypNSigmaHe` |
| TPC crossed rows | `fNTPCCrossedRowsHe`, `fNTPCCrossedRowsPi` | `fHypNTPCCrossedRowsHe`, `fHypNTPCCrossedRowsPi` |
| ITS cluster sizes | `fItsClusterSizesHe`, `fItsClusterSizesPi` | `fHypItsClusterSizesHe`, `fHypItsClusterSizesPi` |
| 主顶点 | `fXPrimVtx`, `fYPrimVtx`, `fZPrimVtx` | 相同 |
| PV→SV 位移（producer 已减 PV） | `fXDecVtx`, `fYDecVtx`, `fZDecVtx` | `fHypXDecVtx`, `fHypYDecVtx`, `fHypZDecVtx` |
| 中心度 | `fCentralityFT0C` | `fCentFT0C` |
| Track occupancy | `fTrackOccupancyInTimeRange` | `fTrackOccupancy` |
| FT0C occupancy | `fFt0cOccupancyInTimeRange` | `fFt0cOccupancy` |

候选质量、pT、eta、phi、ct、cosPA、衰变半径等由运动学和顶点字段重建。ITS cluster size 是 packed 整数；使用前需按各层 nibble 解码，不能把原始整数当连续的平均 cluster size。

`XDecVtx/YDecVtx/ZDecVtx` 已是 `SV−PV`，不是绝对 SV 坐标。cosPA、ct 和 decay radius 按 spectrum 的 `GeneralHelper.hpp` 直接使用保存位移；再次减去 `XPrimVtx/YPrimVtx/ZPrimVtx` 会造成重复扣除。producer 及真实输入对照证据见 [顶点语义审计](../reports/validation/vertex_convention_audit.json)。零位移或零总动量保持无效，不补造 cosPA。

BDT 特征必须同时存在于独立训练树和目标 pair tree，或可由两者共同提供的量重建。比如当前 Hyper 有 `fTofMass`，但 compact Femto 没有相应列；不能因为训练能读到这个变量就将它加入实际应用的 BDT。

## Pair 专用字段

| 字段 | 用途 |
| --- | --- |
| `fIsMixed` | `false` 为 SE，`true` 为 ME |
| `fHadPt`, `fHadEta`, `fHadPhi` | 关联 pion 四动量；与超氚衰变 pion 区分 |
| `fHadSign` | 关联 pion 电荷，配合 `fHypIsMatter` 推导四个 channel |
| `fHadDcaXY` | 关联 pion DCA 选择 |
| `fHadTpcNClsCrossedRows`, `fHadTpcNClsPID` | 关联 pion TPC QA |
| `fHadTpcChi2NCl`, `fHadItsChi2NCl` | 关联 pion track QA |
| `fHadTpcNSigmaPi`, `fHadTofNSigmaPi`, `fHadHasTOF` | 关联 pion PID；TOF 使用需明确 hasTOF 条件 |
| `fIsClosePairRejected` | close-pair 位掩码，见物理契约 |
| `fMixingDepth` | producer 记录的混合深度，作为 QA 或显式权重选择 |
| `fPosZ` | 超氚一侧事件的 z 顶点 |

当前 compact pair tree 不要求 candidate ID、collision ID、run/BC ID，也不重建它们。原始 README 中的 `PairCentFT0C − HypCentFT0C` 等双事件 QA 需要输入确实包含两侧字段；只有一个中心度时无法恢复该量。

## MC 字段和扩展

`MCHyper` 包含生成但未重建的记录。训练只保留 `fIsReco`、`fIsSignal`、`fIsTwoBodyDecay` 满足约定的样本，不能把未重建条目的 sentinel 运动学送入 BDT。

`MCFemto` 的对应 truth 扩展包括 `fHypIsSignal`、`fHypIsReco`、`fHypIsTwoBodyDecay`、`fHadIsPhysicalPrimary`、`fHadIsTruePion`、`fIsTruePrimaryHadHyperPair`、`fIsTruthSelfCorrelation` 和碰撞匹配标志。生成超氚与 hadron 的动量可用于重建 truth k*；不要假定所有 schema 都包含一个直接存储的 `KstarMC`。

不同 MC QA 量需要不同 truth 列。缺列时只将依赖该列的量标记 unavailable，不把未匹配 truth 解释为零 primary fraction。

新增 producer 版本应首先扩展 schema 映射与必要列检查，再增加小样本 fixture 和数据契约测试。不要在 BDT、CF 或绘图函数中散落 `fHyp*`/`fHad*` 分支名判断。

## 本次测试的输入证据

实际树路径、entries 和字段类型记录在 [testset_inventory.json](../reports/validation/testset_inventory.json)，字段清单在 [schema_branches.json](../reports/validation/schema_branches.json)，Femto 质量覆盖范围在 [femto_mass_support.json](../reports/validation/femto_mass_support.json)。这些报告描述本次提供的文件，不自动代表后续生产的数据。
