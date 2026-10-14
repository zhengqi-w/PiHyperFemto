# 独立架构审查记录

历史审查日期：2026-09-27。下文的测试数量、效率分位数、逐中心度 training QA 与缓存粒度只描述当时版本；当前 HIPE4ML/pol0/独立 QA 契约见 [工作点说明](working_point.md) 和最新软件检查记录。审查与实现、TestSet 验证并行进行；本记录说明检查范围、已修复问题与尚需真实物理输入的部分。最终运行的数值证据以 `reports/validation/` 和实际输出 manifest 为准。

## 完整性检查

| 检查项 | 结论 |
| --- | --- |
| 数据角色分离 | Hyper/MCHyper 用于独立候选训练，Femto 仅作 pair 推理与分析 |
| 新版 compact 契约 | 接受 `fHyp*` / `fHad*`，不要求或重建 hypertriton ID |
| 配置和输入 | 未知字段、非法数值、缺失必要角色/列明确报错；可选 MC 独立记录 |
| 运动学 | 独立四动量与 Lorentz 不变性测试；默认名义 H3L 质量对应 producer |
| BDT | 候选特征白名单、质量/pair/truth 特征禁止、surrogate group 划分、工作点与 QA |
| CF | 2 个测试中心度 × 4 个电荷道完整输出，SE/ME 分开，归一化 Jacobian 协方差 |
| 纯度 | SE pair-weighted 质量拟合，sideband 与 fit 状态参与结果有效性判断 |
| 系统变化 | 同一分析链重新切选/统计/归一化/拟合；来源 group 单独组合 |
| MCFemto | 独立接口，signal-window fraction、truth/collision/self-correlation、resolution 与 closure |
| 执行可靠性 | 依赖检查、写锁、临时目录成功后替换、输入/环境/代码/产物指纹 |
| ROOT 持久化 | TH1 sumw2、TH2 covariance、valid mask、仅有效点 graph、保存并重画 TCanvas |
| 可读性与报告 | 模块职责清楚；README、操作指南、schema/物理契约齐全；图表显示无效状态 |

## 审查中定位并解决的问题

1. **旧 README 与当前输入版本不一致。** 原文保留归档，新文档明确 compact 输入不进行 candidate ID 跟踪；不再将旧 ID 方案强加到新版数据。
2. **无效配置可静默进入计算。** 负质量、NaN 阈值、反序 sigma、非整数 chunk size、输入字段拼错、无效 cut 等现由解析阶段拒绝。每个系统 variation 的最终合并配置也单独校验。
3. **spacelike 四矢量被视作 massless。** k* 入口增加 mass-shell/能量/shape 检查，保留数值舍入容差；非物理输入返回无效。
4. **NaN mixing depth 产生 NaN 权重。** inverse-depth 模式明确要求 ME depth 有限且为正；默认 unit weight 与 producer 约定分开记录。
5. **默认重建质量与 producer k* 定义不同。** 改为 nominal mass，reconstructed 仍是显式配置选项。
6. **RMS 缺失覆盖被变成零协方差。** 使用跨有效 nominal bins 的共同完整 variation ensemble，避免 pairwise 缺失填零及非半正定矩阵。
7. **精确 P=1 被无效 sideband 污染。** 对外部明确指定的 P=1、误差=0，返回 raw 值和协方差；默认缺侧带依旧不做自动修正。
8. **MC fraction 质量分母与数据 CF 不同。** fraction 限制到数据 signal window，并记录未加权 pair-row 约定；增加 true-primary-pair 和两侧 collision matching QA。
9. **BDT 变化必须包围质量拟合 WP。** 当前 TestSet 使用 held-out MC 98%/95% signal-efficiency 分位数，对应 score 0.274/0.637，包围 Hyper 质量拟合的 nominal score；重训后必须同步更新。
10. **ROOT 对象的数值与生命周期风险。** 保留加权 sumw2、NaN 与 valid mask；协方差/mask TH2 的矩阵元素误差设为零，二维计数图保留 Poisson variance；写出 canvas 前保持其 primitives 存活，并使用新进程读取、重画验证持久化。

## 验证证据

在当前已有 MLenv 下完成 Python 编译检查；**74 个 unittest 全部通过**，包括物理不变量、ROOT chunk/schema、Hyper data 质量拟合 WP、同一 score 的 Femto 应用、BDT 应用不依赖关联 pion、统计协方差、纯度与系统变化，以及 synthetic ROOT 全链、缓存/产物篡改、MC signal-window QA 与 stat/syst/total 导出。ROOT 专项检查覆盖 WP graph/质量拟合、加权 histogram、协方差/无效 bin、非空 MC 的 Wilson graph/二维 resolution，以及新进程重新打开 TCanvas 并导出。日志见 [unittest_wp.log](../reports/validation/unittest_wp.log)。正式全量 TestSet 的最终数值核对、签名和耗时集中维护在 [独立验收报告](../reports/validation/acceptance_zh.md)。

源代码使用 Black 24.4.2 统一格式化。`pip wheel --no-deps --no-build-isolation --no-index` 离线构建通过；该检查没有向用户环境安装本 package。

目视检查最终 ROOT 生成的两个中心度四道 CF 图、八组质量图、BDT、SE/ME spectra 与 mixing QA：坐标和状态标签可读，质量尖峰与缺失 sideband 如实保留；pair spectra 增加顶部留白后，两个中心度共八个面板的图例均与数据/误差条分离。HTML 报告提供机器可读数值并展示 ROOT canvas 导出的 PDF，不保存 PNG，也不将图表成功生成当作物理验证。

本地文档链接已逐项检查。GitHub Actions 配置了 Python 3.10/3.12 的 conda ROOT 环境和 unittest 检查，但本次没有声称已经执行远端 CI。

最终输出补充 `systematics/final_observables.csv` 及系统/合计协方差矩阵；训练报告补充 `working_points` 与共享 event/candidate cuts 后的分中心度效率。工作点定义明确限定到配置中心度内、finite-feature reconstructed truth validation MC，且在最终分析 cuts 之前；不再把全局目标 80% 暗示为完整重建或最终 pair 效率。

## 当前明确保留的边界

TestSet 没有 MCFemto，而且 Femto 未覆盖配置中的宽 sideband。因此真实 primary fraction、完整 MC closure 和可信的 sideband-corrected CF 仍需要相应输入。synthetic 测试只能验证实现契约。

统计模型保留独立 pair 近似；训练分组使用 primary-vertex surrogate；系统误差 ensemble 含统计涨落；质量纯度对 k* 的依赖尚需物理验证。FSI/source/scattering 参数拟合、事件级重采样、混合池重建与按区间独立模型属于明确的后续分析扩展。

缓存使用整个 package 源码和完整配置指纹，正确性优先但粒度保守。进一步优化应拆分阶段依赖指纹，并增加相应缓存失效测试；不能为了减少重算直接忽略上游变化。
