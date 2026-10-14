# TestSet 输入核查

实际数据路径：`/Users/zhengqingwang/alice/data/derived/PiH3lFemto/TestSet`。本报告由只读全量扫描生成；精确树名、分支名和类型见同目录 `testset_inventory.json` 与 `schema_branches.json`。

| 输入角色 | 实际树 | DF / tree 数 | 原始行数 | 含义 |
|---|---|---:|---:|---|
| Femto | `O2pihypfemto` | 19 | 15,645,511 | SE/ME pair rows |
| Hyper | `O2hypcands` | 1 | 10,088 | background 候选树；训练另取质量 sideband |
| MCHyper | `O2mchypcands` | 17 | 13,183,618 | 包括未重建 generated entries |
| MCFemto | — | 0 | 0 | 目录存在但为空 |

MCHyper 的 `IsReco & IsSignal & IsTwoBodyDecay` 共 **590,085** 行；必须先剔除未重建 entries，再重建质量和训练。不能将 1,318 万 generated entries 全当成重建 signal。

当前 Femto 为新版 compact schema：`fHyp*` 是超氚娘粒子和候选变量，`fHad*` 是配对 pion，`fCentFT0C` 是记录的中心度。四个 charge channels 由 `fHypIsMatter` 与 `fHadSign` 推导。不要求构造或追踪 Femto 超氚 ID。

## 全量原始 SE / ME 计数

这些计数在 BDT、候选选择、close-pair rejection、质量窗口和 k* 范围选择之前，不能直接作为最终 yield 或显著性。

| 中心度 | channel | SE | ME |
|---|---:|---:|---:|
| 0–10 | 0 = H3L + pi+ | 238,752 | 2,319,822 |
| 0–10 | 1 = H3L + pi− | 238,816 | 2,318,325 |
| 0–10 | 2 = anti-H3L + pi− | 207,215 | 2,029,909 |
| 0–10 | 3 = anti-H3L + pi+ | 206,640 | 2,024,077 |
| 10–50 | 0 = H3L + pi+ | 133,912 | 1,280,138 |
| 10–50 | 1 = H3L + pi− | 134,066 | 1,276,767 |
| 10–50 | 2 = anti-H3L + pi− | 149,281 | 1,443,380 |
| 10–50 | 3 = anti-H3L + pi+ | 150,598 | 1,444,238 |

0–10% 共 9,583,556 pair rows；10–50% 共 6,012,380；50–100% 共 49,575。

## 决定可解释范围的限制

- Femto 双精度重建质量范围仅为 **2.98000248–2.99999923 GeV/c²**，即 producer 已裁剪到约 `[2.98, 3.00)`。
- 宽背景 sideband `[2.96, 2.98)` 和 `[3.005, 3.025)` 在 Femto 中均为零。Hyper 背景训练候选仍有对应 sideband，但它不能替代 femto pair sideband 关联函数。
- 缺失 sideband 时，程序应保留可计算的 raw C(k*) 并标记 corrected C(k*) 不可用。不能以零背景、固定纯度、unity 或缩窄且被信号污染的窗口伪造正式 corrected 结果。
- `IsClosePairRejected` 原始计数：0 → 15,642,874；bit 1 → 423；bit 2 → 2,139；bits 1|2 → 75。默认拒绝低两位任意置位，共 2,637 行。4/8 表示不可计算，不应混为 rejection。
- 大量 pair rows 共享候选和事件，pair-Poisson 的统计近似不能证明物理测量已达到精度要求。
- MCFemto 为空，primary fraction、k* response、truth self-correlation 和 MC closure 的扩展入口可以验证结构，但这些物理 QA 不能以当前输入给出结果。

## 执行环境

使用现有 `/opt/anaconda3/envs/MLenv/bin/python`，Python 3.10.16；未安装依赖。现有 NumPy 1.26.4、pandas 2.2.3、SciPy 1.14.1、scikit-learn 1.6.0、uproot 5.5.1、awkward 2.7.2、pyarrow 18.1.0、ROOT 6.38.04。
