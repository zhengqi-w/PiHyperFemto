# ROOT 图形与持久化输出

物理分析图使用 **PyROOT** 绘制，ML QA 使用 **HIPE4ML plot_utils**。只导出 PDF，并保留可重读 ROOT 数值对象，HTML 不保存 PNG 或 base64 图片。training/WP 的完整 QA 在各自阶段产生，位于 report 外：

| 阶段 | 全局模型 | 分 pT 模型 |
| --- | --- | --- |
| training | `training/qa/training.root`（数值）和九张官方 ML QA PDF | `training/models/<id>/qa/` |
| working point | `working_point/qa/working_point.root` 和七张 QA PDF（失败时无 best_fit） | `working_point/models/<id>/qa/` |
| systematics | `systematics/qa/systematics.root` 和逐 variation/source/total 的 PDF | 同一系统目录，应用各 pT 模型对应 score 后汇总 |
| 关联汇总 | `report/analysis.root` 与 CF、SE/ME、纯度、mixing 等 PDF | 同一个汇总文件按模型组织元数据 |

训练 QA 包含特征和质量分布、signal/background Pearson 相关矩阵、train/test score、test score、两条 ROC、请求/实际效率对 score 和特征重要性。开启 Optuna 时另存 `qa/optuna_training.pdf`，训练 ROOT 内保存 trial AUC、累计最佳 AUC 和全部状态/参数 JSON。

训练 QA 使用最终入模的两类 train/test；MC 校准组不进入图中。`metadata/training` 和 `metadata/qa_index` 同时记录共同切选、MC 学习范围和逐类 cutflow，便于区分全源计数与有界抽样计数。物理 canvas 导出时按横/纵 A4 边界缩放并保持宽高比，双面板的下方坐标轴完整保留。

WP QA 包含 3σ significance 对 score/efficiency、2σ–4σ 积分窗口填充带、S3/eff pol0、独立 mean/sigma 误差图、3σ raw signal yield、fit stability、最佳扫描质量谱。质量谱单面板展示 data、DSCB signal、pol1/pol2 background 和 total fit，标注 significance、χ²/ndf、mean/sigma、score/效率及拟合状态，不生成 Data−Fit/pull。窗口带表示积分窗变化；三条曲线和 nominal determination 共用同一个 3σ pol0 的预计信号，nominal 为 3σ 曲线最大值。

两阶段均有 `qa/index.json`，同一索引写入 ROOT `metadata/qa_index`。`scan_fits.json` 和 `qa/working_point.root` 保留所有请求点的质量 histogram、fit expectation、参数/物理计数协方差与状态，重复 threshold 的计算共享但请求点不被删除。`mc_signal_fit/` 保存 80-bin MCHyper 质量谱及固定 DSCB tails 的来源。默认不输出数百张单点 fit PDF；`save_score_fit_frames: true` 可以开启。ROOT pol0 保存真实 `TF1`；RooFit DSCB/Chebychev 拟合保存数值结果和 canvas，没有序列化 RooWorkspace。

所有 WP 质量拟合的 total/signal/background 以连续函数曲线绘制，data 保留 histogram 和计数误差。各拟合目录的 `curves/total`、`curves/signal`、`curves/background` 保存密集函数采样的 `TGraph`，取完整质量域归一化 PDF × yield × data bin width。原 `expected`、`signal`、`background` TH1 仍保存精确 bin 积分供数值读取，但不作为拟合函数画到 canvas；report 同步相同曲线。

曲线数值与原生 RooFit PDF 核对，覆盖双侧 tails 的 n<1、n=1 和较大 n。当前绘图回归见 [wp_curves_software_checks.json](../reports/validation/wp_curves_software_checks.json)，完整 TestSet ROOT/PDF 与阶段依赖核对见 [testset_wp_curves_validation.json](../reports/validation/testset_wp_curves_validation.json)。全部扫描点、MC tails、pol0 和最佳 WP 前后数值一致，见 [数值对照](../reports/validation/wp_curves_numerical_comparison.json)。

HTML report 链接阶段 QA，WP 从当前阶段 ROOT 文件复制四个 canvas 到 `analysis.root`，同时导出 `report/working_point.pdf`、`working_point_efficiency.pdf`、`working_point_yield.pdf`、`working_point_mass_fit.pdf`。这些 report 图与独立 WP QA 使用相同对象和数值，不重新拟合。ML overview 直接链接官方 PDF，不额外生成 `report/bdt_validation.pdf`。主 ROOT 仍保存常用 ROC/score/WP 数值对象作为读取接口，完整诊断以阶段 ROOT 为准。独立 `train` 和 `working-point` 已可得到 QA，无需先运行 report。数值与算法口径见 [工作点说明](working_point.md)。

汇总 ROOT 文件只在 `report` / `run` 阶段生成；上游阶段输出 JSON/CSV/NPZ，systematics 还独立生成 ROOT/PDF QA。分 pT 模式按 `training/models/<id>/` 和 `working_point/models/<id>/` 保存常用对象，物理 copy canvas 名增加 `__<id>`，不同模型 AUC 不合并。

系统误差总览采用 spectrum 的百分比阶梯图样式：每个中心度分别绘制 raw 和 corrected 的四电荷道图，每个 panel 用黑色粗线表示 Total，并叠加各 source 的彩色阶梯线。纵轴是 `100 × sigma_systematic / abs(C_nominal)`，单位 `%`；各独立 source 图也按百分比展示。阶梯覆盖实际 k* bin 的左右边界，有效区间分段绘制，缺失 bin 不补零、不跨越连接。来源失败、部分覆盖和完全不可用的 observable 必须保持可见标记。

该缩放只作用于展示对象。ROOT 的 `systematics_qa/sources/.../values`、`graph` 与 `systematics_qa/total/.../values`、`graph` 仍保存 fraction，`0.05` 对应图中的 `5%`，与 JSON 数值及 validity masks 一致。variation 的 `C_variation/C_nominal` 保留原来的比值单位和 `1` 参考线。读取数值时不要从 canvas 的百分比曲线替代原有 fraction 接口。

已验证的环境包含 Python 与 ROOT **6.38.04**。PyROOT 属于运行环境依赖，不能只安装 Python package 就假定已经具备 ROOT。缺少 PyROOT 时图形阶段应明确失败，不切换到其他绘图库。

## 需要保存的对象

| 内容 | ROOT 表示 | 读取时的约定 |
| --- | --- | --- |
| 信号窗、左右及合并 sideband 的 SE/ME | TH1D | 边界与分析 k* binning 一致，误差来自 sumw2 |
| raw / corrected CF | TH1D、有效标志与 TGraphErrors | graph 只包含有效点，histogram 同时保存对应有效状态 |
| 系统误差与合计结果 | graph / histogram 及有效状态 | 不能在系统覆盖缺失时输出有限合计误差 |
| stat / syst / total 协方差 | TH2D | 行列对应同一 k* bin 顺序，保留缺失状态 |
| SE pair-weighted 质量与拟合期望 | TH1D | 不是唯一候选产额，拟合状态必须保留 |
| BDT ROC、score 与训练 QA | graph / histogram | 效率分母与 validation 样本定义来自训练报告 |
| Hyper data WP significance、效率及最佳质量拟合 | graph / histogram | 最终 score 来自候选质量拟合，不来自 Femto pair |
| mixing / occupancy / MC QA | histogram / graph | 缺列或缺 MCFemto 时记录不可用，不填假统计 |
| 展示图 | TCanvas | 保存可重画对象，而不只保存静态图片 |
| 配置、状态、物理解释 | ROOT 文本对象 | 允许脱离原 Python 进程识别对象语义 |

## ROOT 文件目录

`<bin>` 使用与 JSON/CSV 相同的名字，例如 `cent_0_10__ch_0`。`<observable>` 为 `raw` 或 `corrected`。

| 对象路径 | 内容 |
| --- | --- |
| `raw_qa/femto/<bin>/kstar_se`, `kstar_me` | BDT 前、只按中心度分类的绝对 pair 计数 TH1D |
| `raw_qa/hyper/cent_0_10/mass`（其他中心度同理） | BDT 前独立 Hyper 候选质量 TH1D，合并 matter/antimatter |
| `raw_qa/metadata`、各 histogram 同路径的 `_counts` | 选择与计数口径、rows/nonfinite/underflow/overflow |
| `canvases/raw_kstar_cent0`, `raw_hyper_mass` | 输入 QA canvas，其他中心度按序号增加 |
| `metadata/schema`, `config`, `training`, `working_point`, `systematics`, `provenance` | 同一 `metadata/` 目录下的 TObjString JSON |
| `nominal/<bin>/se_signal`, `me_signal` | 信号窗 SE/ME TH1D，误差保存 sumw2 |
| `nominal/<bin>/se_left`, `se_right`, `se_sideband` | 左、右、合并 sideband 的 SE；`me_...` 为对应 ME |
| `nominal/<bin>/cf_<observable>_stat` | CF 的 TH1D，invalid 内容和误差为 NaN |
| `nominal/<bin>/g_cf_<observable>_stat` | 仅有效点组成的 TGraphErrors |
| `nominal/<bin>/valid_<observable>` | 0/1 有效标志 TH1D |
| `nominal/<bin>/cov_<observable>_stat` | 统计协方差 TH2D；`valid_cov_...` 为其有效标志 |
| `nominal/<bin>/mass_se`, `mass_vs_kstar`, `status` | SE pair 质量、质量–k* 分布及状态 JSON |
| `final/<bin>/cf_<observable>_syst`, `cf_<observable>_total` | 系统/合计误差的 CF TH1D，另有 `g_cf_...` graph 与 `valid_...` mask |
| `final/<bin>/cov_<observable>_syst`, `cov_<observable>_total` | 系统/合计协方差 TH2D |
| `final/<bin>/groups/<group>/error_<observable>`, `coverage_<observable>`, `cov_<observable>_syst` | 各系统来源的误差、variation 覆盖与协方差 |
| `variations/<name>/<bin>/...` | 每个变化的对象，结构与 nominal 相同 |
| `systematics_qa/variations/<name>/<bin>/<raw或corrected>/values`, `valid`, `graph` | variation/nominal 比值 TH1D、有效 mask 和无误差 TGraph |
| `systematics_qa/sources/<group>/<bin>/<observable>/values`, `valid`, `graph` | 各来源绝对系统误差除以 abs(nominal CF)，数值为 fraction；另有 `status` coverage 元数据 |
| `systematics_qa/total/<bin>/<observable>/values`, `valid`, `graph` | 总相对系统误差 fraction 和有效标志；canvas 展示时才乘 100 |
| 各 source/total leaf 下的 `percent_steps/metadata`、`percent_steps/run_<n>` | 百分比展示元数据及分段 TH1D；metadata 记录 `scale=100`、单位和各连续有效区间 `[start_bin, stop_bin)` |
| `systematics_qa/metadata`、`variations/<name>/metadata` | std=nominal 的定义、未传播 ratio error 的原因、来源覆盖和 cut 配置 |
| `canvases/systematics__variation__<name>__cent<i>`、`systematics__systematic__source__<group>__cent<i>` | variation 和独立 source canvas 的精确副本；原始 PDF/ROOT 保存在 `systematics/qa/` |
| `canvases/systematics__systematic__total__cent<i>`、`systematics__systematic__total__corrected__cent<i>` | raw / corrected 的 Total 与全部 source 百分比阶梯线总览；raw 延用原 total canvas 名称 |
| `training/roc`, `score_signal`, `score_background` | ROC graph 和 validation score 计数 histogram；density 对象单独提供 |
| `training/model_settings` | 实际后端、最终有效参数、随机种子与 class weighting 的 JSON；分 pT 模式位于 `training/models/<id>/` |
| 阶段 `training.root`：`training/optuna/trial_auc`, `best_auc`, `summary` | 启用时保存完成 trial 的平均 CV AUC、累计最优 graph 与全部搜索状态 JSON；分 pT 同样使用逐模型目录 |
| 阶段 `working_point.root`：`scan/significance_vs_score/significance_3sigma`、`significance_2sigma`、`significance_4sigma`、`window_envelope` | dynamic 窗口的三条诊断曲线和填充带；efficiency 横轴位于 `scan/significance_vs_efficiency/` |
| 阶段 `working_point.root`：`scan/spectrum_yield_pol0`、`spectrum_yield_pol0_function` | 3σ S3/eff 拟合的状态 JSON 与成功时的 TF1；`scan/yield_pol0` 保存同一 nominal normalization 的兼容接口 |
| `working_point/significance_vs_score`, `efficiency_vs_score` | pol0 expected significance 与 held-out MC signal efficiency 扫描 |
| `working_point/best_mass_data`, `best_mass_fit` | 最佳 WP 的 Hyper 重建质量谱与 signal+background 拟合期望 |
| `mixing_qa/occupancy_se`, `occupancy_me` | 超氚一侧 occupancy；有双侧中心度时另写 `delta_centrality_se/me` |
| `mixing_qa/status`, `mc_qa/status` | 始终可解释 QA 是否可用的 JSON 状态 |
| `mc_qa/<bin>__se/g_primary_pion_fraction`、`<bin>__me/...` | 有 MC 输入与真值列时提供的 Wilson 区间 TGraphAsymmErrors；其他 fraction 同样使用 `g_<fraction>` |
| `mc_qa/<bin>__se/resolution`、`<bin>__me/resolution` | 有 MC 输入时保存的生成–重建 k* 二维计数；能否解释需检查对应 truth 状态 |
| `mc_qa/reco_closure/<bin>/...`, `truth_closure/<bin>/...` | 输入与真值条件满足时保存的 closure，结构与 nominal 相同 |
| `canvases/correlations_cent0`, `correlations_cent1` | 两个测试中心度的四道 CF canvas |
| `canvases/pair_spectra_cent0`, `pair_spectra_cent1` | 两个测试中心度的 SE/ME spectra canvas |
| `canvases/working_point`, `working_point_efficiency`, `working_point_yield`, `working_point_mass_fit` | 当前 WP 的 score、efficiency、raw/eff 和最佳质量谱；分 pT 后缀 `__<id>` |
| `canvases/mass_purity`, `mixing_qa` | 质量与 mixing 展示 canvas；ML QA 不保存 TCanvas |
| 阶段 `training.root` 的 `training/optuna/trial_auc`、`best_auc` | 启用时的数值 AUC history；图由 Optuna 封装直接保存 PDF，无 ML canvas |
| `canvases/mc_primary_cent0`, `mc_primary_cent1` | 有 MCFemto 时的分中心度 primary-pion fraction canvas |

表格中同一行省略的前缀仍属于该行指定目录。文件可用 `TFile.ls()` 逐层浏览；更多中心度按配置增加对应 canvas。JSON/CSV/NPZ 是并行的机器可读接口，ROOT 文件不是静态图片的容器。

WP 质量拟合由 RooFit DSCB/Chebychev 模块执行，其他 pair-purity 拟合由独立 Python 统计模块计算。ROOT 保存质量数据、拟合期望、参数/协方差及状态；这些数值对象不冒称 RooWorkspace 或 ROOT TFitResult。

## 数值不能在绘图时改变含义

SE/ME 的 TH1 内容是 `sum(w)`，误差平方是 `sum(w²)`。`inverse_depth` 模式下，两者通常不同；不能用 `sqrt(bin content)` 替代已计算的误差。

CF histogram 的无效 bin 必须与 mask 一起读取；作图时只使用有效点。没有 ME、没有 SE、没有归一化计数或没有可信纯度都不能被画成 `0 ± 0` 或 `1 ± 0`。缺失 corrected CF 是本次 TestSet 的数据状态，不应被 ROOT 的默认 histogram 零值隐藏。

协方差矩阵的缺失元素也不能通过填零变成“无相关”。拟合前必须选择同时具有有效数值和协方差的 bin，并检查矩阵的数值性质；完整矩阵维度不代表每个元素都可用于物理拟合。

TH2 的 **bin content** 与 **bin error** 有不同语义：covariance 与 mask 对象显式创建零值 sumw2，bin error 为零，避免 ROOT 自动产生 `sqrt(covariance)` 这种并未估计的矩阵元素误差。`mass_vs_kstar` 与 MC resolution 等二维计数分布保存 Poisson variance。协方差内容中的 NaN 保持 NaN，不受零值 bin error 约定影响。

ROOT histogram 的 `GetEntries()` 不自动等于唯一候选数，使用 `SetBinContent` 构建的 histogram 也不能据此恢复原始填充次数。SE/ME 积分、weighted sums 与原始 pair-row 计数应结合对应元数据解释。

当前文件约定：可确定未加权计数时设置对应 Entries，derived quantity 或加权汇总的 Entries 为 0；内容及 sumw2 才是数值接口。该约定写在 `metadata/schema`，因此 `Entries=0` 不意味着加权 ME histogram 没有内容。

## 对象生命周期

ROOT 6 的 histogram 默认注册到当前 `gDirectory`。若归属于 TFile，关闭文件会释放对象；需要在文件关闭后继续绘制的 histogram 应有明确的脱离目录和生命周期管理。官方说明见 [ROOT object ownership](https://root.cern.ch/manual/object_ownership/)。

PyROOT 中，画到 canvas 上的 graph、legend、line、histogram 等对象需要存活到 canvas 写入和导出完成。持久化检查必须重新打开 ROOT 文件，而不是只确认原进程里的 canvas 曾经显示过。

## 复查方式

先在独立 Python 进程确认文件能读取并列出对象：

```python
import ROOT

result = ROOT.TFile.Open("outputs/testset/report/analysis.root", "READ")
if not result or result.IsZombie():
    raise RuntimeError("Cannot read the ROOT analysis artifact")
result.ls()

canvas = result.Get("canvases/correlations_cent0")
canvas.Draw()
canvas.SaveAs("correlations_cent0_reopened.pdf")

raw = result.Get("nominal/cent_0_10__ch_0/g_cf_raw_stat")
print("Valid raw CF points:", raw.GetN())
# 保持 result 存活，直到读取和绘图操作完成。
```

检查具体对象时，对照 JSON/NPZ 核对 bin edges、内容、误差、valid mask、graph 点数和 covariance；对保存的 canvas 再执行一次导出，确认 primitives 未因所有权或序列化问题消失。目录中没有 real MCFemto 时，不要求出现真实 MC 统计图，而应有明确缺失状态。

本接口的主要 ROOT API 参考 [TH1](https://root.cern.ch/doc/master/classTH1.html) 和 [TGraphErrors](https://root.cern.ch/doc/master/classTGraphErrors.html)。实际使用的 ROOT 版本应写入运行记录，在线 master 文档不替代版本记录。

## 多 period 输出

合并 `report/analysis.root` 的训练/WP 路径为 `periods/<id>/training[/models/<model>]/` 与 `periods/<id>/working_point[/models/<model>]/`；`metadata/periods` 保存模型上下文、输入重叠和 synthetic repetition 解释。合并 CF 的 `nominal/`、`final/`、`variations/` 路径保持一致。period/model canvas 后缀使用长度编码，避免名称覆盖。部分 MC 覆盖在 HTML 和 MC 图中可见，详见 [period 工作流](periods.md)。
