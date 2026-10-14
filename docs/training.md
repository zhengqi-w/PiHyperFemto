# HIPE4ML 训练、模型参数与 QA

训练统一使用 **HIPE4ML `ModelHandler` + XGBoost**。创建、训练评估、推理、保存和读取均经过 `ModelHandler`，配置不再提供 `training.backend` 或 HGB 分支。独立 `MCHyper` signal 与 `Hyper` sideband 候选分块读取、上限抽样；Femto pair 不参与训练或 WP，训练不按中心度分模型。`ct/fCt`、质量、pair pion、k*、SE/ME 和 truth 标签不得成为模型特征。

## 与 spectrum 对应的调用链

参考 H3l spectrum 流程中的 `_train_and_save_model_per_bin` 组织方式，实际调用如下：

| 用途 | 本项目调用 |
| --- | --- |
| 模型与训练特征 | `ModelHandler(XGBClassifier(...), training_columns=features, model_params=params)` |
| 最终训练与测试 | `handler.train_test_model([train, y_train, test, y_test], return_prediction=True, output_margin=True, sample_weight=weights)` |
| 打分 | `handler.predict(frame, output_margin=True)` |
| 保存与读取 | `dump_model_handler("model_handler.pkl")`、`ModelHandler().load_model_handler(...)` |
| 效率阈值 | `analysis_utils.score_from_efficiency_array(test_labels, test_scores, efficiencies)` |
| 模型特征分布与相关 | `plot_utils.plot_distr`、`plot_utils.plot_corr` |
| 分数与 ROC | `plot_output_train_test`、`plot_distr`、`plot_roc_train_test` |
| 效率与重要性 | `plot_bdt_eff`、`plot_feature_imp`（SHAP） |

官方接口见 [ModelHandler](https://hipe4ml.github.io/hipe4ml/model_handler.html) 和 [plot_utils](https://hipe4ml.github.io/hipe4ml/plot_utils.html)。HIPE4ML 本身使用 XGBoost 的 sklearn-compatible 接口，并依赖 sklearn；本项目保留分组切分和 ROC/AUC 统计工具，但不再自行维护 sklearn 分类器和双后端模型流程。

所有训练、Optuna、QA、WP 和 apply 分数统一使用与 spectrum 相同的 `output_margin=True` 原始 margin，不做 sigmoid 或缩放；输入特征也不归一化。outer held-out 按来源文件/树/primary vertex surrogate group 隔离，score mapping 失败不换 split 或自动重训。group key 仍不是真实碰撞 ID，正式效率需要独立 MC。旧 probability 模型、NPZ 和 scored metadata 不允许作为当前 margin 产物加载。

默认特征仅为 `DcaV0Daug、DcaHe、DcaPi、NSigmaHe、cos_pa`；`decay_radius`、两条 daughter 的 TPC crossed rows、mass 和 ct 不进入模型。`feature_distributions.pdf` 另加 H3L mass 的 signal/background 分布，标注 `QA only`；相关矩阵和特征重要性仍只使用五个实际输入。质量也用于候选选择和独立质量 QA。`training.signal_mass_window: [2.95, 3.02]` 定义 MC signal 的严格内区间以及 data background 的严格外区间：`2.95 < mass < 3.02` 与 `mass < 2.95 or mass > 3.02`；两个端点均排除。此选择先于 MC 范围校准、reservoir 和训练切分；WP 仍读取完整 Hyper 质量区。

顶点字段保存的是 producer 已减去 PV 的位移。`physics.derive` 按 spectrum `GeneralHelper.hpp` 使用 `L=|XDec,YDec,ZDec|`、`radius=hypot(XDec,YDec)`、`ct=L*m/P`、`cosPA=(P·Dec)/(P*L)`，不再次减 PV，也不裁剪 cosPA 隐藏异常。PV 仍用于分组和事件选择。实际输入审计见 [vertex_convention_audit.json](../reports/validation/vertex_convention_audit.json)。

## 基于 MC 分布的自动输入范围

两份示例配置启用 `training.mc_range_selection`，按 MC 变量分布裁剪 signal/background 的输入宽尾。这里没有新增固定 pT、PID 或 decay-radius 阈值，`training.candidate_cuts` 保持独立的可选显式切选。自动范围保持区间内原来的变量值和两类形状。

```yaml
training:
  mc_range_selection:
    enabled: true
    features: null
    quantiles: [0.001, 0.999]
    margin_fraction: 0.05
    reference_fraction: 0.1
    min_reference_rows: 50
```

`features: null` 使用当前模型的 BDT 输入，可显式指定其子集；不允许质量、ct、标签或来源 ID 进入范围学习。每个 pT 模型先得到有上限的 MC 样本，再按完整 surrogate group 预留 10% 的独立校准组。校准组移出 classifier training、held-out 和 Optuna，仅用来计算一次 `numpy.quantile(method="linear")`。对每个变量的分位区间 `[q_low,q_high]`，两侧各扩展 `margin_fraction*(q_high-q_low)`，冻结为包含两端的筛选范围。分位跨度为零的变量明确记录并跳过，样本/组数不足则失败，不借入 held-out 或自动调整分位点。

其余 MC 在有界样本中按范围筛选；background 在分块读取后、reservoir 采样前筛选。范围作为模型定义写入 metrics/bundle，并同步用于 Hyper WP、Femto/MCFemto 推理；越界行不进入模型的 `bdt_in_scope`。因此 WP 的 score efficiency 分母是在相同范围内的 held-out signal，属于共同预选后的条件效率；不能代替 MC 预选接受率或完整物理效率。有限校准样本的分位数也不保证全 MC 有精确的 99.8% 接受率。

`metrics.json` 的 `training_selection.mc_range_selection` 保存参考分位值、最终边界、随机种子、参考组 hash、组数、样本数和范围口径；`input_cutflow` 保存两类从来源、显式 cuts、pT、背景 sidebands、finite features 到自动范围和采样的计数。每个变量同时记录独立拒绝数和依次筛选的通过数。MC 范围筛选发生在采样/移除参考组之后，其计数明确属于有界 MC 样本；background 的范围计数属于采样前全流。prepared 表没有上游 truth/kinematics 计数时写 null。`qa/index.json` 与 ROOT `metadata/training` 保存相同记录；所有分布、correlations、score、ROC、SHAP 都使用真正进入模型的 train/test，校准组不混入 QA。

## 参数与配置

`training.model_params` 覆盖单一 XGBoost 默认值；随机种子统一来自顶层 `seed`。默认参数与九项 Optuna 搜索域参考 H3l spectrum 配置。仅保留 canonical `gamma`，不同时配置别名 `min_split_loss`。

```yaml
training:
  class_weight: balanced
  model_params:
    objective: binary:logistic
    booster: gbtree
    n_estimators: 800
    max_depth: 6
    learning_rate: 0.05
    min_child_weight: 3
    gamma: 2
    subsample: 0.7
    colsample_bytree: 0.73
    reg_alpha: 0
    reg_lambda: 1
    tree_method: hist
    max_bin: 256
    n_jobs: 4
    eval_metric: auc
  efficiency_min: 0.29
  efficiency_max: 0.99
  npoints_for_effi: 701
  qa_bins: 100
  qa_shap_max_per_class: 1000
  use_optuna: false
```

`class_weight: balanced` 在实际拟合分区内计算每行权重，经 ModelHandler 传给 XGBoost；设 `null` 使用单位权重。验证 AUC 未加权。`qa_shap_max_per_class` 限制官方 `plot_feature_imp` 每类最多使用的 held-out 候选数；小样本采用两类最小样本量。局部设置随机种子并恢复 NumPy RNG 状态，避免 QA 改变训练/优化随机序列。

旧 `backend`、HGB 参数、`permutation_importance_repeats`、`qa_signal_efficiencies`、`qa_centrality_bins` 均被配置校验明确拒绝。特征重要性 PDF 改为官方 SHAP beeswarm/bar；`metrics.json` 和 ROOT 同时保存 XGBoost normalized gain，二者定义不同，不能混用。

训练效率目标为 `unique(round(linspace(min,max,npoints),3))`，默认 0.29–0.99 共 701 点。TXT 为 N×2，第一列 score、第二列请求 efficiency。固定 held-out 两类分数决定 HIPE4ML 插值范围；记录请求/实际效率、signal 分母、版本和实现 SHA256。缺 spline root 时只把失败目标向零移动一个 float64 ULP 后调用同一 API；无 quantile fallback。效率 1 的 inclusive 阈值为该固定 held-out 的最小 margin；storage reserve=1 则用 null 表示不裁剪。库映射使用严格 `>`，apply/storage 使用 `>=` 保留 ties。

## Optuna 与样本隔离

`use_optuna: true` 在 outer training partition 内进行 grouped CV，优化平均 AUC，outer held-out 不进入 objective。每个 fold 通过新的 ModelHandler `fit`、`predict(output_margin=True)` 完成拟合和评估，最优参数再用于最终 `train_test_model`。

没有直接调用 `ModelHandler.optimize_params_optuna`：当前 HIPE4ML 实现内部用普通 `cross_val_score(cv=nfold)`，不提供本项目所需的 group 隔离、逐 fold class weights 和 typed/log/step/categorical 搜索。保留一个 Optuna objective 来保障这些分析条件；没有保留第二套模型训练后端。

```yaml
training:
  use_optuna: true
  optuna:
    metric: auc
    direction: maximize
    n_trials: 20
    timeout: 300
    cv_folds: 3
    n_jobs: 1
    n_startup_trials: 5
    pruner: median
    n_warmup_steps: 1
    search_space:
      learning_rate: {type: float, low: 0.015, high: 0.12, log: true}
      max_depth: {type: int, low: 3, high: 8}
      n_estimators: {type: int, low: 100, high: 1000, step: 100}
```

空 `search_space: {}` 使用默认 XGBoost 搜索域；非空字典替换整个域，未列参数保留基础值。支持 int/float/categorical，严格检查类型、边界和组合。groups 或 fold 中某类不足时失败，不借入 held-out。timeout 约束调度，不能中断正在拟合的 fold；并行或墙钟预算可能改变搜索序列。Optuna history PDF 使用官方 `optuna.visualization.matplotlib.plot_optimization_history(study)`。

## 分 pT 与缓存

每个 pT bin 可覆盖 features、candidate cuts、model_params、use_optuna/optuna、效率网格，独立生成 ModelHandler、MC 自动范围、QA、WP；class weighting、MC 范围配置和 QA 规模使用全局设置。bin candidate cuts 与全局 cuts 递归合并，同变量的显式规则可覆盖，其他共同规则保留。pT 范围均为 `[low,high)`，不允许重叠。只训练部分区间时，未覆盖 pair 不借用其他模型。

```yaml
training:
  use_optuna: false
  pt_bins:
    - id: pt2_4
      pt_range: [2, 4]
      model_params: {max_depth: 4}
      use_optuna: true
      optuna:
        n_trials: 12
        search_space:
          max_depth: {type: int, low: 3, high: 5}
    - id: pt4_8
      pt_range: [4, 8]
      model_params: {n_estimators: 1000}
```

`retrain_bins` 控制局部重训，其余模型只在来源签名和全部 artifact SHA256 通过时复用。缓存绑定 ModelHandler、analysis_utils、plot_utils 的实际源码 SHA（含 editable 安装），XGBoost、Matplotlib、SHAP、ROOT 与相关统计依赖版本。旧 `model.joblib` 不自动转换或加载；首次运行新架构需要重训并重新 apply，旧 scores/WP 不可视作新模型结果。

## QA 与输出

全局模型位于 `training/`，分 pT 位于 `training/models/<id>/`：

| 产物 | 内容 |
| --- | --- |
| `model_handler.pkl` | HIPE4ML 官方 pickle，包含模型、参数和特征顺序 |
| `metrics.json` / `model_manifest.json` | 有效参数、模型范围/id、split、效率、AUC/KS、来源与 checksum |
| `validation_scores.npz` | held-out label/raw-margin score/score_domain；不保存完整候选表 |
| `score_efficiency_array.txt` | N×2 score/requested efficiency |
| `qa/feature_distributions.pdf` | train+held-out 上限抽样候选的实际模型特征分布，另加仅供 QA 的 H3L mass |
| `qa/correlations_signal.pdf` / `correlations_background.pdf` | 两类 Pearson 矩阵 |
| `qa/bdt_output.pdf` / `bdt_test_score.pdf` | train/test 对比分数与 held-out 分数 |
| `qa/roc_train_test.pdf` / `efficiency_vs_score.pdf` | 两条 ROC、请求/实际效率 |
| `qa/feature_importance.pdf` / `feature_importance_shap.pdf` | 官方 SHAP bar / beeswarm |
| `qa/training.root` | 数值 histogram、ROC/efficiency graph、相关矩阵/有效标志、gain 和元数据；无 ML TCanvas |
| `qa/index.json` | 每组 QA 的官方函数、参数、样本范围、类别标签和 ROOT 对象路径；同一索引也写入 ROOT `metadata/qa_index` |
| `optimization/summary.json`、`trials.csv`、`split.json` | Optuna 最优参数、全部 trial 与内部 fold 隔离记录 |
| `qa/optuna_training.pdf` | 启用 Optuna 时的官方 history 图，数值记录也进入 training.root |

全部 ML PDF 由 HIPE4ML plot_utils 生成，Optuna history 用 Optuna 封装；只导出 PDF、及时关闭 figure，不保存 PNG。ROOT 数值使用统一公共 bins，官方分布 PDF 使用其 histogram API；不能假定 PDF 与 ROOT 的自动 bin 定义完全相同。

生成 QA 时使用内存中的有界 train/test frame，调用后释放，不为绘图另写 Parquet。WP/CF/质量/systematics 图继续使用 PyROOT 并保留 TCanvas；report 直接链接 ML PDF，仅复用物理阶段 canvas。HIPE4ML、XGBoost、Optuna、Matplotlib、SHAP 已列为必需依赖；use_optuna 仍是优化执行开关。PyROOT 由分析环境提供，不自动安装。

此前 ModelHandler 迁移的 162 项软件回归全部通过，包含合成 ROOT 完整 workflow、分 pT、真实小规模 Optuna、官方 QA、模型缓存和独立进程 ROOT/SHAP 顺序测试。历史检查记录见 [HIPE4ML 软件检查](../reports/validation/hipe4ml_handler_software_checks.json)。新的 QA 检查和真实 TestSet training/WP 更新状态分别记录在 `reports/validation/qa_refresh_software_checks.json` 与 `testset_qa_refresh.json`；只运行这两个阶段不会自动更新已有 scored/CF/report。
