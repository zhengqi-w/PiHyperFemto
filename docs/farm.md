# Farm 单 period 部署与完整运行

分析的运行入口是安装后的 `pih3l`。依赖安装和激活完成后，只需编辑一个 YAML：输入、输出、中心度、切选、训练、WP、系统变化都由该配置控制，不需要改源代码或设置 `PYTHONPATH`。本说明适用于 Linux CPU 节点；现有容器 bind/作业调度命令可以继续使用。

## 首次安装

在目标 Linux 系统准备 Conda/Miniforge，然后获取项目并安装：

```bash
cd PiHypertritonFemto
bash scripts/install_environment.sh
conda activate pih3l-femto
pih3l doctor
```

安装脚本创建新的 `pih3l-femto` 环境，已有同名环境会失败，避免静默更新正在使用的环境；可用 `bash scripts/install_environment.sh new-name` 指定新名称。[environment.yml](../environment.yml) 是 Farm 和 Linux CI 共用的版本定义，包含 ROOT/PyROOT、Python、数值/ROOT IO、HIPE4ML、XGBoost、ML QA 和 Optuna。脚本以普通 package 安装本项目，并检查依赖和入口；不依赖 editable 安装。

如果已有兼容环境，可以激活后执行 `python -m pip install --no-deps --no-build-isolation .`、`python -m pip check` 和 `pih3l doctor`。缺失或不符合版本约束的依赖需要先补齐。更新源代码后应重新安装 package，正在运行的作业不应更新其环境。

## 唯一的分析配置

```bash
cp configs/farm.single_period.example.yaml configs/farm.yaml
```

编辑 `configs/farm.yaml`：

```yaml
output: /farm/results/pi_h3l/LHC25PbPb
periods: []
inputs:
  femto:
    root: /farm/data/LHC25PbPb/Femto
    patterns: ["AO2D*.root"]
    required: true
  background:
    root: /farm/data/LHC25PbPb/Hyper
    patterns: ["AO2D*.root"]
    required: true
  signal:
    root: /farm/data/LHC25PbPb/MCHyper
    patterns: ["AO2D*.root"]
    required: true
  mc_femto:
    root: null
    patterns: ["AO2D*.root"]
    required: false
```

保留复制文件中的其它配置区，然后设置实际物理选择和训练范围。该 Farm 示例显式 `periods: []`，直接使用四个基础根；通用 production 示例中的非空 period 子配置会覆盖基础输入根，不能只修改基础根后假定它生效。

多个 AO2D 可以通过 patterns 匹配；嵌套目录使用 `"**/AO2D*.root"`，多个存储位置使用 root 列表。绝对 paths 按原值使用，相对 paths 以 YAML 所在目录为基准，支持环境变量。必需的 data 和训练 MC 必须可读；没有 PiHyperFemtoMC 时保持 `root: null, required: false`，workflow 会完成并明确标记 MC QA 缺失。

## 检查和运行

```bash
pih3l doctor -c configs/farm.yaml
pih3l run -c configs/farm.yaml
```

也可从任意工作目录用配置绝对路径运行。`python -m pih3l` 是同一入口。完整 `run` 自动执行 raw QA → 输入登记 → training → WP → compact scored → nominal → systematics → MC QA → report。每个下游阶段处理自己的依赖；输入/model/WP、密集 score-efficiency 映射、图形和 ROOT 输出都在 YAML 指定的 output 中保存。

`doctor` 检查实际 Python、声明的依赖版本、HIPE4ML 的传递依赖、PyROOT/RooFit，以及在临时目录写入并重新打开 ROOT/PDF。带配置时检查每个输入文件和 tree 的配置所需列（包含 variation）、前 32 行的投影/运动学读取、跨 tree 的 compact scored schema，以及输出写入/rename；不训练、不扫描全量 pair、不计算输入文件 SHA256。optional MC 缺失是 warning，不是 failure。机器可读检查使用 `pih3l doctor -c configs/farm.yaml --json`；有失败时退出码为 1。

完整 `run` 也会在运行分析阶段前检查输入契约。输入来自不同 producer schema 时，可选 scored 字段的缺失需要统一处理；移除异构的 `io.scored_optional_columns`，或使用相同 schema 的 ROOT 输入。MC truth/schema 仍必须一致，不会将缺失字段伪造为有效值。

只需原命令再次运行即可续跑，缓存同时核对配置、输入、阶段源码、环境和输出 checksum。`--force` 会重建依赖，普通续跑不要加它。Farm 示例默认 `io.fingerprint: stat`，避免大数据续跑校验时反复读取整个 AO2D；签名记录文件大小和修改时间。需要逐字节输入 provenance 时可改为 `sha256`，代价是读取完整文件。`stat` 无法发现保持相同 size/mtime 的内容修改；`doctor` 的快检也不能替代完整 provenance。

## 批任务、direnv 与 bind

任务实际需要执行的分析命令只有 `pih3l run -c /absolute/config.yaml`。使用 Conda wrapper 可避免批处理 shell 未激活的问题：

```bash
conda run --no-capture-output -n pih3l-femto \
  pih3l run -c /absolute/configs/farm.yaml
```

可选的 [direnv](direnv.md) 已配置时，使用 `direnv exec . pih3l run -c configs/farm.yaml`。direnv 只负责加载环境，不安装依赖或挂载文件。容器中环境、项目/配置、ROOT 数据和输出目录都需要可见，配置填写容器内路径；`--cleanenv` 下显式传递所需运行变量或在容器内加载环境。若项目只读，将 `MPLCONFIGDIR` 指向可写路径。

线程默认受到环境和 YAML 的共同控制。将 `training.model_params.n_jobs` 设置为分配给任务的 CPU 数；OpenMP/BLAS 可以使用 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`，这些不改变 YAML 的训练线程设置。具体 Slurm/PBS/HTCondor 队列、memory、walltime 和 bind 参数由 Farm 现有提交方式提供，本项目不假定调度器。

## 产物和验收范围

`output/report/index.html` 是完整报告，`output/report/analysis.root` 保存物理结果和可重开的 canvas；training/WP/systematics QA 保留在各自目录中，所有图形只输出 PDF。compact pair 数据位于 `output/scored/femto/`，分 pT 模型保存于其子目录。没有 prepared 数据副本。

本机验证覆盖已安装 wheel 在非项目目录的单 period 合成 ROOT 完整 workflow、有效 raw/corrected CF、非空 MC、所有九阶段、ROOT/PDF、系统变化、缓存及损坏重建；真实 TestSet 是软件/产物检查，MC/旁带限制仍保留。Linux 的完整依赖求解由 GitHub CI 使用相同 environment.yml 验证。只有实际绿色的远端 CI 才能作为 Linux 已运行的证据；50 GB 全统计吞吐和物理测量不由这些测试推断。
