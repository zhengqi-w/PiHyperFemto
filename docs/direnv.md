# 使用 direnv 自动加载分析环境

direnv 可以在进入项目目录时加载 `.envrc`，离开时恢复原来的环境变量。它负责激活已有环境、设置项目 import 路径和运行变量；依赖安装仍由 Conda/pip 管理，容器挂载仍由 Apptainer/Singularity 管理。本项目提供 [模板](../.envrc.example)，不自动修改 shell 配置或批准 `.envrc`。

## 首次配置

先在目标机器安装 direnv。按照 [官方 shell hook 文档](https://direnv.net/docs/hook.html)，在 Bash 的 `~/.bashrc` 末尾添加：

```bash
eval "$(direnv hook bash)"
```

Zsh 对应 `~/.zshrc` 和 `eval "$(direnv hook zsh)"`。重新打开 shell 后，在项目根目录执行：

```bash
cp .envrc.example .envrc
```

编辑 `.envrc` 的两个默认值，使其指向目标机器上已安装的环境：

```bash
# 设置为目标机器的 Conda/Miniforge 安装目录和已安装环境
export PIH3L_CONDA_BASE="${PIH3L_CONDA_BASE:-/path/to/miniforge3}"
export PIH3L_CONDA_ENV="${PIH3L_CONDA_ENV:-pih3l-femto}"

# 服务器示例：替换上面两行，使用实际可见路径
# export PIH3L_CONDA_BASE="${PIH3L_CONDA_BASE:-/path/to/miniforge3}"
# export PIH3L_CONDA_ENV="${PIH3L_CONDA_ENV:-/srv/software/pih3l}"
```

`PIH3L_CONDA_ENV` 可以是环境名称或绝对 prefix。模板会 source 对应安装中的 `conda.sh`，然后调用 `conda activate`，让 Conda 的 activation scripts 一起生效。安装或目标环境不存在时明确失败，不继续使用另一套 Python。

确认内容后执行一次：

```bash
direnv allow .
```

之后进入项目即可直接运行：

```bash
python -c 'import sys, ROOT; from hipe4ml.model_handler import ModelHandler; print(sys.executable, ROOT.gROOT.GetVersion()); assert hasattr(ROOT, "RooCrystalBall")'
python -m pip check
python -m pih3l inspect -c configs/testset.yaml
python -m pih3l run -c configs/testset.yaml
```

模板的环境名称与路径由本机 `.envrc` 指定。完整 Python 依赖和 PyROOT/RooFit 由 [Farm 环境定义](../environment.yml)提供。不同平台需安装适合目标系统的环境；首次运行前使用 `pih3l doctor` 检查。

模板保留已有 `PYTHONPATH` 并将项目 `src/` 放在前面；OpenMP/BLAS 线程默认 2，已有值保持不变；ML 绘图使用无界面的 Agg 和可写的 Matplotlib cache。训练的 XGBoost `n_jobs` 仍由分析 YAML 设置。`.envrc` 和 `.direnv/` 已加入忽略规则；实际配置路径只保存在本机，不提交给其他机器。

## 批处理和容器

非交互批任务通常不会执行交互 shell hook。完成上述配置和授权后，在项目目录显式执行：

```bash
direnv exec . python -m pih3l run -c configs/server.yaml
```

`configs/server.yaml` 需要从生产示例复制后设置实际输入和输出；它不是仓库内已经准备好的服务器配置。四类输入以及各 period 的 `root` 必须指向目标进程可见的路径。direnv 不改变这些 YAML 的继承或路径解析规则。

容器中也可以执行 `direnv exec . ...`，但容器需要有 Linux 版 direnv，Conda base（包含 `conda.sh`）、环境 prefix、项目、输入与输出都必须在容器内可见。沿用原有 bind 命令即可；若使用只读项目挂载，需要将 `MPLCONFIGDIR` 设置为可写路径，例如 `/tmp/pih3l-mpl`。宿主上激活 direnv 不会让 `--cleanenv` 容器自动获得相同变量；应在容器内加载，或通过容器的显式环境参数传入。

本次仅对模板进行 Bash 语法检查和本机 Conda/PyROOT/HIPE4ML 激活检查。本机没有安装 direnv，尚未验证真实 hook、目录退出恢复或远程容器运行。
