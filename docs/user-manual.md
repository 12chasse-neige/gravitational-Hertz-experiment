# Gravitational Hertz Experiment 使用手册

本文档说明当前重构后的项目结构、计算逻辑、常用命令和输出文件。项目现在采用“`ghe/` 作为核心 Python 包，`scr/` 作为兼容命令行入口”的结构。

## 1. `scr/` 现在还需要吗？

需要，暂时不建议删除。

原因是：核心计算已经迁移到 `ghe/` 包中，但项目还没有为每个工作流提供新的包级命令，例如 `python -m ghe.source_array ...` 或安装后的 `ghe-source-array` 命令。当前可直接运行的入口仍然是：

```bash
python main.py
python scr/sourceArray.py ...
python scr/bestPosition.py
python scr/fourier.py
python scr/noiseAnalysis.py
python scr/quantumNoise.py
python scr/runSNR.py ...
python scr/plotSNRCurve.py ...
```

所以现在的判断是：

- `ghe/`：真实计算逻辑，后续新代码应优先写在这里。
- `scr/`：兼容旧工作流的薄包装器，仍然承担 CLI 入口职责。
- 暂时不删除 `scr/`，否则 README 和 `docs/current-workflows.md` 中记录的命令会失效。

将来如果想删除 `scr/`，建议先完成这些替代工作：

1. 为主要功能增加 `python -m ghe...` 或 console script 命令。
2. 更新 README、本文档和测试中的命令。
3. 确认旧脚本 API 不再被 notebooks、测试或外部流程引用。

## 2. 当前目录职责

```text
ghe/
  config.py                 # 物理参数、采样参数、噪声参数、运行配置
  paths.py                  # data/img/runs 等项目路径
  geometry.py               # 球坐标、笛卡尔坐标、旋转、向量搬运
  metric.py                 # 单个机械源到探测器响应的核心计算
  optimization.py           # 最优单源几何优化
  signal.py                 # 多源阵列的时域信号叠加
  spectrum.py               # FFT 和频谱保存
  noise.py                  # 量子噪声 PSD 模型
  snr.py                    # SNR 积分
  source_array/
    schema.py               # 源阵列表结构
    layout.py               # 晶格布局和位置生成
    phase.py                # 响应相位恢复与相位包装
    strategies.py           # exact / rigid / chunk_anchor 生成策略
    io.py                   # CSV / NPZ 读写
    generation.py           # 源阵列生成总控

scr/
  *.py                      # 兼容 CLI；内部调用 ghe/ 包

main.py                     # 总入口：源阵列信号 -> FFT -> SNR
data/                       # 默认数据输出
img/                     # 默认图片输出
runs/                       # 可选的可复现实验输出目录
```

## 3. 总计算流程

当前项目的主流程可以概括为：

```text
配置参数
  -> 单源最优几何
  -> 多源阵列生成
  -> 每个源的相位补偿
  -> 多源时域信号叠加
  -> FFT 到频域
  -> 量子噪声 PSD
  -> SNR 积分并换算到 1 年
```

### 3.1 配置与路径

主要文件：

- `ghe/config.py`
- `ghe/paths.py`

重要配置类：

- `SourceConfig`：机械源参数和单源响应使用的物理常数，例如 `R`、`omega`、`L`、`G`、`c`。
- `DetectorConfig`：量子噪声模型中的探测器参数，例如 test mass、arm length、laser power、mirror transmission。
- `SamplingConfig`：时域采样窗口和采样率。
- `SourceArrayConfig`：源阵列生成参数。
- `NoiseConfig`：SNR 积分频段和 squeezing dB。
- `RunConfig`：可序列化的总运行配置。

支持的环境变量：

```bash
GHE_INT_TIME=0.01             # 时域积分窗口，单位 s
GHE_SAMPLE_RATE_HZ=120000     # 采样率，单位 Hz
LIGO_ARM_LENGTH=1000          # metric 中 SourceConfig.L 默认值，同时影响部分旧流程
LIGO_TEST_MASS=39.6           # noise 中 DetectorConfig.testmass 默认值
```

注意：历史代码里 source-side arm length 和 detector noise arm length 的默认值不完全相同。现在它们已经被分到 `SourceConfig` 和 `DetectorConfig`，但旧流程仍然保留环境变量兼容。

### 3.2 单个机械源的 metric 响应

主要文件：

- `ghe/metric.py`
- `ghe/geometry.py`

坐标约定：

- 探测器顶点是原点。
- `+x` 是干涉仪第一条臂。
- `+y` 是第二条臂。
- `+z` 完成右手系。
- `theta` 是从 `+z` 量起的极角。
- `phi` 是从 `+x` 朝 `+y` 量起的方位角。

单源响应计算步骤：

1. `get_hole_coordinate()` 计算每个孔洞在源本体系中的位置。
2. `calculate_whole_tensor()` 构造源本体系中的四极矩张量 `I_ij`。
3. `second_derivative_of_tensor()` 计算四极矩二阶时间导数。
4. `get_metric_tensor_body_frame()` 用推迟时间 `t - r/c` 计算 metric perturbation。
5. `rotation_body_to_detector()` 把源本体系张量旋转到探测器坐标系。
6. `project_to_tt_gauge_dynamic()` 对局部传播方向做 TT 投影。
7. `calculate_delta_t()` 和 `calculate_delta_t_prime()` 沿每条探测器臂积分光程延迟。
8. `calculate_metric_response()` 返回两条臂的差分响应：

```text
h(t) = (delay_x - delay_y) * c / (2L)
```

### 3.3 最优单源几何

主要文件：

- `ghe/optimization.py`
- `scr/bestPosition.py`

优化变量有四个：

```text
theta_src, phi_src, theta_rot, phi_rot
```

含义：

- `(theta_src, phi_src)`：探测器指向源的方向。
- `(theta_rot, phi_rot)`：机械源转子对称轴方向。

优化目标：

1. 在 `t=0` 和近似正交相位处各算一次单源响应。
2. 用两点合成近似振幅。
3. 最大化该振幅。

结果写入：

```text
data/bestPosition.txt
data/bestPosition.json
```

重新优化命令：

```bash
python scr/bestPosition.py
```

### 3.4 源阵列生成

主要文件：

- `ghe/source_array/generation.py`
- `ghe/source_array/layout.py`
- `ghe/source_array/strategies.py`
- `ghe/source_array/phase.py`
- `ghe/source_array/io.py`

源阵列表每一行代表一个机械源，字段定义在 `ghe/source_array/schema.py`：

```text
source_id
x_m, y_m, z_m
distance_to_detector_m
distance_offset_m
propagation_compensation_s
theta_src, phi_src
theta_rot, phi_rot
gw_phase_offset_rad
rotor_phase_offset_rad
```

生成逻辑：

1. 用 `choose_lattice_dimensions()` 为 `N` 个源选择接近立方体的晶格维度。
2. 用 `positions_for_index_range()` 为源编号生成居中笛卡尔坐标。
3. 把探测器放在阵列中心方向的 `-R * n_src_center` 位置。
4. 对每个源计算实际距离和视线方向。
5. 根据策略决定每个源的转子轴。
6. 用 metric response 恢复实际探测器响应相位。
7. 存储使信号相干到达探测器所需的相位补偿。

生成策略：

- `exact`：每个源都重新优化 `(theta_rot, phi_rot)`。最准确，但最慢。
- `rigid`：把参考源的转子轴刚性搬运到每个源的局部视线方向。最快。
- `chunk_anchor`：每组源只优化中心锚点，其它源近似搬运锚点方向，并用距离差修正相位。

常用命令：

```bash
# 只看摘要和前几行，不写文件
python scr/sourceArray.py --summary-only --num-sources 1000

# 快速刚性近似，不逐源优化
python scr/sourceArray.py --num-sources 1000 --no-optimize-each-source

# chunk-anchor 近似
python scr/sourceArray.py \
  --num-sources 1000 \
  --chunk-center-approximation \
  --approximation-chunk-size 100

# 写 NPZ，推荐给后续程序读取
python scr/sourceArray.py \
  --num-sources 1000 \
  --no-optimize-each-source \
  --format npz

# 同时写 CSV 和 NPZ
python scr/sourceArray.py \
  --num-sources 1000 \
  --no-optimize-each-source \
  --format both
```

默认 `--num-sources` 是 `10000000`。直接运行大规模 exact 生成会非常慢，建议先用小规模和 `--no-optimize-each-source` 或 `--chunk-center-approximation` 验证流程。

### 3.5 多源时域信号叠加

主要文件：

- `ghe/signal.py`

核心思想：

```text
h_total(t) = sum_i h_i(t - phase_offset_i / omega)
```

也就是：每个源根据表中存储的转子相位偏移做时间平移，然后调用单源 metric response，最后在同一个时间轴上相加。

当前实现是行为保持优先的简单版本：

```text
for source_chunk:
    for source_row:
        for time_sample:
            calculate_metric_response(...)
```

比重构前更好的地方是：不会在最内层反复读取 CSV 文件。后续如果要优化性能，应优先在 `ghe/signal.py` 的 chunk 计算处做向量化或并行。

### 3.6 FFT 频谱

主要文件：

- `ghe/spectrum.py`
- `scr/fourier.py`

FFT 使用原项目的归一化：

```text
fft_phys = rfft(signal) * dt
magnitude = abs(fft_phys)
```

零频项会被去掉，因为 SNR 只对正频振荡信号积分。

生成单源 FFT：

```bash
python scr/fourier.py
```

输出：

```text
data/freqs.npy
data/magnitude.npy
img/Fouriered Signal.png
```

### 3.7 量子噪声模型

主要文件：

- `ghe/noise.py`
- `scr/quantumNoise.py`

噪声计算链路：

1. `get_laser_power_in_cavity()` 估计 arm cavity circulating power。
2. `get_standard_quantum_limit()` 计算 `h_SQL`。
3. `get_coupling_constant()` 计算 optomechanical coupling `kappa`。
4. `get_quantum_noise_psd()` 计算未压缩量子噪声 PSD。
5. `squeeze_quantum_noise_with_same_angle()` 或 `squeeze_quantum_noise_with_varying_angle()` 加入 squeezed vacuum 模型。

注意：这些函数返回 PSD，不是 ASD。画图时会取平方根。

画 quantum noise 对比图：

```bash
python scr/quantumNoise.py
```

只画 `gwinc`、旧的频率依赖压缩曲线、detuned signal recycling 曲线三者对比：

```bash
python scr/quantumNoise.py --comparison-only
```

输出：

```text
img/Quantum Noise (Before Squeezing).png
img/Quantum Noise (After Squeezing).png
img/Quantum Noise (Curve Comparison).png
```

### 3.8 SNR 积分

主要文件：

- `ghe/snr.py`
- `scr/noiseAnalysis.py`

SNR 公式保持原项目约定：

```text
SNR = sqrt(sum(4 * |h(f)|^2 / S_h(f)) * df)
SNR_year = SNR * sqrt(YEAR_SECONDS / integration_time)
```

默认积分频段：

```text
1 Hz <= f <= 5000 Hz
```

从保存的单源频谱计算 SNR：

```bash
python scr/noiseAnalysis.py
```

从 `main.py` 产生的总阵列频谱计算时，`main.py` 会自动调用对应 SNR 流程。

## 4. 推荐使用方法

### 4.1 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 运行测试

```bash
python -m pytest -q
```

### 4.3 快速检查当前数据的 SNR

```bash
python scr/noiseAnalysis.py
```

### 4.4 生成一个小源阵列并跑完整流程

推荐先使用刚性近似，避免逐源优化过慢：

```bash
python main.py \
  --renew-source-array \
  --source-array-num-sources 100 \
  --source-array-chunk-size 10 \
  --no-optimize-each-source \
  --source-array-format npz
```

这会：

1. 生成 `data/source_array_distribution.npz`。
2. 读取源阵列。
3. 生成总时域信号。
4. 保存总频谱：

```text
data/total_freqs.npy
data/total_magnitude.npy
```

5. 打印 1 年 SNR。

### 4.5 使用 run directory 保存可复现实验

```bash
python main.py \
  --renew-source-array \
  --source-array-num-sources 100 \
  --source-array-chunk-size 10 \
  --no-optimize-each-source \
  --source-array-format npz \
  --run-dir runs/small-rigid-test
```

输出目录结构：

```text
runs/small-rigid-test/
  config.json
  signal.npy
  spectrum.npz
  snr.json
  plots/
```

### 4.6 只生成源阵列，不跑 SNR

```bash
python scr/sourceArray.py \
  --num-sources 1000 \
  --no-optimize-each-source \
  --format npz
```

### 4.7 对 arm length 和 test mass 做参数扫描

```bash
python scr/runSNR.py \
  --masses "20,39.6,80" \
  --lengths "[1000,4000,1000]" \
  --output data/snr_year_table.csv
```

画图：

```bash
python scr/plotSNRCurve.py \
  --input data/snr_year_table.csv \
  --output "img/SNR (3D).png"
```

## 5. Python API 用法

新代码建议直接调用 `ghe/`。

### 5.1 单源 metric response

```python
from ghe.metric import calculate_metric_response

h = calculate_metric_response(
    t=0.0,
    theta_src=0.20227104,
    phi_src=5.96344729,
    theta_rot=1.72327938,
    phi_rot=0.01416633,
)
```

### 5.2 读取源阵列并计算总信号

```python
from ghe.config import build_time_axis
from ghe.signal import calculate_source_array_signal_from_file

time_axis = build_time_axis()
h_total = calculate_source_array_signal_from_file(
    time_axis,
    "data/source_array_distribution.npz",
    chunk_size=1000,
)
```

### 5.3 频谱和 SNR

```python
from ghe.spectrum import calculate_spectrum
from ghe.snr import calculate_snr_from_arrays

spectrum = calculate_spectrum(h_total)
snr_year = calculate_snr_from_arrays(spectrum.magnitude, spectrum.freqs)
```

## 6. 输出文件说明

```text
data/bestPosition.txt
data/bestPosition.json
```

最优单源几何。

```text
data/source_array_distribution.csv
data/source_array_distribution.npz
```

源阵列表。CSV 便于查看，NPZ 更适合程序读取。

```text
data/freqs.npy
data/magnitude.npy
```

单源 FFT 输出。

```text
data/total_freqs.npy
data/total_magnitude.npy
```

源阵列总信号 FFT 输出。

```text
data/snr_year_table.csv
```

参数扫描输出。

```text
img/*.png
```

绘图输出。

```text
runs/<name>/
```

可选的可复现实验目录。

## 7. 常见注意事项

1. 不要直接用默认 `10000000` 源做 exact 优化试跑。先用 `100` 或 `1000` 源确认流程。
2. `--no-optimize-each-source` 是最快的阵列生成模式，适合流程测试。
3. `--chunk-center-approximation` 是速度和精度之间的折中。
4. CSV 适合人工检查，NPZ 更适合大规模程序读取。
5. 修改 `ghe/spectrum.py` 的 FFT 归一化或 `ghe/snr.py` 的 SNR 公式会改变物理结果，需要单独验证。
6. `scr/` 仍是命令行入口，不要在没有替代 CLI 的情况下删除。

## 8. 当前建议的开发规则

- 新的计算函数写到 `ghe/`。
- 旧脚本只做参数解析、打印、文件路径兼容。
- 修改物理公式时，同时更新测试和 baseline 说明。
- 修改源阵列表字段时，同时更新 `ghe/source_array/schema.py`、本文档和读写测试。
- 大规模性能优化优先放在 `ghe/signal.py` 和 `ghe/source_array/generation.py` 的 chunk 逻辑中。
