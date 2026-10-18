# 11 · 写盘：同一块盘，为什么两种写法差 1000 倍

> **一句话在做什么**：在一块普通台式机磁盘（1 TB）上，测"每秒能把多少个 64 KB 的小数据块写进盘里"。对比两种写法——**A 标准文件**（每块都 打开文件→写→关闭）vs **B 预分配块**（先占一个 1 GB 大文件，往固定位置直接写）。本机实测：**A 每秒 1000 个，B 每秒 137 万个**——盘是同一块，差在写法。

## 实验导读

模型对话时产生的"记忆"（KV cache）装不进内存时要落到盘上存着，用的时候再取回来。这层"盘"往往是整个存储层级里最慢的一环，但它到底扛多少写入，通常只有标称数字。这个实验把它实测出来。

跑完（或读完）你会知道：

- 盘这层的写入速度具体是多少（每秒多少个 64 KB 块，数字可直接引用）；
- 两种写法（标准文件 / 预分配块）差多少倍——本机实测差 3–4 个数量级；
- 数据库常开的 "O_DIRECT" 选项，在"大量小文件写"场景下是更快还是更慢（实测：反而慢 1.5×）；
- 写盘的瓶颈在哪——不在盘带宽，在"每次都要新开一个文件"的固定开销。

| 参数 | 人话解释 | 本实验取值 |
| --- | --- | --- |
| 每 op 64 KB | 一次写多大的数据块 | 64 KiB |
| 8192 ops | 一轮总共写多少个 | 8192 |
| 4 / 16 / 32 | 同时用几只"手"在写 | 3 档 |
| buffered / O_DIRECT | 写入走不走系统缓存层 | 2 种 |
| 1 GB | B 写法占用的盘空间 | 1 GiB |

**主题**：KVCache / 数据、存储与状态
**硬件**：任意有本地盘的机器（**与 GPU 无关**，纯存储 I/O）— plan T1–T4 全档
**强化**：第 3–4 周
**状态**：✅ 本机已验证（2026-10-17，本地盘 Micron M600 SATA SSD + ext4，矩阵实测见 `results/local_run.log`）

## 你会怎么猜

写盘的速度大概 = 盘的标称写入速度（官网写的那个），写法差异不大。你的直觉：

- "用标准文件写就行，后端实现差异不大。"
- "O_DIRECT 绕内核页缓存，应该差不多或更慢（少了缓存）。"

## 实测会看到

同一台机器、同一块盘，对两种写法（A 标准文件 / B 预分配块，是什么见开头）在高并发下写入，比每秒能完成多少个。可调：`--concurrency`（几只手）、`--*-odirect`（是否开 O_DIRECT）、`--num-ops`（写多少个），完整命令见「跑法」。

预期看到的反直觉（方向，具体倍数取决于你的盘）：

- **后端实现差异显著**：标准文件路径 vs 裸块路径，在高并发下吞吐可差数倍——KV 写入是**小对象、高并发**的形状，标准 POSIX 写路径的元数据/页缓存开销会被放大。
- **O_DIRECT 是双刃剑**：绕开页缓存省掉一次拷贝和缓存污染，但在小对象高频写下可能损失批量合并的收益——不是"开了就更快"。
- **并发度有拐点**：线程加到某个数后吞吐不再涨（盘/队列饱和），和 lab 03 的 batch 拐点同构。

**本机实测**（2026-10-17，RX 7900 XTX 主机，本地盘 `/dev/sda1` = Micron M600 1TB SATA SSD + ext4；`num_ops=8192`，每 op 64 KiB，完整日志见 `results/local_run.log`）：

| 并发 | local_disk buffered | local_disk O_DIRECT | rust_raw_block buffered | rust_raw_block O_DIRECT |
| --- | --- | --- | --- | --- |
| 4 | 1021 ops/s（8.0 s） | 1008 ops/s（8.1 s） | 1.37M ops/s（6 ms） | 1.09M ops/s（8 ms） |
| 16 | 863 ops/s（9.5 s） | 560 ops/s（14.6 s） | 884K ops/s（9 ms） | 975K ops/s（8 ms） |
| 32 | 820 ops/s（10.0 s） | 503 ops/s（16.3 s） | 612K ops/s（13 ms） | 1.13M ops/s（7 ms） |

实测把上面三条预期全部坐实，而且比"差数倍"更极端：

- **后端实现差 3–4 个数量级**：POSIX 路径每个 64 KiB KV 对象要 `open→write→close` 一个新文件 + chunk 哈希 + LRU 记账（~0.5–1 ms/op），裸块路径是对预分配区域一次 `pwrite`——差的不是"几倍"，是千倍。
- **O_DIRECT 双刃剑坐实**：local_disk 高并发下开 O_DIRECT 反而慢 1.5–1.6×（560/503 vs 863/820），低并发打平（1008 vs 1021）。
- **拐点在左端**：POSIX 路径 4 线程就是吞吐峰值，线程越多越低——瓶颈是元数据/锁争用，不是盘带宽。

**引用上表前必读的数字口径**：

1. ops/s = 内核/设备边界的"写完成"：buffered = 数据进页缓存；O_DIRECT = 设备受理（M600 的 DRAM 写缓存先吸收，刷 NAND 是异步的——本盘刷盘速度用 `dd oflag=direct conv=fsync` 实测为 **116 MB/s**，512 MiB 要 4.6 s）。
2. rust_raw_block 的 1 GiB 设备 = 128 MiB 元数据区 + ~223 个 4 MiB 数据 slot，8192 个 op 反复覆盖同一批 slot（逐出后复用），**唯一落盘数据 ≈ 15 MB**（`du` 实测）——所以它的 ops/s 反映"提交受理能力"，不是顺序介质带宽。要测介质带宽，用 `dd if=/dev/zero of=<file> bs=64k count=8192 oflag=direct conv=fsync` 这类带 fsync 的基线。

**一句话结论**：KV 分级存储的"盘"层不是"一个盘"，而是"盘 + 后端 + I/O 策略"的组合；写入吞吐对这个组合非常敏感，是分级存储设计里容易被低估的一层。

## 为什么会这样

KV block 是相对小、数量巨大的对象，写入路径的固定开销（系统调用、页缓存、元数据）占比高。LocalDiskBackend 走标准文件语义，RustRawBlockBackend 直接管裸块（更接近预分配块设备），两者在小对象高并发下的吞吐形态不同。O_DIRECT 改变的是"经不经过内核页缓存"，在小对象场景下收益和代价并存。这正是 lab 10 里 `B_H2D` / I/O 带宽参数的**底层来源**——你分级存储的盘层实测吞吐，就是决策边界公式里的那个带宽。

## 用了什么工具，为什么

- **为什么用 LMCache 而不是自己写个小程序测**：LMCache 是 vLLM / SGLang 等推理引擎在生产里用的 KV cache 管理层（开源，Apache-2.0）。它里面自带两种"写盘"后端：标准文件写（每个数据块一个文件，大多数程序写文件的常规方式）和预分配块写（先占一个大文件、往固定位置直接写，数据库的写法）。本实验对比的正是这两条**生产路径**——测的是"真实系统写盘时扛多少"，不是自造玩具程序的性能。
- **为什么 B 写法要多一步编译**：它的核心"直接写"系统调用用 Rust 实现（避开 Python 每次调用的开销），需要 `cargo build` 一次（约 30 秒）生成一个小动态库；A 写法是纯 Python，不用编译。
- **O_DIRECT 是什么**：Linux 的一个"打开文件"选项——数据不走系统页缓存，直接写到盘设备上。数据库常开它防止缓存被日志污染。直觉上"少绕一层应该更快"，但本实验"大量小文件写"场景下它反而更慢——为什么见「为什么会这样」。
- **数字口径，引用前必读**：两种写法的 ops/s 都是"内核边界写完成"口径（buffered = 数据进系统缓存；O_DIRECT = 被盘设备受理，内部刷到存储芯片是异步的——本盘真实刷盘速度 `dd` 实测 116 MB/s）。B 的百万每秒表示"写法的受理能力"，不是盘的真实写带宽；要测盘的真实带宽，用带 fsync 的 `dd`。

## 跑法

```bash
# 方式一（官方包）：
pip install lmcache     # 或 pip install -e /path/to/LMCache
python code/storage_backend_io_benchmark.py \
  --num-ops 1024 --concurrency 16 --raw-odirect
# 对比 --raw-odirect 开/关、不同 --concurrency，记录 ops/s
```

**方式二（本机验证过的跑法，2026-10-17）**：不想装完整 LMCache（依赖重，官方包面向 NVIDIA 栈）时，用源码 + 纯 Python 依赖即可跑通本微基准——它是纯 CPU 存储 I/O，不需要 GPU：

```bash
# 1) 独立 venv（--system-site-packages 复用已有的 torch/numpy），装缺失的纯 Python 依赖：
python -m venv --system-site-packages ~/.venvs/lab11
~/.venvs/lab11/bin/pip install sortedcontainers py-cpuinfo aiofile aiofiles msgspec \
  pyzmq aiohttp pyyaml redis safetensors psutil httpx fastapi uvicorn httptools \
  blake3 prometheus_client opentelemetry-api opentelemetry-sdk \
  opentelemetry-exporter-otlp opentelemetry-exporter-prometheus

# 2) rust_raw_block 后端需要一个 Rust 扩展（cargo 编译，~30 s），装进 venv 的 site-packages：
cargo build --release --manifest-path /path/to/LMCache/rust/raw_block/Cargo.toml
cp /path/to/LMCache/rust/raw_block/target/release/liblmcache_rust_raw_block_io.so \
   ~/.venvs/lab11/lib/pythonX.Y/site-packages/lmcache_rust_raw_block_io.so

# 3) 跑。两个关键环境：
#    PYTHONPATH 指向 LMCache 源码树（免 pip 装整个包）；
#    CUDA_VISIBLE_DEVICES= 置空——ROCm 上 torch.cuda.is_available() 恒为 True，
#    不置空 LMCache 会去找编译版 c_ops（CUDA 扩展）；置空后走 non_cuda_equivalents
#    纯 Python 回退，本微基准全程 CPU，数字不受影响。
export PYTHONPATH=/path/to/LMCache CUDA_VISIBLE_DEVICES=
~/.venvs/lab11/bin/python code/storage_backend_io_benchmark.py \
  --num-ops 8192 --concurrency 16 \
  --local-disk-dir /home/<user>/tmp_lab11/bench   # 必须放真实本地盘！/tmp 若是 tmpfs、NAS/NFS 都不行
  --local-disk-odirect --raw-odirect \
  --output-json results.json
# 去掉两个 --*-odirect 再跑一遍 = buffered 对照；换 --concurrency 4/16/32 得完整矩阵。
# 本机矩阵结果（12 行）：results/local_run_matrix.json，运行日志：results/local_run.log
```

此微基准只测存储后端 I/O，**不需要 GPU**，在 AMD 主机上直接可跑。
上层 KV 复用/分级的参考数字（Valkey 4.8× 提速、CPU pinned ~5 GB/s vs L2 <1 GB/s）在本地 `results/reference-valkey-benchmarking.md`。

## 来源

- 微基准脚本 + README：`/mnt/luyuzhou/hpc/LMCache/benchmarks/storage_backend_io/storage_backend_io_benchmark.py`、`README.md`
- 上层背景（KV 复用 / 分级 / 跨实例共享的实测）：`/mnt/luyuzhou/hpc/LMCache/examples/kv_cache_reuse/`（含 Valkey 4.8× 提速、CPU pinned ~5 GB/s vs L2 <1 GB/s 等数字）
- 注：LMCache 主 README 标 "Linux NVIDIA GPU"，但本微基准是纯存储 I/O，与 GPU 架构无关

## 延伸

- 搬运 vs 重算的决策（本 lab 的带宽参数喂给它）见 [10 · KV 搬运 vs 重算](10-kv-transfer-vs-recompute/)。
- 第 3 周阅读：缓存的收益与代价（命中/淘汰/污染）。