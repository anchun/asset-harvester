# Asset Harvester 开发者指南

本文档从开发和研究工程的视角解释 Asset Harvester 的完整流程，适合以下场景：

- 理解仓库如何把稀疏的目标观测转成 3D Gaussian 资产
- 将整套流程适配到非 NCore V4 的自定义数据集
- 找到提升 parser 质量、多视图生成质量和 3D lifting 质量的正确切入点
- 为主要学习模块规划微调工作

本文重点说明仓库中的具体实现，而不是只停留在论文层面的高层描述。

## 1. 流程总览

Asset Harvester 可以理解为一个分阶段的流水线：

1. 将原始驾驶数据或目标中心化的带 mask 图像转换成按目标组织的多视图表示。
2. 将稀疏条件视角整理成统一的推理输入格式。
3. SparseViewDiT 围绕目标生成 16 张目标视角图像。
4. TokenGS 将这些生成视角提升为 3D Gaussian 表示。
5. 可选地生成元数据文件，使输出目录可以被 NVIDIA Omniverse NuRec 消费。

主要有两条使用路径：

- `ncore_parser/run.sh` + `run.sh`
  适合从 NCore V4 驾驶日志开始。
- `run.sh --image-dir ...`
  适合已经有目标中心化图像和 mask，但没有 NCore clip 的情况。

## 2. 仓库结构地图

仓库中最重要的几个区域是：

- `ncore_parser/`
  将 NCore V4 clip 解析成按目标组织的 crop、mask 和相机元数据。
- `run_inference.py`
  多视图扩散和 TokenGS lifting 的主运行入口。
- `models/multiview_diffusion/`
  SparseViewDiT 推理流程、相机/光线预处理、VAE 和 transformer 加载逻辑。
- `models/tokengs/`
  TokenGS 的推理与训练代码。
- `models/camera_estimator/`
  AHC 相机估计器，在只有 mask 图像时使用。
- `utils/generate_external_assets_metadata.py`
  生成用于 NuRec 外部资产插入的 `metadata.yaml`。

## 3. 端到端数据契约

这个仓库最关键的设计点之一，是各阶段之间交接时所使用的数据格式。

完成 parsing 后，每个目标样本应当大致呈现为：

```text
<data_root>/
  sample_paths.json
  <class_name>/
    <sample_id>/
      input_views/
        camera.json
        frame_00.jpeg
        mask_00.png
        frame_01.jpeg
        mask_01.png
        ...
      metadata.json
      reserved_views/             # 可选，不一定存在
```

`run_inference.py` 对 `input_views/camera.json` 的最小字段要求为：

- `frame_filenames`
- `mask_filenames`
- `normalized_cam_positions`
- `cam_dists`
- `cam_fovs`
- `object_lwh`

这其实就是自定义数据集接入时最有价值的抽象层。如果你能够自行生成这种目录结构，就可以完全绕过 `ncore_parser`，直接复用下游的多视图扩散和 TokenGS 流程。

## 4. 阶段一：NCore Parser

### 4.1 目标

NCore parser 的作用是把原始 NCore V4 clip 变成目标中心化的多视图 crop、mask 和相机元数据。概念上它做了四件事：

1. 从 NCore 加载 track 和传感器数据
2. 将 cuboid track 投影到相机视角，并裁出目标中心图像
3. 结合几何和分割结果过滤掉质量差的视角
4. 写出简化后的下游推理格式

### 4.2 入口

主要包装脚本：

```bash
bash ncore_parser/run.sh --component-store <clip-or-manifest>
```

这个脚本最终转发到：

- `python3 -m ncore_parser`

CLI 定义在 `ncore_parser/src/ncore_parser/__main__.py` 中。

### 4.3 Parser 接受什么输入

`--component-store` 支持：

- clip manifest `.json`
- 逗号分隔的 component-store 路径
- `.zarr.itar` glob 模式

parser 会先把 clip manifest 解析成真实的 component store 路径，再开始处理。

### 4.4 内部处理流程

内部处理逻辑可以概括为：

1. `NCoreParser` 初始化：
   - 一个 `NCoreObjectParser`
   - 一个 `Mask2FormerSegmentationEstimator`
2. `NCoreObjectParser.extract(...)` 遍历 tracks、lidar 帧和选中的相机。
3. 对每个 track-view 候选，执行：
   - 将 3D cuboid 投影到相机图像
   - 过滤投影失败或投影面积过小的视角
   - 通过 lidar 和 ray-box intersection 估计遮挡程度
   - 将目标 crop 到目标分辨率
   - 计算归一化相机方向、距离、FOV 和目标尺寸
4. `NCoreParser` 再对每张 crop 图像运行实例分割。
5. 根据 cuboid 投影结果选择最匹配的实例 mask。
6. 进一步过滤：
   - 遮挡严重的视角
   - 分割质量差的视角
   - 目标过小的视角
   - 目标中心偏离 crop 中心太远的视角
7. 最后把结果写成 `run_inference.py` 能直接消费的简化格式。

### 4.5 实现细节要点

#### 类别归一化

原始 NCore 类别会在写样本前做归一化。例如：

- `VRU_pedestrians` 变成 `person`
- `consumer_vehicles` 变成 `automobile`

如果你的自定义数据集类别定义不同，这里通常是最早需要改动的地方之一。

#### 相机位姿表示

parser 会保存：

- `normalized_cam_positions`：物体坐标系中的单位相机方向向量
- `cam_dists`：相机到物体的距离
- `cam_fovs`：相机视场角
- `object_lwh`：物体长宽高

这组字段就是从 NCore 几何世界过渡到 multiview diffusion 和 TokenGS 阶段的桥梁。

#### 视角质量过滤

视角过滤并不只是靠分割结果。parser 还会使用：

- 投影面积过滤
- 基于 lidar 的遮挡率过滤
- mask 质量和 mask 位置过滤

这一点很重要，因为下游模型对条件视角的质量非常敏感。

### 4.6 Parser 输出

parser 会写出：

- `sample_paths.json`
  存储相对样本路径列表，供 `run_inference.py` 使用。
- `input_views/frame_XX.jpeg`
  裁剪后的目标图像。
- `input_views/mask_XX.png`
  实例 mask。
- `input_views/camera.json`
  规范化后的相机/目标元数据。
- `metadata.json`
  轻量级样本元信息。

### 4.7 自定义数据集接入建议

如果你的数据不是 NCore 格式，通常不建议强行转成 NCore。更高效的方式是直接复现 parser 输出契约。

推荐路径：

- 如果你已经有目标 crop、mask 和相机元数据：
  直接写出 `input_views/` 格式。
- 如果你有相机元数据但没有 mask：
  写一个轻量级自定义 parser，并复用 `utils/image_segment.py`。
- 如果你只有带 mask 的目标图像：
  使用 `run.sh --image-dir ...`，让 AHC 估计相机元数据。

### 4.8 Parser 最值得优化的点

如果样本质量差，优先提升 parser，而不是先改生成模型。

收益最高的改进项通常是：

- 更好的实例分割
- 更好的目标中心化裁剪
- 更严格的遮挡过滤
- 更稳定的类别归一化
- 更准确地把相机标定转换成 `normalized_cam_positions`、`cam_dists` 和 `cam_fovs`

## 5. 阶段二：多视图扩散运行时

### 5.1 目标

多视图扩散阶段从 1 到 4 张条件视角图像出发，生成围绕目标的一圈 16 张视角图像。

运行入口是：

```bash
bash run.sh --data-root ./outputs/ncore_parser --output-dir ./outputs/ncore_harvest
```

或者，如果你已经有自定义的带 mask 图像：

```bash
bash run.sh --image-dir <image-root> --output-dir <out>
```

### 5.2 两种输入模式

`run_inference.py` 支持两种模式：

- `--data_root`
  期望输入是 parser 风格目录，包含 `sample_paths.json` 和 `input_views/camera.json`。
- `--image_dir`
  期望每个样本一个目录，其中有 `frame*.jpeg|jpg` 和匹配的 `mask*.png`。

当使用 `--image_dir` 时，流程会先调用 AHC estimator 来预测：

- `distance`
- `lwh`
- `fov`
- `cam_pose`

这样就能补齐原本由 NCore parser 提供的相机/目标字段。

### 5.3 运行时样本发现

在加载重模型之前，`run_inference.py` 会先：

1. 发现样本
2. 校验图像和 mask
3. 可选地运行 `ImageGuard`
4. 构造轻量级 sample specs

这个设计对于批量处理很有用，因为坏样本可以在前面尽早剔除。

### 5.4 稀疏条件视角策略

在 `data_root` 模式下，代码最多使用 4 张条件视角。如果可用视角更多，会在相机位置上做 farthest-point sampling，挑出更有多样性的子集。

这样做的原因是：

- 太多冗余视角会增加计算，但未必增加有效信息
- 多样化的条件视角更有利于生成一致的多视图结果

### 5.5 `preproc(...)`：SparseViewDiT 的真实接口

多视图预处理中最关键的函数是：

- `multiview_diffusion/data/nre_preproc.py: preproc(...)`

它接受 `MVData`，输出供模型直接使用的 `AttrDict`，其中包括：

- `x`
  供模型输入的带背景处理图像
- `x_original`
  原始图像
- `x_white_background`
  白背景版本的各视图
- `x_msk`
  二值 mask
- `n_target`
  要生成的目标视角数
- `cam_poses`、`dists`、`fovs`
- `intrinsics`
- `c2w_relatives`
- `plucker_image`
  每个视图的 Plucker 光线嵌入
- `relative_brightness`

在推理用的 eval 模式下：

- 最多 4 张真实可用视图作为 conditioning views
- 16 个规范化的环绕相机视角作为 target views

因此推理布局可以理解为：

- conditioning views：真实观测到的视角
- target views：围绕物体的一圈 16 个目标视角

### 5.6 SparseViewDiT 实际上使用了哪些条件

多视图管线并不是只依赖输入 RGB 图像。

它同时显式使用：

- 基于 C-RADIO 的条件图像嵌入，并对多个条件视图做聚合
- Plucker 光线嵌入
- 每视图相机嵌入
  `c2w_relatives` 展平后的 16 个值，再加上 FOV
- conditioning mask
  用于标记哪些视图已知、哪些视图需要生成
- 相对亮度 `relative_brightness`

这也是为什么相机元数据的质量如此关键。这个模型本身就是显式带几何条件的。

### 5.7 SparseViewDiT Pipeline 内部流程

从高层看，`SparseViewDiTPipeline` 做的事是：

1. 用 C-RADIO 编码条件图像
2. 用 VAE 将图像编码到 latent 空间
3. 将 Plucker rays 编码或下采样到 latent 网格
4. 拼接相机和条件信息
5. 用 classifier-free guidance 运行 SparseViewDiT transformer
6. 只将 16 个 target latent 解码回 RGB 图像

### 5.8 分辨率和视角数

当前公开推理配置基本固定在：

- 图像分辨率：`512 x 512`
- 目标视角数：`16`
- 条件视角数：最多 `4`

这在运行时和模型构建代码中有多处固定，不只是 CLI 层面的默认值。

### 5.9 Multiview Diffusion 输出

对每个样本，`run_inference.py` 会写出：

- `multiview/0.png ... 15.png`
  生成的目标视角图像
- `multiview.mp4`
  生成视角的视频
- `multiview/fov.txt`
- `multiview/dist.txt`
- `multiview/lwh.txt`
- `input/frame_*.jpeg`
- `input/mask_*.png`

其中 `multiview/` 目录就是后续 TokenGS 的核心输入。

### 5.10 不重训练时的主要质量杠杆

如果不改模型，提升多视图生成质量的最有效方法通常是：

- 提升条件 mask 的质量
- 提升相机元数据的质量
- 提升条件视图的多样性
- 增加 `--num-steps`
- 调整 `--cfg-scale`
- 在更看重数值稳定性时使用 `--precision fp32`

## 6. 阶段 2A：AHC 相机估计器

AHC 是一个辅助模块，但对自定义数据非常关键。

### 6.1 它做什么

给定带 mask 的图像，它会预测：

- `distance`
- `lwh`
- `fov`
- `cam_pose`

输出布局为：

```text
[distance(1) | lwh(3) | fov(1) | cam_pose(3)]
```

### 6.2 为什么它重要

如果你的自定义数据集没有可靠的相机/目标几何元数据，这个模型就是整条 Asset Harvester 流程还能跑起来的关键。

### 6.3 它的限制

这个仓库只提供了 AHC 的推理代码，没有完整公开的训练流程。如果你要提升 AHC 在自己领域上的表现，通常需要：

- 拿到原始训练流程，或者
- 基于 `AttributeModel` 自己搭一个回归训练作业

## 7. 阶段三：TokenGS Gaussian Lifting

### 7.1 目标

TokenGS 负责读取生成好的多视图图像，并预测 3D Gaussian 表示。

运行时通过以下组件集成：

- `models/tokengs/tokengs/lifting_inference.py` 中的 `TokengsLiftingRunner`

### 7.2 TokenGS 使用的输入

对每个样本，lifting 会使用：

- 16 张生成好的 multiview 图像
- `fov`
- `dist`
- `lwh`

其中 `lwh` 会被用来把相机距离归一化到模型内部的尺度约定中。

### 7.3 内部流程

lifting 运行时做的事情可以概括为：

1. 加载 TokenGS checkpoint
2. 读取 checkpoint metadata，例如：
   - `input_res`
   - `num_gs_tokens`
3. 将输入 resize 到 TokenGS 的输入分辨率
4. 构造和 multiview 生成阶段一致的 orbit cameras
5. 预测 Gaussian 参数
6. 从预测出的 Gaussians 渲染一圈可视化图像
7. 导出 `gaussians.ply`

### 7.4 当前仓库中的 Gaussian 表示

有几个实现上的重要事实：

- 当前导出的 PLY 只使用 0 阶 SH
- 保存的 PLY 只包含 `f_dc_*` 字段
- 当前模型不会预测更高阶的 `f_rest_*` SH 系数

当前每个 Gaussian 学到的通道包括：

- position
- opacity
- scale
- rotation
- RGB 或 DC color

### 7.5 容量

TokenGS 的容量主要由以下参数控制：

- `num_gs_tokens`
- `deconv_patch_size`

当前仓库默认 TokenGS 配置使用：

- `num_gs_tokens = 4096`
- `deconv_patch_size = 8`

对应的结构上限是：

```text
4096 * 8 * 8 = 262144 Gaussians
```

实际写入 `gaussians.ply` 的点数通常更少，因为导出器会裁掉低 opacity 的高斯。

### 7.6 尺度语义

TokenGS 会用 `lwh` 来归一化相机半径，但导出的 `gaussians.ply` 不应简单理解成“自动保证真实尺度一致的 metric mesh 替代品”。

更实用的理解方式是：

- `gaussians.ply`
  归一化物体空间下的 Gaussian 资产
- `multiview/lwh.txt`
  与该资产对应的目标尺寸信息

这也是为什么 `generate_external_assets_metadata.py` 会同时把 PLY 路径和尺寸信息写进 `metadata.yaml`。

### 7.7 TokenGS 输出

对每个样本，lifting 会输出：

- `3d_lifted/0.png ...`
  从 Gaussian 表示重新渲染出的视图
- `3d_lifted.mp4`
- `gaussians.ply`

### 7.8 TokenGS 最值得优化的点

在不重训练的情况下：

- 先提升 multiview 图像质量
- 先提升条件 mask 质量
- 保持相机元数据一致性

涉及模型改造时：

- 在目标领域数据上微调
- 增加 `num_gs_tokens`
- 考虑启用或扩展 test-time training 路径
- 如果需要更高阶 SH 或更高的高斯密度控制能力，重新设计输出 head

## 8. 阶段四：NuRec 元数据

在 lifting 结束后，`utils/generate_external_assets_metadata.py` 会扫描输出目录树，寻找所有包含 `gaussians.ply` 的完整样本，并生成：

- `metadata.yaml`

每个资产条目包含：

- `ply_file`
- `label_class`
- `cuboids_dims`

这是将 Asset Harvester 输出交给 NuRec 外部资产工作流所需的最后一步。

## 9. 如何把 Asset Harvester 适配到自定义数据集

### 9.1 推荐的接入策略

最快的方式通常不是重写整个流程，而是选择最轻量的接入层。

常见有三种策略。

### 9.2 策略 A：模拟 `ncore_parser` 输出

适合你已经掌握以下信息的情况：

- 目标 crop
- 前景 mask
- 大致相机方向
- 距离
- FOV
- 目标长宽高

你只需要生成 parser 风格目录结构，并自己写出 `camera.json`。这样就可以几乎无改动地直接接入下游 Asset Harvester。

这通常是以下场景下最好的方案：

- 自定义 AV 数据集
- 有标定相机的机器人数据集
- 合成数据集

### 9.3 策略 B：使用 `--image-dir` 加 AHC

适合你只有以下数据的情况：

- 带 mask 的目标图像
- 没有可靠相机元数据

每个样本目录结构如下：

```text
<image_root>/
  sample_000/
    frame.jpeg
    mask.png
  sample_001/
    frame.jpeg
    mask.png
```

然后运行：

```bash
bash run.sh --image-dir <image_root> --ahc-ckpt <ckpt> --output-dir <out>
```

这种路径很好用，但整体效果通常弱于直接提供真实相机元数据。

### 9.4 策略 C：编写自定义 Parser

适合你的原始数据包含以下信息：

- 多相机图像
- 跟踪标注
- 相机内参与外参
- 可能还有 lidar 或 depth

你不需要百分之百复现 NCore parser 的所有细节。真正的关键是输出高质量的 `input_views/` 样本，并附带可信的相机元数据。

### 9.5 自定义数据集检查清单

对于每个样本，建议检查：

- 目标是否居中，像素面积是否足够大
- mask 是否贴合、是否发生明显偏移
- 条件视图之间是否具有足够的视角多样性
- `normalized_cam_positions` 是否是在统一物体坐标系下的单位方向向量
- `cam_dists` 和 `object_lwh` 是否使用一致单位
- `cam_fovs` 是否与下游使用的 crop 几何一致
- 类别是否被映射到稳定的标签体系中

## 10. 微调与模型优化路线图

### 10.1 先修 Parser，再谈训练

很多看起来像模型能力不足的问题，本质上其实是输入质量问题。

在计划微调前，先检查：

- crop 是否合理
- mask 是否准确
- 视图是否多样
- 遮挡是否过重
- 相机元数据是否一致

如果这些环节质量差，直接重训后面的模型往往收益有限。

### 10.2 Multiview Diffusion 微调

这个仓库包含 SparseViewDiT 的主要推理栈，但没有像 TokenGS 那样直接可运行的公开训练脚本。

目前已经具备且可复用的部分包括：

- `MVData` 和 `preproc(...)`
- 相机和 Plucker ray 条件构造逻辑
- SparseViewDiT transformer 实现
- VAE 和 checkpoint 加载工具
- 推理配置类

实际做 multiview fine-tuning 时，建议的工程路径是：

1. 保持当前运行时接口不变
2. 基于现有预处理和模型代码自行补一个训练 harness
3. 使用和推理契约一致的目标中心多视图数据训练
4. 保持和当前推理一致的条件协议：
   - 最多 4 个 observed views
   - 16 个 target views
   - 默认 512 分辨率，除非你明确要训练一个新变体

适合做 multiview diffusion fine-tuning 的情况包括：

- 新的目标类别
- 新的 mask 或背景风格
- 新的传感器域
- 更强的视角外推能力
- 更好的纹理一致性

训练数据中最有价值的属性是：

- 稳定的 mask
- 一致的白背景 target，或你自己明确设定的背景约定
- 准确的每视图相机几何
- 训练目标里足够广泛的方位角覆盖

### 10.3 TokenGS 微调

在这个仓库里，TokenGS 是最容易直接开展微调的部分，因为已经公开了训练入口。

训练入口：

```bash
cd models/tokengs
python main.py tokengs ...
```

默认的 `tokengs` 配置已经指向 `assetharvest` 数据模式。

### 期望的数据集布局

TokenGS 训练大致期望的数据目录为：

```text
<dataset_root>/
  <sample_id>/
    recon_input/
      0.png
      1.png
      ...
      15.png
      fov.txt
      dist.txt
      lwh.txt
    cond_views/               # 可选，但通常很有帮助
      0.jpeg
      0_mask.png
      ...
      fov.txt
      dist.txt
      lwh.txt
      cam_pos.txt
```

### 默认数据注册表的注意事项

当前 `models/tokengs/tokengs/data/registry.py` 中的数据路径是写死的。实际使用时，通常要么直接改这里的路径，要么将其重构为通过 CLI 传入数据根目录。

### 微调策略

推荐顺序是：

1. 先做同结构微调
   - 不改 `num_gs_tokens`
   - 不改 `deconv_patch_size`
   - 不改 `img_size`
2. 从公开 checkpoint 继续训练
3. 在此基础上再尝试更大容量模型

原因是：

- 同结构微调可以最大化复用 checkpoint 权重
- 增加 `num_gs_tokens` 或改变 head 结构会引入 shape mismatch，导致一部分参数随机初始化

### 什么时候增加 TokenGS 容量

当瓶颈明确来自 3D 细节不足，而不是 multiview 图像质量差时，再考虑加大模型容量。

常见信号包括：

- 资产整体结构对了，但细节偏糊
- 几何基本有了，但不够精细
- 大型车辆或复杂类别需要更多 primitives

如果增加 `num_gs_tokens`，建议同时考虑启用基于已有 tokens 的初始化方式，而不是让新增容量完全从随机状态开始训练。

### 10.4 AHC 微调

如果你的自定义数据主要依赖 `--image-dir` 路径，那么提升 AHC 往往也非常有价值。

当前仓库已经提供：

- 推理封装
- 回归模型定义

但没有完整公开的 AHC 训练流程。因此如果你要做领域适配，通常意味着基于以下监督信号自己搭建训练任务：

- 带 mask 的目标图像
- 真实 `distance`
- 真实 `lwh`
- 真实 `fov`
- 真实相机方向

## 11. 质量提升排查手册

当输出质量不理想时，建议按以下顺序排查。

### 11.1 第一步：先修数据

优先检查：

- mask 是否贴合且居中
- 条件视图是否覆盖不同方位
- 相机距离和 FOV 是否一致
- crop 是否过近或过远

### 11.2 第二步：调推理参数

对于 multiview diffusion：

- 提高 `--num-steps`
- 调整 `--cfg-scale`
- 对数值稳定性敏感的实验使用 `fp32`

对于 TokenGS：

- 先确认 multiview 输出已经足够好，再怀疑 lifting
- 分开检查 `3d_lifted.mp4` 和 `gaussians.ply`

### 11.3 第三步：微调 Multiview Diffusion

适用于：

- 生成视图之间不一致
- 纹理在不同视角之间漂移
- 未见过视角外推效果很差

### 11.4 第四步：微调 TokenGS

适用于：

- 生成视图已经不错
- 但 3D 资产仍然结构松散、过于平滑或者细节不足

### 11.5 第五步：升级模型容量

只有在前面几步都比较稳定后，再考虑这一步。

典型方向包括：

- 更高的 multiview 分辨率
- 更多的 TokenGS tokens
- 更高阶 SH
- 更复杂的 Gaussian densification 策略

## 12. 面向自定义资产提取开发的推荐实施顺序

针对新数据集，更稳妥的开发路线通常是：

1. 先构造一个 20 到 50 个目标的小规模样本集
2. 先实现 parser-compatible 的 `input_views/` 输出
3. 在不改核心模型的情况下跑通 `run_inference.py`
4. 检查：
   - 条件 mask
   - 生成的多视图图像
   - lifting 后的 Gaussian 渲染结果
   - 最终 PLY 的统计信息
5. 先修 parser 和数据问题
6. 再调推理参数
7. 第三步再做 multiview diffusion 微调
8. 第四步再做 TokenGS 微调
9. 只有现有管线明显达到瓶颈时，再考虑增加模型容量

## 13. 总结

理解 Asset Harvester 最有用的心智模型是：

- `ncore_parser` 把原始日志变成干净的目标中心稀疏观测
- SparseViewDiT 把稀疏观测视图补全为规范化的一圈多视图
- TokenGS 再把这一圈多视图提升为 3D Gaussian 资产
- `metadata.yaml` 则让这些输出可以接入下游仿真工具链

对自定义数据集来说，最重要的抽象其实不是 NCore 本身，而是 parser 输出契约。一旦你能稳定地产出 `input_views/` 和 `camera.json`，下游绝大部分系统都可以直接复用。

对效果提升来说，最重要的原则是优先优化前面阶段。更好的 mask、更好的 crop、更好的相机元数据，以及更好的多视图生成，通常会比一开始就把 Gaussian 模型做大带来更显著的收益。
