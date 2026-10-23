# Operator Learning 文献地图

这份索引把论文分成“先建立共同语言，再掌握两条主线，最后进入前沿”的顺序。不要按发表年份从头扫到尾；每读一篇，都回答四个问题：

1. 学习的是哪个算子，输入和输出分别是什么函数？
2. 离散网格只是表示，还是模型的一部分？
3. 训练分布、测试分布和评价指标是什么？
4. 论文证明或实验真正支持了什么，哪些结论没有被支持？

BibTeX 统一维护在 operator_learning.bib。优先引用正式出版版本；只有在尚无正式版本时保留预印本。

## P0：必须精读

| 主题 | 文献 | 为什么读 |
| --- | --- | --- |
| DeepONet | [Learning nonlinear operators via DeepONet](https://doi.org/10.1038/s42256-021-00302-5) | branch/trunk 分解与算子逼近的核心范式 |
| FNO | [Fourier Neural Operator for Parametric PDEs](https://arxiv.org/abs/2010.08895) | 频域积分核、分辨率迁移与计算效率 |
| 统一框架 | [Neural Operator: Learning Maps Between Function Spaces](https://jmlr.org/papers/v24/21-1524.html) | 从函数空间视角统一理解神经算子 |
| 工具 | [NeuralOperator library](https://github.com/neuraloperator/neuraloperator) | 经过维护的 FNO/TFNO 工程基线 |
| 基准 | [PDEBench](https://arxiv.org/abs/2210.07182) | 数据、任务、指标与基准设计 |

## P1：核心扩展

### 理论与误差

- [DeepONet error estimates](https://arxiv.org/abs/2102.09618)：把误差分解为编码、逼近与重构部分。
- [FNO approximation bounds](https://arxiv.org/abs/2107.07562)：理解 FNO 的普适逼近与复杂度。
- [FNO discretization error](https://arxiv.org/abs/2405.02221)：区分连续算子误差和数值离散误差。
- [Continuum Attention for Neural Operators](https://arxiv.org/abs/2406.06486)：从连续注意力角度重新理解算子学习。

### 物理约束

- [Physics-informed DeepONets](https://doi.org/10.1126/sciadv.abi8605)：用方程残差缓解标签需求。
- [Physics-Informed Neural Operator](https://arxiv.org/abs/2111.03794)：将数据监督与物理残差结合。

### 几何与非规则网格

- [Geo-FNO](https://arxiv.org/abs/2207.05209)：通过几何变形处理复杂域。
- [GINO](https://arxiv.org/abs/2309.00583)：结合图积分与 Fourier operator。
- [Beyond Regular Grids](https://arxiv.org/abs/2205.03548)：关注不规则采样上的神经算子。
- [MeshGraphNets](https://arxiv.org/abs/2010.03409)：重要的网格图网络背景，不等同于标准 neural operator。

### 时间演化与注意力

- [Galerkin Transformer](https://arxiv.org/abs/2105.14995)
- [OFormer](https://arxiv.org/abs/2205.11495)
- [Transolver](https://arxiv.org/abs/2402.02366)
- [CViT](https://arxiv.org/abs/2405.19548)
- [PDE-Refiner](https://arxiv.org/abs/2308.05732)
- [FourCastNet](https://arxiv.org/abs/2202.11214)

### 基础模型与跨方程迁移

- [Poseidon](https://arxiv.org/abs/2405.19101)
- [DPOT](https://arxiv.org/abs/2403.03542)
- [PDEformer-1](https://arxiv.org/abs/2402.12652)
- [PDEformer-2](https://arxiv.org/abs/2502.02088)
- [CoDA-NO](https://arxiv.org/abs/2309.02596)
- [Unisolver](https://arxiv.org/abs/2405.17527)
- [PROSE-PDE](https://arxiv.org/abs/2404.12355)

## P2：基准与应用数据

- [PDEArena](https://arxiv.org/abs/2308.12927)：时变 PDE 的系统比较。
- [The Well](https://arxiv.org/abs/2412.00568)：大规模多物理场数据集。
- [AirfRANS](https://arxiv.org/abs/2212.07564)：外流场 CFD 数据。
- [CFDBench](https://arxiv.org/abs/2310.05963)：参数化 CFD 基准。
- [LagrangeBench](https://arxiv.org/abs/2309.16342)：Lagrangian 模拟基准。

## 参考书与先修材料

- Evans, Partial Differential Equations：PDE 的弱形式、椭圆/抛物型方程基础。
- LeVeque, Finite Difference Methods for Ordinary and Partial Differential Equations：离散、稳定性、收敛性。
- Trefethen, Spectral Methods in MATLAB：理解 Fourier/谱方法的最佳短路径之一。
- Boyd, Chebyshev and Fourier Spectral Methods：更系统的谱方法参考。
- Goodfellow, Bengio, Courville, Deep Learning：优化、正则化和表示学习基础。
- Murphy, Probabilistic Machine Learning: An Introduction：概率、泛化与不确定性工具。

## 建议阅读顺序

1. 第 1–2 周：Evans/LeVeque 选读，加上 Neural Operator 总览。
2. 第 3 周：DeepONet 论文，复现 branch/trunk。
3. 第 4 周：FNO 论文和谱方法材料。
4. 第 5–6 周：PDEBench、误差与离散化论文。
5. 第 7–9 周：按时间、物理约束或几何方向三选一。
6. 第 10–12 周：基础模型、跨任务迁移和个人结课项目。

前沿条目会持续变化；真正开始某个方向时，应再次核对作者版本、正式出版信息和官方代码。
