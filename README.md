# Operator Learning from Zero

一套从数学基础、DeepONet 与 Fourier Neural Operator（FNO），逐步走到独立解决偏微分方程算子学习问题的开源学习路径。

本仓库不是论文链接的堆积。每个阶段都包含：

- 明确的学习目标；
- 最少但足够的必读材料；
- 可运行实验；
- 练习与验收标准；
- 从分布内拟合到分辨率迁移、分布外测试和长期稳定性的评测要求。

## 学完以后能够做什么

完成主线后，你应该能够：

1. 把一个科学计算问题写成函数空间之间的算子学习问题；
2. 判断问题更适合 DeepONet、FNO、几何神经算子，还是传统数值方法；
3. 独立生成可信的 PDE 数据并避免数据泄漏；
4. 训练并公平比较 DeepONet、FNO 和有限维基线；
5. 区分通用逼近、离散化无关性、分辨率泛化和分布外泛化；
6. 为新的 PDE 或动力系统设计数据、模型、损失与评估协议；
7. 完成一个可复现的研究型结课项目。

## 推荐路线

| 阶段 | 周数 | 产出 |
|---|---:|---|
| 数学与数值基础 | 1–2 | 手算并数值求解一维椭圆方程 |
| DeepONet | 3 | 理解 branch/trunk 分解并完成点查询模型 |
| Neural Operator 与 FNO | 4 | 实现谱卷积和网格到网格预测 |
| 公平复现 | 5–6 | 在同一数据上比较 DeepONet、FNO 与基线 |
| 时间系统与物理约束 | 7–8 | 完成 rollout 与 PDE residual 实验 |
| 复杂几何和前沿模型 | 9–10 | 阅读 GINO、Transolver、PDE foundation models |
| 研究设计与结课项目 | 11–12 | 完成自选问题、消融实验和复现报告 |

完整教学计划见 [curriculum/README.md](curriculum/README.md)。

## 五分钟运行

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/run_experiment.py --config configs/smoke.json
pytest
```

冒烟实验会：

1. 随机生成一维变系数 Poisson 方程的数据；
2. 用同一训练集分别训练 DeepONet 与 FNO；
3. 在独立测试函数上计算相对 L2、边界误差与频谱误差；
4. 将结果保存到 `artifacts/smoke/summary.json`。

`configs/smoke.json` 只用于验证环境和流程。正式实验请使用 `configs/full.json`，并根据硬件和研究问题调整样本量、分辨率与训练轮数。

## 核心问题

仓库中的第一个完整任务是学习解算子

\[
-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right)=f(x),\qquad
u(0)=u(1)=0,
\]

也就是从系数函数映射到解函数：

\[
\mathcal G:a(\cdot)\mapsto u(\cdot).
\]

选择这个问题是因为它足够小，可以在 CPU 上跑通完整流程；同时又包含算子学习最关键的组成：函数分布、数值真值、传感器、查询坐标、边界条件、网格分辨率和分布外变化。

## 仓库结构

```text
operator-learning-from-zero/
├── curriculum/       # 12 周课程、作业和验收标准
├── docs/             # 概念、训练、评测和结课项目指南
├── references/       # 分层阅读清单和 BibTeX
├── configs/          # CPU 冒烟与正式训练配置
├── src/              # 数据、模型、损失、训练和评估代码
├── scripts/          # 一键实验入口
├── tests/            # 数值、模型和指标测试
└── .github/          # CI 与贡献模板
```

## 学习原则

- 先建立可靠的数值基线，再训练神经网络。
- 先跑通最小问题，再扩大数据、模型和方程难度。
- 训练集与测试集按“函数实例”划分，而不是按网格点划分。
- 报告分布内结果，也报告参数、频率、网格、几何或时间跨度上的分布外结果。
- “模型能在另一分辨率上运行”不等于“模型能恢复训练数据没有包含的高频信息”。
- 只在 many-query 场景中比较神经算子和传统求解器的总成本。

## 阅读入口

- [核心概念与模型选择](docs/core-concepts.md)
- [训练经验手册](docs/training-playbook.md)
- [统一评估协议](docs/evaluation-protocol.md)
- [结课项目指南](docs/capstone.md)
- [分层文献地图](references/README.md)

## 项目状态

当前版本包含完整课程骨架、一维 Poisson 数据生成器、DeepONet、FNO、统一训练脚本和自动测试。后续里程碑包括二维 Darcy、时间依赖 Burgers 方程、PINO 训练和复杂几何案例。
