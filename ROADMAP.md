# Roadmap

## v0.1：从概念到首个可信实验

- 12 周课程与精读文献地图
- 1D 变系数 Poisson 数据生成
- DeepONet/FNO 同数据、同预算对比
- CPU smoke test、测试与 CI

## v0.2：跨分辨率与二维问题

- 训练/测试网格分辨率矩阵
- 2D Darcy flow
- H1、守恒残差和不确定性指标
- NeuralOperator 官方实现对照

## v0.3：时间与物理约束

- Burgers/Navier–Stokes 时间滚动
- teacher forcing、autoregressive rollout 与误差累积
- Physics-informed DeepONet/PINO

## v1.0：真实问题闭环

- 明确业务或科研问题的算子定义
- 数据生成与数值求解器校验
- 至少两个强基线和消融实验
- 可复现实验报告、局限与失败案例
