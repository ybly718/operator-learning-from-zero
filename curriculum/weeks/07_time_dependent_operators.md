# 第 7 周：时间依赖算子与长期 rollout

## 核心问题

为什么很小的一步预测误差会在长期模拟中失控？

## 学习目标

- 理解流映射、半群性质和自回归算子；
- 区分 teacher forcing、closed-loop rollout 和全时空预测；
- 理解 exposure bias、稳定性和守恒误差；
- 学会按时间和频率报告误差。

## 实验

把任务扩展到一维 Burgers 或其他简单演化方程。比较 one-step loss 与 10、50、100 步 rollout error。

## 作业

比较单步训练、逐步增加 rollout horizon 和 pushforward-style 训练；报告显存、训练时间与稳定性的权衡。

## 验收

不得以单步误差替代长期表现；必须展示至少一个失败轨迹并解释误差累积机制。
