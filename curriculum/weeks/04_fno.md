# 第 4 周：Fourier Neural Operator

## 核心问题

如何在函数空间中高效表示具有全局作用的积分核？

## 学习目标

- 复习卷积定理、FFT、频谱截断和 Nyquist 频率；
- 理解 lifting、谱卷积、局部线性支路和 projection；
- 理解模态数、通道宽度、padding 和边界效应；
- 区分参数共享、离散化收敛和分辨率泛化。

## 实验

逐行阅读 `SpectralConv1D`；训练 `FNO1D`；比较 4、8、12、20 个模态，绘制频谱误差。

## 作业

关闭坐标通道或 padding，观察误差变化；解释非周期边界为什么可能出现 wrap-around artifact。

## 验收

能够从输入张量写出 FFT、频域乘法和逆 FFT 的完整尺寸变化，并用实验解释模态数的偏差—方差权衡。
