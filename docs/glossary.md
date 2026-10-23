# 术语表

- **Operator / 算子**：从一个函数空间到另一个函数空间的映射。
- **Solution operator / 解算子**：把 PDE 的系数、forcing、初边值条件等映射到解。
- **Optimizee**：不是本仓库主题；它属于 learned optimizer 文献。
- **Sensor**：输入函数被观测或采样的位置。
- **Query point**：希望得到输出函数值的位置。
- **Branch net**：DeepONet 中编码输入函数的网络。
- **Trunk net**：DeepONet 中编码输出坐标的网络。
- **Spectral convolution**：在变换域中完成的核乘法或卷积。
- **Modes**：FNO 中保留并学习的频率模态。
- **Discretization invariance**：模型参数化不绑定于一个固定离散维度。
- **Discretization convergence**：网格加密时离散模型趋近同一连续算子。
- **Resolution generalization**：训练和测试分辨率不同仍保持预测精度。
- **OOD**：测试输入分布偏离训练输入分布。
- **Rollout**：将模型预测递归地作为未来输入。
- **Physics-informed**：把方程、边界或守恒约束加入训练。
- **Many-query**：同一类方程或系统需要反复求解的应用场景。
