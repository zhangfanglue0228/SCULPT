## Method
### 1. 核心动机 (Motivation)

SVDLoRA 虽然利用了 SVD 的物理特性，但它假设所有层对于下游任务的重要性是一样的（例如所有层都取 $r=16$）。
然而根据 Eckart-Young 定理，奇异值的大小决定了信息量的多少。不同层权重的奇异值分布（Spectral Distribution）差异很大：
- 有的层“谱能量”集中在前几个奇异值，适合低秩。
- 有的层“谱能量”分布长尾，需要更高秩。
**AdaSVD 的目标**： 利用奇异值作为天然的“重要性指标”，在微调过程中动态地修剪或增强特定维度的谱能量。
---
### 2. 参数化与前向传播 (Parametrization)
在 SVDLoRA 中，权重更新为 $\Delta W = U_r \Sigma_r V_r^T$。
在 AdaSVD 中，我们将初始秩设定为一个较大的“搜索空间” $r_{init}$ (例如 $r_{init} = 2r$ 或 $3r$)，并引入一个可学习的谱门控向量 (Spectral Gate)。
#### 定义参数：
- $U \in \mathbb{R}^{m \times r_{init}}$, $V \in \mathbb{R}^{n \times r_{init}}$: 初始化为 SVD 的前 $r_{init}$ 个奇异向量，并在训练中保持正交约束 6。
- $\mathbf{s} \in \mathbb{R}^{r_{init}}$: **可学习的奇异值向量**，初始化为 SVD 的奇异值 $\Sigma_{init}$。这就是我们要“修剪”的核心对象。
#### 前向传播公式：
为了实现自动秩选择，我们将 $\mathbf{s}$ 分解为“基础谱能量”和“重要性掩码”：
$$\Delta W = U \cdot \text{diag}(\mathbf{s} \odot \mathbf{m}) \cdot V^T$$
其中：
- $\mathbf{s}$ 是训练参数，代表奇异值的大小。
- $\mathbf{m} \in \{0, 1\}^{r_{init}}$ 是二值掩码向量（Binary Mask），由重要性评分决定。
- $\odot$ 是哈达玛积（逐元素相乘）。
---