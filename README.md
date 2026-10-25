# 深度学习理论 · 课后作业

> 课程：深度学习基础  
> 内容：奇异值分解（SVD）过程与实例、贝叶斯方法建模应用、交叉熵的定义与推导  
> 提交格式：Markdown（GitHub 仓库）

---

## 1. 奇异值分解（SVD）的分解过程及实例

### 1.1 定义与结论

对任意 $m \times n$ 实矩阵 $A$（秩为 $r$），必存在分解

$A = U \Sigma V^{\mathsf{T}}$

其中：

- $U = [u_1, u_2, \dots, u_m] \in \mathbb{R}^{m \times m}$，由 $AA^{\mathsf{T}}$ 的单位正交特征向量组成，称为**左奇异向量**；
- $V = [v_1, v_2, \dots, v_n] \in \mathbb{R}^{n \times n}$，由 $A^{\mathsf{T}}A$ 的单位正交特征向量组成，称为**右奇异向量**；
- $\Sigma \in \mathbb{R}^{m \times n}$ 为对角矩阵，对角元 $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$，称为**奇异值**，且 $\sigma_i = \sqrt{\lambda_i(A^{\mathsf{T}}A)}$。

等价形式：$A = \sum_{i=1}^{r} \sigma_i u_i v_i^{\mathsf{T}}$，即 $A$ 是 $r$ 个秩一矩阵的加权和。

### 1.2 分解过程推导

**第一步：考察 $A^{\mathsf{T}}A$ 的特征结构。**

$A^{\mathsf{T}}A$ 是半正定对称矩阵（对任意 $x$ 有 $x^{\mathsf{T}}A^{\mathsf{T}}Ax = \|Ax\|^2 \ge 0$），故其特征值非负，记为 $\lambda_1 \ge \dots \ge \lambda_r > 0$，对应单位正交特征向量 $v_1, \dots, v_n$。定义奇异值

$\sigma_i = \sqrt{\lambda_i}, \quad i = 1, \dots, r$

**第二步：构造左奇异向量。**

对 $i = 1, \dots, r$，令

$u_i = \frac{1}{\sigma_i} A v_i$

验证其单位正交性（关键一步）：

$u_i^{\mathsf{T}} u_j = \frac{v_i^{\mathsf{T}} A^{\mathsf{T}} A v_j}{\sigma_i \sigma_j} = \frac{\lambda_j}{\sigma_i \sigma_j} v_i^{\mathsf{T}} v_j = \delta_{ij}$

故 $u_1, \dots, u_r$ 是 $\mathbb{R}^m$ 中的单位正交组，将其扩充为完整标准正交基即得 $U$。

**第三步：验证 $A = U\Sigma V^{\mathsf{T}}$。**

只需验证两边作用在 $V$ 的每一列上相等。对 $i \le r$，有 $A v_i = \sigma_i u_i = U \Sigma V^{\mathsf{T}} v_i$；对 $i > r$（零空间部分），两边作用结果均为 $0$。故等式成立。

### 1.3 实例：手工完成一个 2×2 矩阵的 SVD

设

$A = \begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix}$

**（1）计算 $A^{\mathsf{T}}A$ 并求特征值：**

$A^{\mathsf{T}}A = \begin{pmatrix} 25 & -15 \\ -15 & 25 \end{pmatrix}$

特征方程为 $(25-\lambda)^2 - 225 = 0$，解得 $\lambda_1 = 40$，$\lambda_2 = 10$，故奇异值

$\sigma_1 = 2\sqrt{10}, \qquad \sigma_2 = \sqrt{10}$

**（2）求右奇异向量 $V$：**

- 对 $\lambda_1 = 40$：由 $-15x - 15y = 0$ 得 $y = -x$，取 $v_1 = \dfrac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$
- 对 $\lambda_2 = 10$：由 $15x - 15y = 0$ 得 $y = x$，取 $v_2 = \dfrac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$

**（3）求左奇异向量 $U$：**

$u_1 = \frac{Av_1}{\sigma_1} = \frac{1}{2\sqrt{10}} \begin{pmatrix} 2\sqrt{2} \\ 4\sqrt{2} \end{pmatrix} = \frac{1}{\sqrt{5}}\begin{pmatrix} 1 \\ 2 \end{pmatrix}$

$u_2 = \frac{Av_2}{\sigma_2} = \frac{1}{\sqrt{10}} \begin{pmatrix} 2\sqrt{2} \\ -\sqrt{2} \end{pmatrix} = \frac{1}{\sqrt{5}}\begin{pmatrix} 2 \\ -1 \end{pmatrix}$

**（4）写出分解结果：**

$\underbrace{\begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix}}_{A} = \underbrace{\frac{1}{\sqrt{5}}\begin{pmatrix} 1 & 2 \\ 2 & -1 \end{pmatrix}}_{U} \, \underbrace{\begin{pmatrix} 2\sqrt{10} & 0 \\ 0 & \sqrt{10} \end{pmatrix}}_{\Sigma} \, \underbrace{\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}}_{V^{\mathsf{T}}}$

**（5）回代验证：**

$U\Sigma V^{\mathsf{T}} = \frac{1}{\sqrt{5}}\begin{pmatrix} 1 & 2 \\ 2 & -1 \end{pmatrix} \begin{pmatrix} 2\sqrt{5} & -2\sqrt{5} \\ \sqrt{5} & \sqrt{5} \end{pmatrix} = \begin{pmatrix} 4 & 0 \\ 3 & -5 \end{pmatrix} = A$

验证通过。

### 1.4 几何意义

SVD 揭示了一个线性变换 $x \mapsto Ax$ 的三步几何过程：**旋转/反射**（$V^{\mathsf{T}}$）→ **沿坐标轴缩放**（$\Sigma$，缩放因子即奇异值）→ **再旋转/反射**（$U$）。最大的奇异值 $\sigma_1$ 表示该变换的最大"拉伸倍数"，对应矩阵的谱范数 $\|A\|_2 = \sigma_1$。

---

## 2. 贝叶斯方法：原理与实际项目建模案例

### 2.1 贝叶斯推断的核心

贝叶斯推断把模型参数 $\theta$ 视为随机变量，用概率分布表达不确定性，核心是**贝叶斯定理**：

$p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}$

三要素：$p(\theta)$ 为**先验**（数据到来之前的经验），$p(D \mid \theta)$ 为**似然**（数据由参数生成的概率），$p(\theta \mid D)$ 为**后验**（看到数据之后的认知）。

与频率学派"参数是固定未知量"不同，贝叶斯学派的学习过程是**先验 → 数据 → 后验**的连续更新：每次获得新数据，就把当前的后验当作下一步的先验，实现知识的递推积累。

### 2.2 项目案例：课程项目"校园二手书交易平台"的转化率评估

**业务问题：** 团队为平台首页设计了新的推荐位（方案 B），想回答："新方案的点击转化率 $\theta$ 是否显著高于旧方案（约 5%）？是否值得全量上线？"

**为什么用贝叶斯而不是频率学派？** 灰度测试只有 200 名用户，样本小，直接用 $\hat{\theta} = k/n$ 点估计波动很大，无法给出"上线/不上线"的可靠决策。贝叶斯方法可以显式引入历史经验（先验），并直接输出概率化结论。

#### 数学建模过程

**第一步：建立似然模型。**

每名用户是否点击是伯努利试验，$n = 200$ 次独立试验中点击 $k$ 次：

$p(D \mid \theta) = \binom{n}{k} \theta^{k} (1-\theta)^{n-k} \propto \theta^{k} (1-\theta)^{n-k}$

**第二步：选择先验分布（共轭先验）。**

旧方案长期转化率约 5%，据此取先验 $\theta \sim \mathrm{Beta}(\alpha, \beta)$，其中 $\alpha = 5$，$\beta = 95$（先验均值 $\frac{5}{100} = 5\%$，相当于 100 人次的历史虚拟样本）：

$p(\theta) = \frac{1}{B(\alpha,\beta)} \, \theta^{\alpha-1}(1-\theta)^{\beta-1}$

**第三步：推导后验分布（共轭更新）。**

$p(\theta \mid D) \propto \theta^{k}(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} = \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}$

即后验仍是 Beta 分布，完成共轭更新：

$\theta \mid D \sim \mathrm{Beta}(\alpha + k,\ \beta + n - k)$

**第四步：代入数据计算。**

灰度测试结果为 $n = 200$ 人、$k = 16$ 人点击，后验为

$\theta \mid D \sim \mathrm{Beta}(21,\ 279)$

后验均值为 $\frac{21}{300} = 7\%$，较先验的 5% 明显上移。

**第五步：做出概率化决策（贝叶斯方法相对显著性检验的最大优势）。**

直接计算感兴趣事件的后验概率，即"新方案转化率超过旧方案基准 5%"的概率：

$P(\theta > 0.05 \mid D) = \int_{0.05}^{1} \frac{\theta^{20}(1-\theta)^{278}}{B(21,279)} \, d\theta \approx 0.90$

**决策结论：** 有 90% 的把握认为新方案优于旧方案，且后验均值 7% 相对基线提升 40%。结合业务成本判断（推荐位改造成本低），项目决策为：**扩大灰度至 50% 流量继续观察，后验 $\mathrm{Beta}(21, 279)$ 作为下一阶段的先验继续递推更新。**

### 2.3 案例小结

| 要素    | 本案例                                         |
| ----- | ------------------------------------------- |
| 似然    | 二项分布 $\mathrm{Binomial}(n,\theta)$          |
| 先验    | $\mathrm{Beta}(5, 95)$，来自历史转化率              |
| 后验    | $\mathrm{Beta}(\alpha+k,\ \beta+n-k)$，共轭更新  |
| 决策依据  | 后验概率 $P(\theta > 0.05 \mid D) \approx 0.90$ |
| 贝叶斯优势 | 小样本可用、结论可概率化、支持增量更新                         |

---

## 3. 交叉熵：定义、原理与应用推导

### 3.1 从信息熵到交叉熵

**信息熵**（Shannon, 1948）度量分布 $p$ 的不确定性：

$H(p) = -\sum_{x} p(x) \log p(x)$

**交叉熵**度量"用分布 $q$ 去编码真实分布 $p$"的平均编码长度：

$H(p, q) = -\sum_{x} p(x) \log q(x)$

注意 $H(p,q)$ 对 $p$ 和 $q$ **不对称**，一般 $H(p,q) \ne H(q,p)$。

### 3.2 与 KL 散度的关系（原理核心）

KL 散度度量两个分布的差异：

$D_{\mathrm{KL}}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$

展开即得三者关系：

$H(p, q) = \underbrace{D_{\mathrm{KL}}(p \| q)}_{\text{分布差异}} + \underbrace{H(p)}_{\text{常数项}}$

由于 $H(p)$ 不依赖 $q$，**最小化交叉熵等价于最小化 $D_{\mathrm{KL}}(p\|q)$**，即让预测分布 $q$ 逼近真实分布 $p$。这就是交叉熵能作为分类损失函数的理论依据。

另外由 Gibbs 不等式 $D_{\mathrm{KL}}(p\|q) \ge 0$ 可知 $H(p,q) \ge H(p)$，当且仅当 $q = p$ 时取等号——交叉熵有理论下界。

### 3.3 应用实例：Softmax + 交叉熵的完整推导

**场景：** 三分类问题（猫/狗/鸟），真实标签的 one-hot 分布为 $p = (1, 0, 0)$（真实类别为"猫"），网络输出 logits 为 $z = (2, 1, 0)$。

**第一步：Softmax 将 logits 归一化为预测分布。**

$q = \frac{1}{e^{2} + e^{1} + e^{0}} \left( e^{2}, e^{1}, e^{0} \right) \approx (0.665,\ 0.245,\ 0.090)$

**第二步：计算交叉熵损失。**

代入 $p$ 的 one-hot 结构，只有真实类别一项非零：

$L = H(p, q) = -\sum_{i=1}^{3} p_i \log q_i = -\log q_1 = -\log 0.665 \approx 0.408$

由此得到分类任务中常见的简化形式：

$L = -\log q_{y} \quad (y \text{ 为真实类别})$

**第三步：反向传播推导（交叉熵 + Softmax 的梯度为何简洁）。**

对 Softmax 求导有分段形式：

$\frac{\partial q_i}{\partial z_j} = q_i(\delta_{ij} - q_j)$

代入损失 $L = -\sum_i p_i \log q_i$：

$\frac{\partial L}{\partial z_j} = -\sum_i p_i \cdot \frac{1}{q_i} \cdot q_i(\delta_{ij} - q_j) = -\sum_i p_i(\delta_{ij} - q_j) = q_j - p_j$

即

$\frac{\partial L}{\partial z_j} = q_j - p_j$

**结果解读：** 梯度形式极其简洁——"预测概率 − 真实标签"。预测越准（$q_y \to 1$）梯度越小，学习自动减速；预测越离谱梯度越大，修正力度越强。这一性质使交叉熵成为分类任务的标准损失函数。

### 3.4 数值例子验证

| 预测分布 $q$（真实类别为猫）        | 交叉熵损失 $-\log q_1$ | 含义           |
| ----------------------- | ----------------- | ------------ |
| $(0.665, 0.245, 0.090)$ | $0.408$           | 预测较好，损失小     |
| $(0.50, 0.25, 0.25)$    | $0.693$           | 有区分度但不自信     |
| $(0.34, 0.33, 0.33)$    | $1.078$           | 接近瞎猜         |
| $(0.05, 0.90, 0.05)$    | $3.00$            | 自信地预测错误，损失极大 |

可见交叉熵同时惩罚"错误"和"过度自信的错误"，这正是分类模型训练所期望的行为。

---

## 参考文献

1. Golub, G. H., & Van Loan, C. F. *Matrix Computations*.（SVD 理论）
2. Murphy, K. P. *Machine Learning: A Probabilistic Perspective*.（贝叶斯建模与共轭先验）
3. Shannon, C. E. (1948). A Mathematical Theory of Communication.（信息熵与交叉熵）
4. Goodfellow, I., et al. *Deep Learning*, Chapter 3.（Softmax 与交叉熵梯度）
