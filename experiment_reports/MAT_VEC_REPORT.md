test_matvec_accuracy.py - 新的测试文件，包含 15 个测试：

准确性测试
TestSymmetricIndex (2 tests)

对称索引覆盖 (136 entries)
对称索引对称性
TestSymmetricExpansion (2 tests)

Identity 矩阵展开
随机 SPD 矩阵展开保持对称性
TestBlockMatVec (4 tests)

Identity MatVec: z = I * r == r
Diagonal MatVec: 对角矩阵乘向量
随机 SPD MatVec: 与完整矩阵展开对比
逆矩阵 MatVec: 验证 A * (A^{-1} * r) ≈ r
TestTaichiMatVec (3 tests)

Taichi Identity MatVec (f32 vs NumPy f64)
Taichi 随机 SPD MatVec
数值精度对比 (f32 vs f64, max 误差 ~1.6e-07)
TestFullSolverMatVec (4 tests)

apply() 无 NaN
g^T z > 0 有效性
Block 0 local solve 准确性 (相对误差 ~3.5e-08)
求解器变体一致性 (conflict_free, parallel, full_solve)
性能基准
变体	平均时间	g^Tz > 0
full_solve (default)	2.53ms	Yes
conflict_free	2.28ms	Yes
parallel	2.32ms	Yes
diagonal_only	0.37ms	Yes
使用方法

# 运行所有测试
cd /root/PNCG_IPC/unittest/tests
python test_matvec_accuracy.py -v

# 仅运行性能基准
python test_matvec_accuracy.py --benchmark