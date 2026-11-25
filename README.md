# 接触分段的非均匀样条张量与导数交接

`data/record.json` 是 接触分段的非均匀样条张量与导数交接 使用的本地合成交换样本。数组按 JSON 的下标顺序排列。本文的坐标、单位和枚举约定与该记录配套。记录中如果有 `scale` 或 `limits`，对应同一输入协议的部署规格。

时间单位为秒，p 为米，v 为米每秒。observations 每行是观测 ID、时间、类型、值和平方误差的权重。分段使用每段两端重结点的三次样条定义，internal_knots 只列内部结点。p-only 连接仅位置相等，p-v 还要求一阶导数相等。约束每行是时间、时间侧、类型和值。seed 的三个分量对应位置、一阶导、二阶导。形状记录为单关节案例，部署批次按关节分量给观测与约束。

## 读取样本

使用 Python 3.11 或 3.12，并在虚拟环境安装锁定依赖。读取入口返回原始记录。作为命令运行时，它会列出各顶层输入项目以及 NumPy 版本。

```sh
python -m pip install -r requirements.txt
python -m spline_kinematics_input
python -m unittest discover -s tests -v
```

## 批量拟合、查询与反向

算子主体在 `spline_kinematics.py`；`spline_operator.py` 等文件只是常用命名别名。命令行可直接查看样本两个修订版的查询和梯度：

```sh
python -m spline_kinematics data/record.json --backward-check
```

程序化用法：

```python
from spline_kinematics import BatchSplineOperator, QueryBudget
from spline_kinematics_input import load_record

record = load_record()
op = BatchSplineOperator(budget=QueryBudget.from_scale(record.get("scale")))
op.add_record(record, "teleop-0", fit_all_revisions=True)

old = op.get("teleop-0", revision=1)
left = old.query_one(0.58, "left")
right = old.query_one(0.58, "right")
print(left.p[0], left.v[0], left.a[0])
print(right.p[0], right.v[0], right.a[0])

# p-only 边界不保证 v/a 连续；auto 会给出 NaN 和 side 原因，query_one 抛 AmbiguousSideError。
ambiguous = old.query([[0.58, "auto"]])

# 上游对 p/v/a 的梯度一次性交回；返回值只包含该快照实际使用的观测和约束。
grads = left.backward(grad_position=1.0, grad_velocity=0.0, grad_acceleration=0.0)
# 也可直接使用记录中的 seed=[dL/dp,dL/dv,dL/da]：
seed_grads = left.vjp(record["seed"])
print(grads.observations, grads.constraints, seed_grads.ok)

new_snap, diff = op.revise_and_compare(
    "teleop-0", record["partitions"][1], new_revision=2, old_revision=1
)
```

约定如下：

- 每个接触区间有独立的 clamped 三次 B 样条，结点向量为 `[a,a,a,a, *internal_knots, b,b,b,b]`。
- 目标函数为 `sum_i w_i (q_i-y_i)^2 + regularization * sum_s integral (s_s''(t))^2 dt`；速度行按物理时间秒求导，内部时间尺度只改善数值缩放。
- `p-only` 强制界面位置相等；`p-v` 还强制速度相等。加速度在界面两侧均不自动视为同一值，因此查询恰在界面时应给 `side="left"/"right"`。
- 矛盾约束抛出的 `ContradictoryConstraintError` 带 `trajectory_id`、`joint`、`boundary` 和 `involved`。`fit_batch` 会记录单条轨迹失败并继续拟合批次中的其他独立轨迹。
- 正则化不能固定各段位置/斜率零空间时，`JointFit.status="unidentifiable"`；查询的 p/v/a 为 `NaN` 并带相同原因，反向返回 `ok=False` 和原因，不把系数默认填零。
- `QueryResult.trustworthy_components` 给出形状 `(n_query, 3)` 的当前正则化拟合可信标记；`data_certain` 进一步表示该分量是否只由正权重观测和精确约束（不依赖正则先验）确定。
- `FitSnapshot` 永久持有对应修订的观测、分界、约束和稀疏 KKT 分解；旧快照的反向梯度不会包含修订后才加入的观测。
- 拟合和反向按轨迹/关节建立稀疏 KKT 系统，查询只用 B 样条的局部非零基；`QueryBudget.scratch_bytes` 会把一组查询分块。代码不构造跨整批观测的稠密拟合矩阵。

测试读取交换样本，核对其中的张量尺寸、标识与数据约定。
