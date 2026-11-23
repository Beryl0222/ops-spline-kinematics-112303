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

测试读取交换样本，核对其中的张量尺寸、标识与数据约定。
