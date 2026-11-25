"""语义别名：批量样条算子的主要实现见 :mod:`spline_kinematics`。"""

from __future__ import annotations

from spline_kinematics import *  # noqa: F403
import spline_kinematics as _core
from spline_kinematics_input import SAMPLE, load_record  # noqa: F401

__all__ = list(getattr(_core, "__all__", ())) + ["load_record", "SAMPLE"]

# 常见命名别名，方便不同测试/训练端入口。
SplineOperator = _core.BatchSplineOperator
BatchSplineFitter = _core.BatchSplineOperator
SplineBatchOperator = _core.BatchSplineOperator
SplineKinematicsOperator = _core.BatchSplineOperator
FittedSpline = _core.FitSnapshot
SplineFit = _core.FitSnapshot
TrajectoryFit = _core.FitSnapshot
FitResult = _core.BatchFitResult
SplineFitter = _core.BatchSplineOperator
BatchFitter = _core.BatchSplineOperator
RevisionManager = _core.RevisionStore

# 函数式命名别名。
fit_splines = _core.fit_batch
fit_spline_batch = _core.fit_batch
fit_trajectories = _core.fit_batch
query_splines = _core.query_batch
backward_splines = _core.backward_batch
