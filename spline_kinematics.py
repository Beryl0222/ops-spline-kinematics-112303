"""接触分段、非均匀结点的批量三次 B 样条拟合、查询与导数交接。

本模块只依赖 NumPy/SciPy。批次中的轨迹/关节分别建立稀疏 KKT 系统，绝不把
整批观测拼成一个稠密最小二乘矩阵。旧修订版返回的 :class:`FitSnapshot` 持有
当时的观测、分界、约束和因子分解，因此可以重复查询和反向传播。
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline
from scipy.sparse.linalg import lsqr, splu

try:  # Python 3.11
    from enum import StrEnum
except ImportError:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):
        pass


__all__ = [
    "AmbiguousSideError",
    "BatchBackwardResult",
    "BatchFitResult",
    "BatchQueryResult",
    "BatchSplineOperator",
    "BudgetExceededError",
    "ContradictoryConstraintError",
    "HardConstraint",
    "Observation",
    "PartitionSpec",
    "QueryBudget",
    "QueryRequest",
    "QueryResult",
    "RevisionDiff",
    "RevisionStore",
    "SplineKinematicsError",
    "TrajectorySpec",
    "UnidentifiableError",
    "backward_batch",
    "fit_batch",
    "fit_trajectory",
    "load_operator_from_record",
    "query_batch",
]


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


class SplineKinematicsError(RuntimeError):
    """本模块所有可预期错误的基类。"""


class ContractError(ValueError, SplineKinematicsError):
    """输入交换契约不合法。"""


class ContradictoryConstraintError(ContractError):
    """精确约束或界面连接条件互相矛盾。"""

    def __init__(
        self,
        message: str,
        *,
        trajectory_id: Any = None,
        joint: int | None = None,
        boundary: float | int | None = None,
        boundary_index: int | None = None,
        involved: Sequence[Any] = (),
    ) -> None:
        super().__init__(message)
        self.trajectory_id = trajectory_id
        self.joint = joint
        self.boundary = boundary
        self.boundary_index = boundary_index
        self.involved = list(involved)


class AmbiguousSideError(ContractError):
    """恰在速度/加速度可能断开的边界上且没有声明侧别。"""

    def __init__(
        self,
        message: str,
        *,
        trajectory_id: Any = None,
        joint: int | None = None,
        time: float | None = None,
        boundary: float | int | None = None,
        choices: Sequence[str] = ("left", "right"),
        values: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.trajectory_id = trajectory_id
        self.joint = joint
        self.time = time
        self.boundary = boundary if boundary is not None else time
        self.choices = tuple(choices)
        self.values = dict(values or {})


class UnidentifiableError(SplineKinematicsError):
    """观测和精确约束不足以唯一确定该段。"""

    def __init__(
        self,
        message: str,
        *,
        trajectory_id: Any = None,
        joint: int | None = None,
        revision: Any = None,
        snapshot_id: str | None = None,
        sections: Sequence[int] = (),
        reasons: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.trajectory_id = trajectory_id
        self.joint = joint
        self.revision = revision
        self.snapshot_id = snapshot_id
        self.sections = tuple(sections)
        self.reasons = tuple(reasons)


class BudgetExceededError(SplineKinematicsError):
    """请求超过显式工作预算。"""


class SnapshotNotFoundError(KeyError, SplineKinematicsError):
    """请求了不存在的旧修订快照。"""


class ObservationNotInSnapshotError(KeyError, SplineKinematicsError):
    """梯度或回查触及了不属于该拟合快照的观测。"""


# ---------------------------------------------------------------------------
# 输入契约
# ---------------------------------------------------------------------------

_KIND_ALIASES = {
    "p": "p",
    "pos": "p",
    "position": "p",
    "x": "p",
    "v": "v",
    "vel": "v",
    "velocity": "v",
    "dx": "v",
}
_SIDE_ALIASES = {
    "left": "left",
    "l": "left",
    "-": "left",
    "before": "left",
    "right": "right",
    "r": "right",
    "+": "right",
    "after": "right",
    "auto": "auto",
    "none": "auto",
    "mid": "auto",
    "interior": "auto",
    None: "auto",
}
_MODE_ALIASES = {
    "p-only": "p-only",
    "ponly": "p-only",
    "p_only": "p-only",
    "position-only": "p-only",
    "position_only": "p-only",
    "p": "p-only",
    "0": "p-only",
    "p-v": "p-v",
    "pv": "p-v",
    "p_v": "p-v",
    "position-velocity": "p-v",
    "continuous": "p-v",
    "both": "p-v",
    "1": "p-v",
    "none": "none",
    "disconnected": "none",
    "free": "none",
    "": "none",
    "v-only": "v-only",
    "v_only": "v-only",
}


def _as_float(x: Any, what: str) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{what} 不是数值：{x!r}") from exc
    if not math.isfinite(y):
        raise ContractError(f"{what} 必须有限：{x!r}")
    return y


def _norm_kind(kind: Any) -> str:
    key = str(kind).strip().lower()
    if key not in _KIND_ALIASES:
        raise ContractError(f"未知观测/约束类型 {kind!r}；只支持 p/position 与 v/velocity")
    return _KIND_ALIASES[key]


def _norm_side(side: Any) -> str:
    if side is None:
        return "auto"
    key = str(side).strip().lower()
    if key not in _SIDE_ALIASES:
        raise ContractError(f"未知时间侧 {side!r}；支持 left/right/auto")
    return _SIDE_ALIASES[key]


def _norm_mode(mode: Any) -> str:
    key = str(mode).strip().lower()
    if key not in _MODE_ALIASES:
        raise ContractError(f"未知界面连接模式 {mode!r}；支持 p-only、p-v、none、v-only")
    return _MODE_ALIASES[key]


@dataclass(frozen=True, slots=True)
class Observation:
    id: Any
    t: float
    kind: str
    value: float
    weight: float = 1.0
    joint: int = 0
    side: str = "auto"

    def __post_init__(self) -> None:
        kind = _norm_kind(self.kind)
        side = _norm_side(self.side)
        t = _as_float(self.t, "观测时间")
        value = _as_float(self.value, "观测值")
        weight = _as_float(self.weight, "观测权重")
        if weight < 0:
            raise ContractError(f"观测 {self.id!r} 的权重不能为负：{weight}")
        if not isinstance(self.joint, (int, np.integer)):
            raise ContractError(f"观测 {self.id!r} 的 joint 必须是整数")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "joint", int(self.joint))

    @staticmethod
    def from_raw(row: Any, default_joint: int = 0) -> "Observation":
        if isinstance(row, Observation):
            return row
        if isinstance(row, Mapping):
            oid = row.get("id", row.get("observation_id", row.get("name")))
            t = row.get("time", row.get("t"))
            kind = row.get("type", row.get("kind", row.get("derivative")))
            value = row.get("value", row.get("y", row.get("target")))
            weight = row.get("weight", row.get("w", 1.0))
            joint = row.get("joint", row.get("coordinate", row.get("j", default_joint)))
            side = row.get("side", row.get("time_side", "auto"))
            return Observation(oid, t, kind, value, weight, joint, side)  # type: ignore[arg-type]
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ContractError(f"观测行必须是数组或对象：{row!r}")
        # [id, joint, t, kind, value, weight]
        if (
            len(row) >= 6
            and isinstance(row[1], (int, np.integer))
            and isinstance(row[2], (int, float, np.floating))
            and isinstance(row[3], str)
        ):
            oid, joint, t, kind, value, weight = row[:6]
            side = row[6] if len(row) > 6 else "auto"
            return Observation(oid, t, kind, value, weight, int(joint), side)  # type: ignore[arg-type]
        if len(row) >= 5:
            oid, t, kind, value, weight = row[:5]
            side = row[5] if len(row) > 5 else "auto"
            joint = row[6] if len(row) > 6 else default_joint
            return Observation(oid, t, kind, value, weight, int(joint), side)  # type: ignore[arg-type]
        raise ContractError(f"观测行至少需要 [id,time,type,value,weight]：{row!r}")


@dataclass(frozen=True, slots=True)
class HardConstraint:
    t: float
    side: str
    kind: str
    value: float
    joint: int = 0
    id: Any = None

    def __post_init__(self) -> None:
        t = _as_float(self.t, "约束时间")
        value = _as_float(self.value, "约束值")
        side = _norm_side(self.side)
        kind = _norm_kind(self.kind)
        if not isinstance(self.joint, (int, np.integer)):
            raise ContractError("约束 joint 必须是整数")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "joint", int(self.joint))

    @staticmethod
    def from_raw(row: Any, index: int = 0, default_joint: int = 0) -> "HardConstraint":
        if isinstance(row, HardConstraint):
            return row
        cid = index
        if isinstance(row, Mapping):
            cid = row.get("id", row.get("constraint_id", index))
            return HardConstraint(
                row.get("time", row.get("t")),
                row.get("side", row.get("time_side", "auto")),
                row.get("type", row.get("kind")),
                row.get("value", row.get("target")),
                row.get("joint", row.get("coordinate", row.get("j", default_joint))),
                cid,
            )
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ContractError(f"约束行必须是数组或对象：{row!r}")
        # [joint, time, side, type, value]
        if (
            len(row) == 5
            and isinstance(row[0], (int, np.integer))
            and isinstance(row[1], (int, float, np.floating))
            and isinstance(row[2], str)
        ):
            joint, t, side, kind, value = row
            return HardConstraint(t, side, kind, value, int(joint), cid)
        if len(row) >= 4:
            t, side, kind, value = row[:4]
            joint = row[4] if len(row) > 4 and isinstance(row[4], (int, np.integer)) else default_joint
            cid = row[5] if len(row) > 5 else index
            return HardConstraint(t, side, kind, value, int(joint), cid)
        raise ContractError(f"约束行至少需要 [time,side,type,value]：{row!r}")


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    breaks: tuple[float, ...]
    internal_knots: tuple[tuple[float, ...], ...]
    interfaces: tuple[str, ...]
    rev: Any = None

    def __post_init__(self) -> None:
        breaks = tuple(_as_float(x, "分界时间") for x in self.breaks)
        if len(breaks) < 2:
            raise ContractError("partition.breaks 至少需要起点和终点")
        if any(b <= a for a, b in zip(breaks[:-1], breaks[1:])):
            raise ContractError(f"breaks 必须严格递增：{breaks}")
        s = len(breaks) - 1
        raw_knots = self.internal_knots
        if raw_knots is None:
            knots: tuple[tuple[float, ...], ...] = tuple(() for _ in range(s))
        elif len(raw_knots) == s and all(isinstance(x, Sequence) for x in raw_knots):
            knots = tuple(tuple(_as_float(x, "内部结点") for x in section) for section in raw_knots)  # type: ignore[arg-type]
        elif s == 1 and isinstance(raw_knots, Sequence) and all(
            isinstance(x, (int, float, np.floating)) for x in raw_knots
        ):
            knots = (tuple(_as_float(x, "内部结点") for x in raw_knots),)  # type: ignore[arg-type]
        else:
            raise ContractError("internal_knots 必须按区段给出，长度为 len(breaks)-1")
        for si, ((a, b), ks) in enumerate(zip(zip(breaks[:-1], breaks[1:]), knots)):
            if any(k <= a or k >= b for k in ks):
                raise ContractError(f"第 {si} 段内部结点必须位于 ({a}, {b})：{ks}")
            if any(nxt < cur for cur, nxt in zip(ks[:-1], ks[1:])):
                raise ContractError(f"第 {si} 段内部结点必须非递减：{ks}")
        raw_if = tuple(self.interfaces) if self.interfaces is not None else tuple("p-only" for _ in range(s - 1))
        if len(raw_if) != s - 1:
            raise ContractError(f"interfaces 长度应为 {s - 1}，实际 {len(raw_if)}")
        interfaces = tuple(_norm_mode(x) for x in raw_if)
        object.__setattr__(self, "breaks", breaks)
        object.__setattr__(self, "internal_knots", knots)
        object.__setattr__(self, "interfaces", interfaces)

    @staticmethod
    def from_raw(raw: "PartitionSpec | Mapping[str, Any]", rev: Any = None) -> "PartitionSpec":
        if isinstance(raw, PartitionSpec):
            return raw if rev is None or raw.rev == rev else replace(raw, rev=rev)
        if not isinstance(raw, Mapping):
            raise ContractError(f"partition 必须是对象或 PartitionSpec：{raw!r}")
        use_rev = raw.get("rev", raw.get("revision", rev))
        return PartitionSpec(
            raw["breaks"],
            raw.get("internal_knots", raw.get("knots", ())),
            raw.get("interfaces", ()),
            use_rev,
        )

    @property
    def n_sections(self) -> int:
        return len(self.breaks) - 1

    def p_equal_at(self, boundary_index: int) -> bool:
        """boundary_index 是 breaks 的内部下标 1..S-1。"""
        mode = self.interfaces[boundary_index - 1]
        return mode in ("p-only", "p-v")

    def v_equal_at(self, boundary_index: int) -> bool:
        mode = self.interfaces[boundary_index - 1]
        return mode == "p-v"


@dataclass(frozen=True, slots=True)
class QueryRequest:
    time: float
    side: str = "auto"
    joint: int = 0
    id: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _as_float(self.time, "查询时间"))
        object.__setattr__(self, "side", _norm_side(self.side))
        object.__setattr__(self, "joint", int(self.joint))

    @staticmethod
    def from_raw(row: Any, index: int = 0, default_joint: int = 0) -> "QueryRequest":
        if isinstance(row, QueryRequest):
            return row
        if isinstance(row, Mapping):
            return QueryRequest(
                row.get("time", row.get("t")),
                row.get("side", "auto"),
                row.get("joint", row.get("coordinate", default_joint)),
                row.get("id", index),
            )
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
            if len(row) >= 3 and isinstance(row[0], (int, np.integer)) and isinstance(row[1], (int, float, np.floating)):
                # [joint, time, side]
                return QueryRequest(row[1], row[2], int(row[0]), index)
            if len(row) >= 2:
                t, side = row[0], row[1]
                joint = row[2] if len(row) > 2 and isinstance(row[2], (int, np.integer)) else default_joint
                qid = row[3] if len(row) > 3 else index
                return QueryRequest(t, side, int(joint), qid)
        raise ContractError(f"查询行应为 [time, side] 或对象：{row!r}")


@dataclass(frozen=True, slots=True)
class TrajectorySpec:
    trajectory_id: Any
    observations: tuple[Observation, ...]
    partition: PartitionSpec
    constraints: tuple[HardConstraint, ...] = ()
    regularization: float = 1e-3
    time_scale: float | str | None = "auto"
    n_joints: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        obs = tuple(Observation.from_raw(x) for x in self.observations)
        cons = tuple(HardConstraint.from_raw(x, i) for i, x in enumerate(self.constraints))
        if not isinstance(self.partition, PartitionSpec):
            raise ContractError("partition 必须是 PartitionSpec")
        reg = _as_float(self.regularization, "regularization")
        if reg < 0:
            raise ContractError("regularization 不能为负")
        joints = self.n_joints
        if joints is None:
            joints = max(
                [0] + [o.joint for o in obs] + [c.joint for c in cons]
            ) + 1
        if joints < 1:
            raise ContractError("n_joints 必须至少为 1")
        object.__setattr__(self, "observations", obs)
        object.__setattr__(self, "constraints", cons)
        object.__setattr__(self, "regularization", reg)
        object.__setattr__(self, "n_joints", int(joints))

    @staticmethod
    def from_record(
        record: Mapping[str, Any],
        *,
        trajectory_id: Any = "record",
        revision: Any = None,
        partition: Mapping[str, Any] | PartitionSpec | None = None,
        observations: Sequence[Any] | None = None,
        constraints: Sequence[Any] | None = None,
    ) -> "TrajectorySpec":
        if partition is None:
            partitions = record.get("partitions")
            if isinstance(partitions, Sequence) and partitions:
                if revision is None:
                    partition_raw = partitions[-1]
                else:
                    matches = [p for p in partitions if str(p.get("rev")) == str(revision)]
                    if not matches:
                        avail = [p.get("rev") for p in partitions]
                        raise SnapshotNotFoundError(f"记录中没有修订 {revision!r}，可用 {avail}")
                    partition_raw = matches[0]
            elif record.get("breaks") is not None:
                partition_raw = record
            else:
                raise ContractError("记录缺少 partitions 或 breaks")
        else:
            partition_raw = partition
        part = PartitionSpec.from_raw(partition_raw, revision)
        obs_raw = record.get("observations", ()) if observations is None else observations
        con_raw = record.get("hard_constraints", record.get("constraints", ())) if constraints is None else constraints
        reg = record.get("regularization", record.get("reg", 1e-3))
        time_scale = record.get("time_scale", "auto")
        return TrajectorySpec(
            trajectory_id,
            obs_raw,
            part,
            con_raw,
            reg,
            time_scale,
            record.get("n_joints", record.get("joints")),
            {k: v for k, v in record.items() if k not in {"observations", "partitions", "hard_constraints", "constraints"}},
        )

    def canonical_hash(self) -> str:
        payload = {
            "id": self.trajectory_id,
            "observations": [
                [o.id, o.t, o.side, o.kind, o.value, o.weight, o.joint] for o in self.observations
            ],
            "partition": {
                "rev": self.partition.rev,
                "breaks": list(self.partition.breaks),
                "internal_knots": [list(x) for x in self.partition.internal_knots],
                "interfaces": list(self.partition.interfaces),
            },
            "constraints": [
                [c.id, c.t, c.side, c.kind, c.value, c.joint] for c in self.constraints
            ],
            "regularization": self.regularization,
            "time_scale": self.time_scale,
            "n_joints": self.n_joints,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 局部三次 B 样条
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SectionBasis:
    index: int
    a: float
    b: float
    physical_knots: np.ndarray
    u_knots: np.ndarray
    tau: float
    t_ref: float
    n: int
    d1: sp.csr_matrix
    d21: sp.csr_matrix

    @staticmethod
    def build(index: int, a: float, b: float, internal: Sequence[float], time_scale: float | str | None) -> "_SectionBasis":
        physical = np.asarray([a, a, a, a, *internal, b, b, b, b], dtype=float)
        if time_scale in (None, "auto"):
            tau = max(float(b - a), 1e-12)
        else:
            tau = _as_float(time_scale, "time_scale")
            if tau <= 0:
                raise ContractError("time_scale 必须为正数或 'auto'")
        t_ref = a
        u = (physical - t_ref) / tau
        n = len(physical) - 4
        d1 = _derivative_matrix(u, 3)
        d2 = _derivative_matrix(u[1:-1], 2)
        d21 = (d2 @ d1).tocsr()
        return _SectionBasis(index, a, b, physical, u, tau, t_ref, n, d1, d21)

    def to_u(self, t: np.ndarray | Sequence[float]) -> np.ndarray:
        x = (np.asarray(t, dtype=float) - self.t_ref) / self.tau
        return np.clip(x, self.u_knots[0], self.u_knots[-1])

    def design(self, times: np.ndarray | Sequence[float], derivative: int = 0) -> sp.csr_matrix:
        if derivative not in (0, 1, 2):
            raise ContractError("本算子只直接提供 0/1/2 阶导数")
        t = np.asarray(times, dtype=float)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        x = self.to_u(t)
        if derivative == 0:
            mat = BSpline.design_matrix(x, self.u_knots, 3).tocsr()
        elif derivative == 1:
            base = BSpline.design_matrix(x, self.u_knots[1:-1], 2).tocsr()
            mat = (base @ self.d1) / self.tau
        else:
            base = BSpline.design_matrix(x, self.u_knots[2:-2], 1).tocsr()
            mat = (base @ self.d21) / (self.tau**2)
        if scalar:
            return mat.tocsr()
        return mat.tocsr()

    def regularization_matrix(self) -> sp.csr_matrix:
        """lambda 之前的 physical-time 加速度平方积分矩阵。"""
        knots = self.u_knots
        rs = sp.csr_matrix((self.n, self.n), dtype=float)
        unique = np.unique(knots)
        qx, qw = np.polynomial.legendre.leggauss(4)
        for left, right in zip(unique[:-1], unique[1:]):
            width = float(right - left)
            if width <= 0:
                continue
            x = 0.5 * (left + right) + 0.5 * width * qx
            # 正则项是二阶导，不乘 physical 的 tau^-2；最后统一 tau^-3。
            base = BSpline.design_matrix(x, knots[2:-2], 1).tocsr()
            acc_u = base @ self.d21
            rs = rs + (0.5 * width) * (acc_u.T @ sp.diags(qw) @ acc_u)
        return (rs / (self.tau**3)).tocsr()


def _derivative_matrix(knots: np.ndarray, degree: int) -> sp.csr_matrix:
    """B_{i,p} 系数 -> 一阶导 B_{i,p-1} 系数。"""
    n = len(knots) - degree - 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i in range(n - 1):
        den = knots[i + degree + 1] - knots[i + 1]
        if den != 0.0:
            rows.extend((i, i))
            cols.extend((i, i + 1))
            scale = degree / den
            data.extend((-scale, scale))
    return sp.csr_matrix((data, (rows, cols)), shape=(n - 1, n))


# ---------------------------------------------------------------------------
# 定位与工作预算
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryBudget:
    scratch_bytes: int = 12 * 1024 * 1024
    bytes_per_query_row: int = 768
    max_total_basis_evaluations: int | None = None

    def chunk_size(self) -> int:
        if self.scratch_bytes <= 0:
            return 1
        return max(1, int(self.scratch_bytes // max(1, self.bytes_per_query_row)))

    @staticmethod
    def from_scale(scale: Mapping[str, Any] | None) -> "QueryBudget":
        if not scale:
            return QueryBudget()
        return QueryBudget(
            scratch_bytes=int(scale.get("scratch_bytes", 12 * 1024 * 1024)),
        )


def _time_tol(breaks: Sequence[float], t: float, tol: float = 1e-9) -> float:
    return max(tol, 1e-10 * max(1.0, abs(t), *(abs(x) for x in breaks)))


def locate_time(
    partition: PartitionSpec,
    t: float,
    side: str = "auto",
    *,
    time_tolerance: float = 1e-9,
) -> tuple[int, str, int | None]:
    """返回 (section, 实际侧别, break_index)；break_index 为 None 表示区间内部。"""
    side = _norm_side(side)
    breaks = partition.breaks
    tol = _time_tol(breaks, t, time_tolerance)
    distances = [abs(x - t) for x in breaks]
    bidx = int(np.argmin(distances))
    if distances[bidx] <= tol:
        if bidx == 0:
            if side == "left":
                raise ContractError(f"时间 {t} 之前超出轨迹起点 {breaks[0]}")
            return 0, "right", 0
        if bidx == len(breaks) - 1:
            if side == "right":
                raise ContractError(f"时间 {t} 之后超出轨迹终点 {breaks[-1]}")
            return partition.n_sections - 1, "left", bidx
        if side == "left":
            return bidx - 1, "left", bidx
        if side == "right":
            return bidx, "right", bidx
        return bidx, "auto", bidx  # 调用方在需要单侧行时决定
    if t < breaks[0] - tol or t > breaks[-1] + tol:
        raise ContractError(f"查询/约束时间 {t} 超出轨迹区间 [{breaks[0]}, {breaks[-1]}]")
    s = int(np.searchsorted(np.asarray(breaks), t, side="right") - 1)
    s = min(max(s, 0), partition.n_sections - 1)
    return s, "interior", None


# ---------------------------------------------------------------------------
# 拟合结果
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class JointFit:
    trajectory_id: Any
    joint: int
    revision: Any
    snapshot_id: str
    spec: TrajectorySpec
    sections: tuple[_SectionBasis, ...]
    offsets: np.ndarray
    n_coefficients: int
    c: np.ndarray | None = None
    multipliers: np.ndarray | None = None
    kkt: sp.csc_matrix | None = None
    lu: Any = None
    selected_constraint_rows: np.ndarray | None = None
    a_all: sp.csr_matrix | None = None
    b_all: np.ndarray | None = None
    constraint_sources: list[dict[str, Any]] = field(default_factory=list)
    b_obs: sp.csr_matrix | None = None
    w_obs: np.ndarray | None = None
    y_obs: np.ndarray | None = None
    g_obs: sp.csr_matrix | None = None
    obs_rows: list[int] = field(default_factory=list)
    user_constraint_rows: list[int] = field(default_factory=list)
    status: str = "fitted"
    identifiable: bool = True
    reasons: tuple[str, ...] = ()
    section_reasons: dict[int, tuple[str, ...]] = field(default_factory=dict)
    sparse_nnz: int = 0
    contradiction: ContradictoryConstraintError | None = None
    unidentifiable: UnidentifiableError | None = None
    minimum_norm_candidate: np.ndarray | None = None
    certainty_cache: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "fitted" and self.c is not None and np.all(np.isfinite(self.c))


@dataclass(slots=True)
class QueryResult:
    trajectory_id: Any
    snapshot_id: str
    revision: Any
    requests: tuple[QueryRequest, ...]
    times: np.ndarray
    joints: np.ndarray
    sides_requested: tuple[str, ...]
    sides_resolved: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    data_certain: np.ndarray
    side_certain: np.ndarray
    side_component_certain: np.ndarray | None = None
    fit_identifiable: bool = True
    reasons: tuple[list[str], ...] = ()
    snapshot: "FitSnapshot | None" = field(default=None, repr=False)
    q_matrix: sp.csr_matrix | None = field(default=None, repr=False)
    q_row_offsets: np.ndarray | None = field(default=None, repr=False)
    joint_fit_ids: np.ndarray | None = field(default=None, repr=False)
    chunk_sizes: tuple[int, ...] = ()
    basis_evaluations: int = 0

    @property
    def ok(self) -> bool:
        return self.fit_identifiable and bool(np.all(self.side_certain)) and not any(self.reasons)

    @property
    def trustworthy_components(self) -> np.ndarray:
        """形状 (n_query, 3)：正则化拟合唯一且侧别无歧义。"""
        side = self.side_component_certain
        if side is None:
            side = np.broadcast_to(self.side_certain[:, None], (len(self.requests), 3))
        return np.broadcast_to(self.fit_identifiable, side.shape) & side

    @property
    def trustworthy(self) -> np.ndarray:
        """每个查询的 p/v/a 是否都可作为可信监督；逐分量见 data_certain。"""
        return np.all(self.trustworthy_components, axis=1)

    @property
    def strictly_data_identified_components(self) -> np.ndarray:
        """不依赖正则先验、仅由正权重观测和精确约束张成的分量。"""
        return self.trustworthy_components & self.data_certain

    @property
    def p(self) -> np.ndarray:
        return self._single_joint_squeeze(self.position)

    @property
    def v(self) -> np.ndarray:
        return self._single_joint_squeeze(self.velocity)

    @property
    def a(self) -> np.ndarray:
        return self._single_joint_squeeze(self.acceleration)

    def _single_joint_squeeze(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr[:, 0]
        return arr

    def values_at(self, index: int) -> dict[str, float | None]:
        def val(arr: np.ndarray) -> float | None:
            x = arr[index, 0] if arr.ndim == 2 else arr[index]
            return None if not np.isfinite(x) else float(x)

        return {"p": val(self.position), "v": val(self.velocity), "a": val(self.acceleration)}

    def raise_if_failed(self) -> None:
        if not self.fit_identifiable:
            for jf in self.snapshot.joint_fits.values() if self.snapshot else ():
                if jf.unidentifiable:
                    raise jf.unidentifiable
        for i, reason_list in enumerate(self.reasons):
            for reason in reason_list:
                if "超出" in reason or "未知时间侧" in reason:
                    raise ContractError(reason)
        if self.side_component_certain is not None:
            for i, req in enumerate(self.requests):
                bad = [name for k, name in enumerate(("p", "v", "a")) if not self.side_component_certain[i, k]]
                if bad:
                    raise AmbiguousSideError(
                        f"查询时间 {req.time:g} 的 {','.join(bad)} 在边界上需要显式 side=left/right",
                        trajectory_id=self.trajectory_id,
                        joint=req.joint,
                        time=req.time,
                    )
        for i, reason in enumerate(self.reasons):
            if reason:
                raise UnidentifiableError(
                    f"查询第 {i} 项不可唯一确定：{'; '.join(reason)}",
                    trajectory_id=self.trajectory_id,
                    revision=self.revision,
                    snapshot_id=self.snapshot_id,
                    reasons=reason,
                )

    def backward(
        self,
        grad_position: Any = None,
        grad_velocity: Any = None,
        grad_acceleration: Any = None,
        *,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> "BackwardResult":
        if self.snapshot is None:
            raise SplineKinematicsError("QueryResult 未持有快照，无法反向传播")
        gp, gv, ga = _resolve_gradient_args(
            len(self.requests), grad_position, grad_velocity, grad_acceleration, kwargs
        )
        return self.snapshot.backward(
            self,
            gp,
            gv,
            ga,
            raise_on_error=raise_on_error,
        )

    def vjp(self, seed: Sequence[float] | Mapping[str, Any] | None = None, **kwargs: Any) -> "BackwardResult":
        """``backward`` 的别名；``seed=[dL/dp,dL/dv,dL/da]``。"""
        if seed is not None:
            kwargs["seed"] = seed
        return self.backward(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        queries: list[dict[str, Any]] = []
        for idx, (t, j, sr, ss, dc, sc, reason) in enumerate(zip(
            self.times,
            self.joints,
            self.sides_requested,
            self.sides_resolved,
            self.data_certain,
            self.side_certain,
            self.reasons,
        )):
            j = int(j)
            p = self.position[idx, j] if j < self.position.shape[1] else np.nan
            v = self.velocity[idx, j] if j < self.velocity.shape[1] else np.nan
            a = self.acceleration[idx, j] if j < self.acceleration.shape[1] else np.nan
            side_components = (
                self.side_component_certain[idx]
                if self.side_component_certain is not None
                else np.asarray([sc, sc, sc])
            )
            queries.append({
                "time": float(t),
                "joint": j,
                "side_requested": sr,
                "side_resolved": ss,
                "p": None if not np.isfinite(p) else float(p),
                "v": None if not np.isfinite(v) else float(v),
                "a": None if not np.isfinite(a) else float(a),
                "data_certain": [bool(x) for x in dc],
                "side_certain": bool(sc),
                "side_component_certain": [bool(x) for x in side_components],
                "trustworthy_components": [bool(x) for x in self.trustworthy_components[idx]],
                "reasons": list(reason),
            })
        return {
            "trajectory_id": self.trajectory_id,
            "revision": self.revision,
            "snapshot_id": self.snapshot_id,
            "queries": queries,
            "chunk_sizes": list(self.chunk_sizes),
            "basis_evaluations": self.basis_evaluations,
        }


@dataclass(slots=True)
class BackwardResult:
    ok: bool
    trajectory_id: Any
    revision: Any
    snapshot_id: str
    observation_ids: tuple[Any, ...]
    observation_gradients: np.ndarray
    constraint_ids: tuple[Any, ...]
    constraint_gradients: np.ndarray
    reasons: tuple[str, ...] = ()
    joint_gradients: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def observations(self) -> dict[Any, float]:
        return {oid: float(g) for oid, g in zip(self.observation_ids, self.observation_gradients)}

    @property
    def constraints(self) -> dict[Any, float]:
        return {cid: float(g) for cid, g in zip(self.constraint_ids, self.constraint_gradients)}

    def grad_for_observation(self, oid: Any) -> float:
        for x, g in zip(self.observation_ids, self.observation_gradients):
            if x == oid:
                return float(g)
        raise ObservationNotInSnapshotError(
            f"观测 {oid!r} 不属于快照 {self.snapshot_id}；梯度不能交给后续修订才进入的观测"
        )

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise UnidentifiableError(
                "反向传播失败：" + "; ".join(self.reasons),
                trajectory_id=self.trajectory_id,
                revision=self.revision,
                snapshot_id=self.snapshot_id,
                reasons=self.reasons,
            )


@dataclass(slots=True)
class FitSnapshot:
    trajectory_id: Any
    revision: Any
    snapshot_id: str
    spec: TrajectorySpec
    joint_fits: dict[int, JointFit]
    created_index: int = 0

    @property
    def ok(self) -> bool:
        return all(jf.ok for jf in self.joint_fits.values())

    def query(
        self,
        queries: Sequence[QueryRequest | Mapping[str, Any] | Sequence[Any]] | QueryRequest | Mapping[str, Any] | Sequence[Any],
        *,
        budget: QueryBudget | None = None,
        time_tolerance: float = 1e-9,
        compute_data_certainty: bool = True,
        strict: bool = False,
        raise_on_error: bool = False,
    ) -> QueryResult:
        if isinstance(queries, (QueryRequest, Mapping)) or (
            isinstance(queries, Sequence) and queries and not isinstance(queries[0], (QueryRequest, Mapping, Sequence))
        ):
            query_rows = [queries]  # type: ignore[list-item]
        else:
            query_rows = list(queries)  # type: ignore[arg-type]
        reqs = tuple(QueryRequest.from_raw(q, i) for i, q in enumerate(query_rows))
        budget = budget or QueryBudget()
        if budget.max_total_basis_evaluations is not None:
            needed = 3 * len(reqs) * 4
            if needed > budget.max_total_basis_evaluations:
                raise BudgetExceededError(
                    f"查询至少需要 {needed} 次局部基函数求值，预算仅 {budget.max_total_basis_evaluations}"
                )
        result = _evaluate_snapshot(
            self,
            reqs,
            budget=budget,
            time_tolerance=time_tolerance,
            compute_data_certainty=compute_data_certainty,
        )
        if strict or raise_on_error:
            result.raise_if_failed()
        return result

    def query_one(
        self,
        time: float,
        side: str = "auto",
        joint: int = 0,
        *,
        budget: QueryBudget | None = None,
    ) -> QueryResult:
        """单时刻查询；边界侧别或可识别性有问题时直接抛出异常。"""
        return self.query(QueryRequest(time, side, joint, 0), budget=budget, strict=True)

    def backward(
        self,
        query_result: QueryResult,
        grad_position: Any = None,
        grad_velocity: Any = None,
        grad_acceleration: Any = None,
        *,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> BackwardResult:
        gp, gv, ga = _resolve_gradient_args(
            len(query_result.requests), grad_position, grad_velocity, grad_acceleration, kwargs
        )
        return _backward_snapshot(
            self,
            query_result,
            grad_p=gp,
            grad_v=gv,
            grad_a=ga,
            raise_on_error=raise_on_error,
        )

    def vjp(self, query_result: QueryResult, seed: Sequence[float] | None = None, **kwargs: Any) -> BackwardResult:
        if seed is not None:
            kwargs["seed"] = seed
        return self.backward(query_result, **kwargs)


@dataclass(slots=True)
class BatchFitResult:
    snapshots: dict[Any, FitSnapshot]
    failures: dict[Any, SplineKinematicsError]
    stats: dict[str, Any]
    order: tuple[Any, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def results(self) -> dict[Any, FitSnapshot]:
        return self.snapshots

    def __getitem__(self, key: Any) -> FitSnapshot:
        if key in self.failures:
            raise self.failures[key]
        if key in self.snapshots:
            return self.snapshots[key]
        if isinstance(key, int) and 0 <= key < len(self.order):
            trajectory_id = self.order[key]
            if trajectory_id in self.failures:
                raise self.failures[trajectory_id]
            return self.snapshots[trajectory_id]
        raise KeyError(key)

    def __iter__(self):
        for tid in self.order or tuple(self.snapshots):
            yield self.snapshots[tid]

    def __len__(self) -> int:
        return len(self.snapshots)

    def get(self, trajectory_id: Any) -> FitSnapshot | None:
        return self.snapshots.get(trajectory_id)


@dataclass(slots=True)
class BatchQueryResult:
    results: dict[Any, QueryResult]

    def __getitem__(self, trajectory_id: Any) -> QueryResult:
        return self.results[trajectory_id]

    def to_dict(self) -> dict[str, Any]:
        return {str(k): v.to_dict() for k, v in self.results.items()}


@dataclass(slots=True)
class BatchBackwardResult:
    results: dict[Any, BackwardResult]

    def __getitem__(self, trajectory_id: Any) -> BackwardResult:
        return self.results[trajectory_id]


@dataclass(slots=True)
class RevisionDiff:
    trajectory_id: Any
    old_revision: Any
    new_revision: Any
    old_snapshot_id: str
    new_snapshot_id: str
    moved_breaks: list[tuple[int, float, float]]
    added_breaks: list[tuple[int, float]]
    removed_breaks: list[tuple[int, float]]
    changed_knot_sections: list[int]
    changed_interface_indices: list[int]
    directly_coupled_sections_new: list[tuple[int, int, str]]
    directly_affected_sections_old: list[int]
    directly_affected_sections_new: list[int]
    dependency_closure_joints: list[int]
    reusable_trajectory_ids: list[Any]
    reused_joint_keys: list[tuple[Any, int]]
    unchanged_old_sections: tuple[int, ...] = ()
    unchanged_new_sections: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class RevisionStore:
    operator: "BatchSplineOperator"
    history: dict[tuple[Any, Any], str] = field(default_factory=dict)
    snapshots: dict[str, FitSnapshot] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def add(self, snap: FitSnapshot) -> None:
        self.snapshots[snap.snapshot_id] = snap
        self.history[(snap.trajectory_id, _rev_key(snap.revision))] = snap.snapshot_id
        if snap.snapshot_id not in self.order:
            self.order.append(snap.snapshot_id)

    def get(self, trajectory_id: Any, revision: Any = None, snapshot_id: str | None = None) -> FitSnapshot:
        if snapshot_id is not None:
            if snapshot_id not in self.snapshots:
                raise SnapshotNotFoundError(f"没有快照 {snapshot_id!r}")
            return self.snapshots[snapshot_id]
        if revision is None:
            candidates = [s for s in self.snapshots.values() if s.trajectory_id == trajectory_id]
            if not candidates:
                raise SnapshotNotFoundError(f"轨迹 {trajectory_id!r} 没有任何快照")
            return candidates[-1]
        key = (trajectory_id, _rev_key(revision))
        if key not in self.history:
            raise SnapshotNotFoundError(f"轨迹 {trajectory_id!r} 没有修订 {revision!r}")
        return self.snapshots[self.history[key]]


def _rev_key(rev: Any) -> Any:
    try:
        return str(rev)
    except Exception:  # pragma: no cover
        return rev


# ---------------------------------------------------------------------------
# 拟合实现
# ---------------------------------------------------------------------------


def fit_trajectory(
    trajectory: TrajectorySpec | Mapping[str, Any],
    *,
    trajectory_id: Any = None,
    revision: Any = None,
    budget: QueryBudget | None = None,
    raise_on_error: bool = True,
) -> FitSnapshot:
    """拟合一条轨迹的所有关节分量。

    矛盾约束默认直接抛出 :class:`ContradictoryConstraintError`；可识别性失败则
    保留在返回快照的 ``joint_fits`` 中，使查询和反向能返回同一份原因。批量入口
    :func:`fit_batch` 会捕获单轨迹错误并继续拟合其他独立轨迹。
    """
    if not isinstance(trajectory, TrajectorySpec):
        tid = trajectory_id if trajectory_id is not None else trajectory.get("trajectory_id", trajectory.get("id", "trajectory"))  # type: ignore[union-attr]
        spec = TrajectorySpec.from_record(trajectory, trajectory_id=tid, revision=revision)  # type: ignore[arg-type]
    else:
        spec = trajectory
        if trajectory_id is not None and trajectory_id != spec.trajectory_id:
            spec = replace(spec, trajectory_id=trajectory_id)
        if revision is not None and spec.partition.rev != revision:
            spec = replace(spec, partition=replace(spec.partition, rev=revision))
    snapshot_id = f"{spec.trajectory_id}:{spec.partition.rev}:{spec.canonical_hash()}"
    joint_fits: dict[int, JointFit] = {}
    for joint in range(spec.n_joints or 1):
        joint_fits[joint] = _fit_joint(spec, joint, snapshot_id)
    if raise_on_error:
        for jf in joint_fits.values():
            if jf.contradiction is not None:
                raise jf.contradiction
    return FitSnapshot(spec.trajectory_id, spec.partition.rev, snapshot_id, spec, joint_fits)


def fit_batch(
    trajectories: Iterable[TrajectorySpec | Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
    *,
    budget: QueryBudget | None = None,
    raise_on_contract_error: bool = False,
) -> BatchFitResult:
    """逐条轨迹、逐个关节拟合；不会建立跨轨迹的稠密矩阵。"""
    if isinstance(trajectories, Mapping):
        items = list(trajectories.items())
        specs: list[Any] = []
        for tid, raw in items:
            if isinstance(raw, TrajectorySpec):
                specs.append(replace(raw, trajectory_id=tid) if raw.trajectory_id != tid else raw)
            else:
                specs.append(TrajectorySpec.from_record(raw, trajectory_id=tid))
    else:
        specs = list(trajectories)
    snapshots: dict[Any, FitSnapshot] = {}
    failures: dict[Any, SplineKinematicsError] = {}
    order: list[Any] = []
    total_obs = 0
    total_nnz = 0
    max_coeff = 0
    for item in specs:
        tid = getattr(item, "trajectory_id", None)
        if tid is None and isinstance(item, Mapping):
            tid = item.get("trajectory_id", item.get("id"))
        order.append(tid)
        try:
            snap = fit_trajectory(item, budget=budget)
            order[-1] = snap.trajectory_id
            snapshots[snap.trajectory_id] = snap
            total_obs += len(snap.spec.observations)
            for jf in snap.joint_fits.values():
                total_nnz += jf.sparse_nnz
                max_coeff = max(max_coeff, jf.n_coefficients)
                if jf.contradiction is not None:
                    failures[snap.trajectory_id] = jf.contradiction
                elif not jf.ok and jf.unidentifiable is not None:
                    failures[snap.trajectory_id] = jf.unidentifiable
        except SplineKinematicsError as exc:
            tid = getattr(item, "trajectory_id", None)
            if tid is None and isinstance(item, Mapping):
                tid = item.get("trajectory_id", item.get("id"))
            if tid is None:
                tid = "trajectory"
            order[-1] = tid
            failures[tid] = exc
            if raise_on_contract_error:
                raise
    stats = {
        "n_trajectories": len(specs),
        "n_snapshots": len(snapshots),
        "n_failures": len(failures),
        "total_observations": total_obs,
        "total_sparse_nnz": total_nnz,
        "max_joint_coefficients": max_coeff,
        "dense_global_fit_matrix": False,
    }
    return BatchFitResult(snapshots, failures, stats, tuple(order))


def _fit_joint(spec: TrajectorySpec, joint: int, snapshot_id: str) -> JointFit:
    part = spec.partition
    sections = tuple(
        _SectionBasis.build(s, a, b, ks, spec.time_scale)
        for s, ((a, b), ks) in enumerate(zip(zip(part.breaks[:-1], part.breaks[1:]), part.internal_knots))
    )
    offsets = np.zeros(part.n_sections + 1, dtype=int)
    for s, sec in enumerate(sections):
        offsets[s + 1] = offsets[s] + sec.n
    n_total = int(offsets[-1])
    jf = JointFit(
        trajectory_id=spec.trajectory_id,
        joint=joint,
        revision=part.rev,
        snapshot_id=snapshot_id,
        spec=spec,
        sections=sections,
        offsets=offsets,
        n_coefficients=n_total,
    )

    obs = [o for o in spec.observations if o.joint == joint]
    jf.obs_rows = [i for i, o in enumerate(spec.observations) if o.joint == joint]
    trip_rows: list[int] = []
    trip_cols: list[int] = []
    trip_data: list[float] = []
    y = np.empty(len(obs), dtype=float)
    w = np.empty(len(obs), dtype=float)
    # 按 (section, derivative) 分组，减少 SciPy design_matrix 调用。
    groups: dict[tuple[int, int], list[int]] = {}
    obs_section: list[int] = []
    for i, o in enumerate(obs):
        s, resolved, bidx = locate_time(part, o.t, o.side)
        if bidx is not None and 0 < bidx < part.n_sections:
            mode = part.interfaces[bidx - 1]
            if o.kind == "v" and not part.v_equal_at(bidx):
                if o.side == "auto":
                    err = AmbiguousSideError(
                        f"轨迹 {spec.trajectory_id!r} 关节 {joint} 的速度观测位于可断开边界 t={o.t:g}，"
                        "观测行必须显式给 side=left/right",
                        trajectory_id=spec.trajectory_id,
                        joint=joint,
                        time=o.t,
                        boundary=part.breaks[bidx],
                    )
                    jf.status = "ambiguous_side"
                    jf.identifiable = False
                    jf.reasons = (str(err),)
                    jf.unidentifiable = UnidentifiableError(
                        str(err), trajectory_id=spec.trajectory_id, joint=joint, revision=part.rev,
                        snapshot_id=snapshot_id, reasons=("ambiguous_observation_side",),
                    )
                    return jf
        obs_section.append(s)
        derivative = 0 if o.kind == "p" else 1
        groups.setdefault((s, derivative), []).append(i)
        y[i] = o.value
        w[i] = o.weight
    for (s, derivative), idxs in groups.items():
        times = [obs[i].t for i in idxs]
        mat = sections[s].design(times, derivative).tocoo()
        for local_row, col, val in zip(mat.row, mat.col, mat.data):
            i = idxs[int(local_row)]
            trip_rows.append(i)
            trip_cols.append(int(offsets[s] + col))
            trip_data.append(float(val))
    b_obs = sp.csr_matrix(
        (trip_data, (trip_rows, trip_cols)), shape=(len(obs), n_total), dtype=float
    )
    jf.b_obs = b_obs
    jf.w_obs = w
    jf.y_obs = y
    weights = sp.diags(w)
    h_data = (b_obs.T @ weights @ b_obs).tocsr()
    g = (b_obs.T @ (w * y)).A1 if hasattr(b_obs.T @ (w * y), "A1") else np.asarray(b_obs.T @ (w * y)).ravel()

    r_total = sp.block_diag([sec.regularization_matrix() for sec in sections], format="csr")
    h = (h_data + spec.regularization * r_total).tocsr()
    h = 0.5 * (h + h.T)  # 消除浮点不对称
    jf.g_obs = sp.csr_matrix(g.reshape(-1, 1))

    # 用户精确约束。
    a_rows: list[sp.csr_matrix] = []
    b_vals: list[float] = []
    sources: list[dict[str, Any]] = []
    for ci, c in enumerate(spec.constraints):
        if c.joint != joint:
            continue
        s, resolved, bidx = locate_time(part, c.t, c.side)
        if bidx is not None and 0 < bidx < part.n_sections and c.side == "auto":
            if c.kind == "v" and not part.v_equal_at(bidx):
                raise AmbiguousSideError(
                    f"轨迹 {spec.trajectory_id!r} 关节 {joint} 的速度约束在可断开边界 t={c.t:g}，必须声明 left/right",
                    trajectory_id=spec.trajectory_id, joint=joint, time=c.t, boundary=part.breaks[bidx],
                )
            if c.kind == "p" and not part.p_equal_at(bidx):
                raise AmbiguousSideError(
                    f"轨迹 {spec.trajectory_id!r} 关节 {joint} 的位置约束在完全断开边界 t={c.t:g}，必须声明 left/right",
                    trajectory_id=spec.trajectory_id, joint=joint, time=c.t, boundary=part.breaks[bidx],
                )
        derivative = 0 if c.kind == "p" else 1
        row_local = sections[s].design([c.t], derivative).tocsr()
        row = _embed_row(row_local, s, offsets, n_total)
        a_rows.append(row)
        b_vals.append(c.value)
        sources.append({
            "kind": "hard", "index": ci, "id": c.id if c.id is not None else ci,
            "time": c.t, "section": s, "boundary_index": bidx,
            "boundary": part.breaks[bidx] if bidx is not None else None,
            "constraint_kind": c.kind, "side": resolved, "value": c.value,
        })
        jf.user_constraint_rows.append(len(a_rows) - 1)

    # 界面连接条件。
    for bidx in range(1, part.n_sections):
        left_s = bidx - 1
        right_s = bidx
        if part.p_equal_at(bidx):
            row = sp.csr_matrix(([ -1.0, 1.0], ([0, 0], [offsets[left_s + 1] - 1, offsets[right_s]])), shape=(1, n_total))
            a_rows.append(row)
            b_vals.append(0.0)
            sources.append({"kind": "interface_p", "boundary_index": bidx, "boundary": part.breaks[bidx],
                            "sections": (left_s, right_s)})
        if part.v_equal_at(bidx):
            left_row = sections[left_s].design([part.breaks[bidx]], 1)
            right_row = sections[right_s].design([part.breaks[bidx]], 1)
            row = _embed_row(left_row, left_s, offsets, n_total) - _embed_row(right_row, right_s, offsets, n_total)
            a_rows.append(row.tocsr())
            b_vals.append(0.0)
            sources.append({"kind": "interface_v", "boundary_index": bidx, "boundary": part.breaks[bidx],
                            "sections": (left_s, right_s)})

    a_all = sp.vstack(a_rows, format="csr") if a_rows else sp.csr_matrix((0, n_total))
    b_all = np.asarray(b_vals, dtype=float)
    jf.a_all = a_all
    jf.b_all = b_all
    jf.constraint_sources = sources

    try:
        selected = _check_and_select_constraints(a_all, b_all, sources, spec.trajectory_id, joint)
    except ContradictoryConstraintError as err:
        jf.status = "contradictory_constraints"
        jf.identifiable = False
        jf.contradiction = err
        jf.reasons = (str(err),)
        return jf
    jf.selected_constraint_rows = selected
    a_sel = a_all[selected] if len(selected) else sp.csr_matrix((0, n_total))
    b_sel = b_all[selected] if len(selected) else np.empty(0)

    kkt = sp.bmat([[h, a_sel.T], [a_sel, None]], format="csc")
    jf.kkt = kkt
    rhs = np.concatenate([g, b_sel])

    id_reasons, section_reasons = _regularized_identifiability(
        sections, offsets, b_obs, w, a_all, spec.regularization
    )
    if id_reasons:
        err = UnidentifiableError(
            f"轨迹 {spec.trajectory_id!r} 关节 {joint} 未达到可识别条件：{'; '.join(id_reasons)}",
            trajectory_id=spec.trajectory_id,
            joint=joint,
            revision=part.rev,
            snapshot_id=snapshot_id,
            sections=tuple(section_reasons.keys()),
            reasons=tuple(id_reasons),
        )
        jf.status = "unidentifiable"
        jf.identifiable = False
        jf.reasons = err.reasons
        jf.section_reasons = section_reasons
        jf.unidentifiable = err
        jf.c = None
        jf.multipliers = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate = lsqr(kkt, rhs, atol=1e-12, btol=1e-12, iter_lim=10 * max(kkt.shape))[0]
        jf.minimum_norm_candidate = candidate
        jf.sparse_nnz = int(kkt.nnz)
        return jf

    try:
        lu = splu(kkt)
        z = lu.solve(rhs)
        if not np.all(np.isfinite(z)):
            raise RuntimeError("SuperLU returned non-finite solution")
        jf.lu = lu
        jf.c = z[:n_total]
        jf.multipliers = z[n_total:]
        jf.status = "fitted"
        jf.identifiable = True
    except Exception as exc:  # singular / numerical failure -> 明确原因，不填零
        reasons, section_reasons = _diagnose_underidentification(
            b_obs, w, a_all, h, a_sel, kkt, spec, joint
        )
        err = UnidentifiableError(
            f"轨迹 {spec.trajectory_id!r} 关节 {joint} 未达到可识别条件：{'; '.join(reasons) or exc}",
            trajectory_id=spec.trajectory_id,
            joint=joint,
            revision=part.rev,
            snapshot_id=snapshot_id,
            sections=tuple(section_reasons.keys()),
            reasons=reasons or ("singular_kkt",),
        )
        jf.status = "unidentifiable"
        jf.identifiable = False
        jf.reasons = err.reasons
        jf.section_reasons = section_reasons
        jf.unidentifiable = err
        jf.c = None
        jf.multipliers = None
        # 仍然记录一个最小范数候选，仅供高级诊断；默认查询会输出 NaN。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidate = lsqr(kkt, rhs, atol=1e-12, btol=1e-12, iter_lim=10 * max(kkt.shape))[0]
        jf.sparse_nnz = int(kkt.nnz)
        jf.minimum_norm_candidate = candidate
        return jf

    jf.sparse_nnz = int(kkt.nnz)
    return jf


def _embed_row(local: sp.spmatrix, section: int, offsets: np.ndarray, n_total: int) -> sp.csr_matrix:
    coo = local.tocoo()
    rows = np.zeros_like(coo.row)
    cols = coo.col + offsets[section]
    return sp.csr_matrix((coo.data, (rows, cols)), shape=(1, n_total))


def _check_and_select_constraints(
    a: sp.csr_matrix,
    b: np.ndarray,
    sources: Sequence[Mapping[str, Any]],
    trajectory_id: Any,
    joint: int,
) -> np.ndarray:
    m, n = a.shape
    if m == 0:
        return np.zeros(0, dtype=int)
    ad = a.toarray()
    sigma = np.linalg.svd(ad, compute_uv=False)
    scale = sigma[0] if sigma.size and sigma[0] > 0 else max(1.0, float(np.max(np.abs(ad))) if ad.size else 1.0)
    tol = max(1e-10, max(m, n) * np.finfo(float).eps * scale)
    rank = int(np.sum(sigma > tol))
    x, *_ = np.linalg.lstsq(ad, b, rcond=None)
    residual = ad @ x - b
    b_norm = max(1.0, float(np.linalg.norm(b)))
    abs_res = float(np.linalg.norm(residual))
    if abs_res > max(1e-7 * b_norm, 1e-8):
        worst = int(np.argmax(np.abs(residual)))
        src = sources[worst] if worst < len(sources) else {}
        boundary = src.get("boundary")
        bidx = src.get("boundary_index")
        involved = [sources[i] for i in np.argsort(np.abs(residual))[::-1][: min(3, m)]]
        where = f"边界 t={boundary:g}" if boundary is not None else f"时间 {src.get('time')}"
        raise ContradictoryConstraintError(
            f"轨迹 {trajectory_id!r} 关节 {joint} 在{where}的精确约束/界面条件矛盾，"
            f"残差={abs_res:.3g}；涉及 {involved}",
            trajectory_id=trajectory_id,
            joint=joint,
            boundary=boundary,
            boundary_index=bidx,
            involved=involved,
        )
    # 行主元 QR of A^T：A^T[:, pivot] 的 pivot 即 A 的独立行。
    q, r, pivot = scipy_linalg_qr_pivot(ad.T)
    diag = np.abs(np.diag(r))
    qr_rank = int(np.sum(diag > tol * max(1.0, diag[0] if diag.size else 1.0)))
    rank = max(rank, qr_rank)
    return np.asarray(pivot[:rank], dtype=int)


def scipy_linalg_qr_pivot(a: np.ndarray):
    from scipy.linalg import qr

    return qr(a, mode="economic", pivoting=True)


def _piecewise_linear_null_basis(
    sections: Sequence[_SectionBasis], offsets: np.ndarray, n_total: int
) -> np.ndarray:
    """构造各段零加速度空间 1、t 的系数基。

    三次 clamped B 样条中，线性多项式的第 i 个系数等于它在 Greville 横坐标
    ``mean(knots[i+1:i+4])`` 处的值，因此不需要对可能含数值噪声的 R 做秩判定。
    """
    z = np.zeros((n_total, 2 * len(sections)), dtype=float)
    for s, sec in enumerate(sections):
        greville = np.array(
            [sec.physical_knots[i + 1 : i + 4].mean() for i in range(sec.n)], dtype=float
        )
        z[offsets[s] : offsets[s + 1], 2 * s] = 1.0
        z[offsets[s] : offsets[s + 1], 2 * s + 1] = greville
    return z


def _regularized_identifiability(
    sections: Sequence[_SectionBasis],
    offsets: np.ndarray,
    b_obs: sp.csr_matrix,
    w: np.ndarray,
    a_all: sp.csr_matrix,
    regularization: float,
) -> tuple[tuple[str, ...], dict[int, tuple[str, ...]]]:
    n_total = int(offsets[-1])
    positive = w > 0
    if regularization <= 0.0:
        g = sp.csr_matrix((n_total, n_total))
        if np.any(positive):
            bo = b_obs[positive]
            g = g + bo.T @ sp.diags(w[positive]) @ bo
        if a_all.shape[0]:
            g = g + a_all.T @ a_all
        rank = int(np.linalg.matrix_rank(g.toarray()))
        if rank >= n_total:
            return (), {}
        return (f"data_rank={rank}<coefficients={n_total}（regularization=0）",), {
            s: ("data_matrix_rank_deficient",) for s in range(len(sections))
        }

    z = _piecewise_linear_null_basis(sections, offsets, n_total)
    blocks = []
    if np.any(positive):
        blocks.append((b_obs[positive] @ z))
    if a_all.shape[0]:
        blocks.append((a_all @ z))
    if not blocks:
        f = np.zeros((0, z.shape[1]))
    else:
        f = np.vstack([np.asarray(x.toarray()) if sp.issparse(x) else np.asarray(x) for x in blocks])
    rank = int(np.linalg.matrix_rank(f))
    n_affine = z.shape[1]
    if rank == n_affine:
        return (), {}

    _, sval, vh = np.linalg.svd(f, full_matrices=True)
    tol = max(1e-10, max(f.shape) * np.finfo(float).eps * (sval[0] if sval.size else 1.0))
    svd_rank = int(np.sum(sval > tol))
    alpha_null = vh[svd_rank:].T if svd_rank < vh.shape[0] else np.empty((n_affine, 0))
    section_reasons: dict[int, tuple[str, ...]] = {}
    for s in range(len(sections)):
        if alpha_null.size:
            energy = float(np.sum(alpha_null[2 * s : 2 * s + 2, :] ** 2))
            total = float(np.sum(alpha_null**2))
        else:
            energy, total = 1.0, 1.0
        if energy > 1e-12 * max(total, 1.0):
            section_reasons[s] = ("affine_nullspace_not_fixed_by_observations_or_constraints",)
    return (f"regularized_identifiability_rank={rank}<{n_affine}（各段位置/斜率未被固定）",), section_reasons


def _diagnose_underidentification(
    b_obs: sp.csr_matrix,
    w: np.ndarray,
    a_all: sp.csr_matrix,
    h: sp.csr_matrix,
    a_sel: sp.csr_matrix,
    kkt: sp.csc_matrix,
    spec: TrajectorySpec,
    joint: int,
) -> tuple[tuple[str, ...], dict[int, tuple[str, ...]]]:
    reasons: list[str] = []
    section_reasons: dict[int, tuple[str, ...]] = {}
    positive = w > 0
    if np.any(positive):
        bd = b_obs[positive]
        g_data = bd.T @ sp.diags(w[positive]) @ bd
    else:
        g_data = sp.csr_matrix((spec.partition.n_sections and _fit_n(spec), _fit_n(spec)))
    g_con = a_all.T @ a_all if a_all.shape[0] else sp.csr_matrix(g_data.shape)
    g = (g_data + g_con).toarray()
    n = g.shape[0]
    try:
        u, s, vh = np.linalg.svd(0.5 * (g + g.T), full_matrices=True)
        tol = max(1e-9, n * np.finfo(float).eps * (s[0] if s.size else 1.0))
        rank = int(np.sum(s > tol))
        null = vh[rank:].T
    except Exception:
        null = np.empty((n, 0))
        rank = n
    offsets = np.zeros(spec.partition.n_sections + 1, dtype=int)
    # 诊断时重新计算段自由度数；与 _fit_joint 保持同一公式。
    for si, ks in enumerate(spec.partition.internal_knots):
        offsets[si + 1] = offsets[si] + len(ks) + 4
    if null.size:
        reasons.append(f"determining_matrix_rank={rank}<{n}")
        for si in range(spec.partition.n_sections):
            sl = slice(offsets[si], offsets[si + 1])
            energy = np.sum(null[sl, :] ** 2)
            total = np.sum(null**2)
            local_rank = rank  # 仅用于信息
            local_rows = _local_determining_rows(b_obs, a_all, si, offsets)
            local_n = offsets[si + 1] - offsets[si]
            local_rank = int(np.linalg.matrix_rank(local_rows, tol=1e-9)) if local_rows.size else 0
            sec_reasons: list[str] = []
            if energy > 1e-12 * max(total, 1.0):
                sec_reasons.append("nullspace_mode_touches_section")
            if local_rank < local_n:
                sec_reasons.append(f"local_rank={local_rank}<coefficients={local_n}")
            if sec_reasons:
                section_reasons[si] = tuple(sec_reasons)
    # 正则化零空间（每段线性多项式）上的数据覆盖尤其关键。
    if section_reasons:
        reasons.append("insufficient_position_velocity_or_exact_constraint_coverage")
    if not reasons:
        reasons.append("singular_kkt_or_numerically_indefinite")
    return tuple(reasons), section_reasons


def _fit_n(spec: TrajectorySpec) -> int:
    return sum(len(k) + 4 for k in spec.partition.internal_knots)


def _local_determining_rows(b_obs: sp.csr_matrix, a_all: sp.csr_matrix, section: int, offsets: np.ndarray) -> np.ndarray:
    lo, hi = offsets[section], offsets[section + 1]
    rows = []
    if b_obs.shape[0]:
        # 只保留非零列落在本段的观测行；B 样条行本身只有局部列。
        mask = np.zeros(b_obs.shape[0], dtype=bool)
        coo = b_obs.tocoo()
        for r, c in zip(coo.row, coo.col):
            if lo <= c < hi:
                mask[r] = True
        if np.any(mask):
            rows.append(b_obs[mask, lo:hi].toarray())
    if a_all.shape[0]:
        coo = a_all.tocoo()
        mask = np.zeros(a_all.shape[0], dtype=bool)
        for r, c in zip(coo.row, coo.col):
            if lo <= c < hi:
                mask[r] = True
        if np.any(mask):
            rows.append(a_all[mask, lo:hi].toarray())
    return np.vstack(rows) if rows else np.empty((0, hi - lo))


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------


def _evaluate_snapshot(
    snap: FitSnapshot,
    reqs: tuple[QueryRequest, ...],
    *,
    budget: QueryBudget,
    time_tolerance: float,
    compute_data_certainty: bool,
) -> QueryResult:
    nq = len(reqs)
    pos = np.full((nq, max(1, snap.spec.n_joints or 1)), np.nan)
    vel = np.full_like(pos, np.nan)
    acc = np.full_like(pos, np.nan)
    data_certain = np.zeros((nq, 3), dtype=bool)
    side_certain = np.zeros(nq, dtype=bool)
    side_component_certain = np.ones((nq, 3), dtype=bool)
    reasons: tuple[list[str], ...] = tuple([] for _ in range(nq))
    sides_resolved: list[str] = ["auto"] * nq
    joint_ids = np.array([q.joint for q in reqs], dtype=int)
    times = np.array([q.time for q in reqs], dtype=float)

    q_rows: list[sp.csr_matrix] = []
    q_offsets = [0]
    q_joint_fit_ids: list[int] = []
    chunk_sizes: list[int] = []
    basis_evals = 0

    # 为每个 joint 收集查询，同时保持原始顺序。
    for joint in sorted(set(q.joint for q in reqs)):
        jf = snap.joint_fits.get(joint)
        idxs = [i for i, q in enumerate(reqs) if q.joint == joint]
        if jf is None or not jf.ok:
            err = jf.unidentifiable if jf is not None else None
            for i in idxs:
                reasons[i].append("joint_fit_unidentifiable")
                side_component_certain[i, :] = False
                if err is not None:
                    reasons[i].extend(err.reasons)
                q_rows.append(sp.csr_matrix((3, jf.n_coefficients if jf else 0)))
                q_offsets.append(q_offsets[-1] + 3)
                q_joint_fit_ids.append(joint)
            continue

        assert jf.c is not None
        for start in range(0, len(idxs), budget.chunk_size()):
            chunk = idxs[start : start + budget.chunk_size()]
            chunk_sizes.append(len(chunk))
            for i in chunk:
                q = reqs[i]
                part = snap.spec.partition
                try:
                    s, resolved, bidx = locate_time(part, q.time, q.side, time_tolerance=time_tolerance)
                except ContractError as exc:
                    reasons[i].append(str(exc))
                    side_component_certain[i, :] = False
                    q_rows.append(sp.csr_matrix((3, jf.n_coefficients)))
                    q_offsets.append(q_offsets[-1] + 3)
                    q_joint_fit_ids.append(joint)
                    continue

                # 自动侧别在内部边界需要比较左右。
                candidate_sides = [resolved]
                if bidx is not None and 0 < bidx < part.n_sections and q.side == "auto":
                    candidate_sides = ["left", "right"]
                component_rows: list[sp.csr_matrix] = []
                component_values: list[float] = []
                component_data_certain: list[bool] = []
                ambiguous = False
                ambiguity_reasons: list[str] = []
                for deriv in range(3):
                    side_rows_vals = []
                    for side_name in candidate_sides:
                        if side_name == "left":
                            ss = bidx - 1 if bidx is not None else s
                        elif side_name == "right":
                            ss = bidx if bidx is not None else s
                        else:
                            ss = s
                        mat = jf.sections[ss].design([q.time], deriv)
                        row = _embed_row(mat, ss, jf.offsets, jf.n_coefficients)
                        value = float((row @ jf.c)[0])
                        side_rows_vals.append((side_name, row, value))
                        basis_evals += 4
                    if len(side_rows_vals) == 2:
                        (ln, lrow, lv), (rn, rrow, rv) = side_rows_vals
                        if deriv == 0:
                            contract_equal = part.p_equal_at(bidx)
                            name = "p"
                        elif deriv == 1:
                            contract_equal = part.v_equal_at(bidx)
                            name = "v"
                        else:
                            contract_equal = False  # 界面从不约束二阶导
                            name = "a"
                        actual_jump = not np.isclose(lv, rv, rtol=1e-7, atol=1e-7 * max(1.0, abs(lv), abs(rv)))
                        if not contract_equal:
                            # 契约不保证相等时，即使本次数值恰好相同，也不能在未声明
                            # 侧别的情况下交付一个有长期契约意义的“无歧义”导数。
                            ambiguous = True
                            side_component_certain[i, deriv] = False
                            if actual_jump:
                                ambiguity_reasons.append(
                                    f"t={q.time:g} 边界的 {name} 左侧={lv:.8g}/右侧={rv:.8g} 不相等，必须声明 side"
                                )
                            else:
                                ambiguity_reasons.append(
                                    f"t={q.time:g} 的界面契约不保证 {name} 连续；必须声明 side"
                                )
                            component_rows.append(lrow)  # 占位；严格反向会跳过该分量
                            component_values.append(np.nan)
                            component_data_certain.append(False)
                        else:
                            component_rows.append(lrow)
                            component_values.append(lv)
                            component_data_certain.append(True)
                    else:
                        side_name, row, value = side_rows_vals[0]
                        component_rows.append(row)
                        component_values.append(value)
                        component_data_certain.append(True)
                side_certain[i] = bool(np.all(side_component_certain[i]))
                if ambiguous:
                    reasons[i].extend(ambiguity_reasons)
                    sides_resolved[i] = "ambiguous"
                else:
                    sides_resolved[i] = candidate_sides[0] if len(candidate_sides) == 1 else "both_equal"
                pos[i, joint] = component_values[0] if side_component_certain[i, 0] else np.nan
                vel[i, joint] = component_values[1] if side_component_certain[i, 1] else np.nan
                acc[i, joint] = component_values[2] if side_component_certain[i, 2] else np.nan
                q_rows.extend(component_rows)
                q_offsets.append(q_offsets[-1] + 3)
                q_joint_fit_ids.extend([joint] * 3)

                if compute_data_certainty and jf.ok:
                    flags = _rows_data_certain(jf, component_rows)
                    # 界面/正则导致的 KKT 唯一不等于纯数据行空间确定。
                    data_certain[i, :] = flags
                    data_certain[i, ~side_component_certain[i]] = False
                else:
                    data_certain[i, :] = False

    qmat = sp.vstack(q_rows, format="csr") if q_rows else sp.csr_matrix((0, next(iter(snap.joint_fits.values())).n_coefficients if snap.joint_fits else 0))
    # qmat 是按查询/joint 收集顺序，而非全局请求顺序；为了反向简单，构造时每个请求连续 3 行，
    # 但上面按 joint 分组会打乱顺序。因此重新按原始请求构造 Q，避免隐式顺序错误。
    qmat, q_joint_fit_ids, q_offsets = _build_query_qmatrix(snap, reqs)

    fit_identifiable = all(jf.ok for jf in snap.joint_fits.values() if jf.joint in set(q.joint for q in reqs))
    result = QueryResult(
        trajectory_id=snap.trajectory_id,
        snapshot_id=snap.snapshot_id,
        revision=snap.revision,
        requests=reqs,
        times=times,
        joints=joint_ids,
        sides_requested=tuple(q.side for q in reqs),
        sides_resolved=tuple(sides_resolved),
        position=pos,
        velocity=vel,
        acceleration=acc,
        data_certain=data_certain,
        side_certain=side_certain,
        side_component_certain=side_component_certain,
        fit_identifiable=fit_identifiable,
        reasons=tuple(reasons),
        snapshot=snap,
        q_matrix=qmat,
        q_row_offsets=np.asarray(q_offsets, dtype=int),
        joint_fit_ids=np.asarray(q_joint_fit_ids, dtype=int),
        chunk_sizes=tuple(chunk_sizes),
        basis_evaluations=basis_evals,
    )
    return result


def _build_query_qmatrix(
    snap: FitSnapshot, reqs: Sequence[QueryRequest]
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """按原始查询顺序构造 3*nq 行全局（按 joint 分块仅用于反向）算子。

    不同 joint 的系数长度不同，因此这里的“全局”是每个 joint 局部矩阵的纵向拼接；
    ``joint_fit_ids`` 与行一一对应，反向时按 joint 取局部行。
    """
    rows: list[sp.csr_matrix] = []
    joint_ids: list[int] = []
    offsets = [0]
    fallback_n = next(iter(snap.joint_fits.values())).n_coefficients
    for q in reqs:
        jf = snap.joint_fits.get(q.joint)
        for deriv in range(3):
            if jf is None or not jf.ok:
                rows.append(sp.csr_matrix((1, jf.n_coefficients if jf is not None else fallback_n)))
                joint_ids.append(q.joint)
                continue
            part = snap.spec.partition
            try:
                s, resolved, bidx = locate_time(part, q.time, q.side)
            except ContractError:
                rows.append(sp.csr_matrix((1, jf.n_coefficients)))
                joint_ids.append(q.joint)
                continue
            use_s = s
            if bidx is not None and 0 < bidx < part.n_sections and q.side in ("left", "right"):
                use_s = bidx - 1 if q.side == "left" else bidx
            mat = jf.sections[use_s].design([q.time], deriv)
            rows.append(_embed_row(mat, use_s, jf.offsets, jf.n_coefficients))
            joint_ids.append(q.joint)
        offsets.append(offsets[-1] + 3)
    return sp.vstack(rows, format="csr"), np.asarray(joint_ids), np.asarray(offsets)


def _rows_data_certain(jf: JointFit, rows: Sequence[sp.csr_matrix]) -> np.ndarray:
    """查询行是否属于 D=[B_obs;A] 的行空间（不依赖正则先验）。"""
    if jf.b_obs is None or jf.a_all is None:
        return np.zeros(len(rows), dtype=bool)
    n = jf.n_coefficients
    positive = jf.w_obs > 0
    g = sp.csr_matrix((n, n))
    if np.any(positive):
        bo = jf.b_obs[positive]
        g = g + bo.T @ sp.diags(jf.w_obs[positive]) @ bo
    if jf.a_all.shape[0]:
        g = g + jf.a_all.T @ jf.a_all

    # 系数维通常远小于观测数；对单个关节缓存一次零空间，避免每个查询重复 SVD。
    if n <= 512:
        if "null" not in jf.certainty_cache:
            gd = 0.5 * (g.toarray() + g.toarray().T)
            try:
                _, sval, vh = np.linalg.svd(gd, full_matrices=True)
                tol = max(1e-8, n * np.finfo(float).eps * (sval[0] if sval.size else 1.0))
                rank = int(np.sum(sval > tol))
                jf.certainty_cache["null"] = vh[rank:].T
            except np.linalg.LinAlgError:
                jf.certainty_cache["null"] = None
        null = jf.certainty_cache["null"]
        flags = np.ones(len(rows), dtype=bool)
        if null is None:
            return np.zeros(len(rows), dtype=bool)
        if null.size == 0:
            return flags
        for k, row in enumerate(rows):
            b = row.toarray().ravel()
            scale = max(1.0, float(np.linalg.norm(b)))
            projection = float(np.linalg.norm(b @ null)) / scale
            flags[k] = projection < 1e-6
        return flags

    flags = np.zeros(len(rows), dtype=bool)
    for k, row in enumerate(rows):
        b = row.toarray().ravel()
        norm_b = max(1.0, float(np.linalg.norm(b)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = lsqr(g, b, atol=1e-9, btol=1e-9, iter_lim=max(200, 20 * n))[0]
        residual = float(np.linalg.norm(g @ sol - b)) / norm_b
        flags[k] = residual < 1e-6
    return flags


# ---------------------------------------------------------------------------
# 反向
# ---------------------------------------------------------------------------


def _backward_snapshot(
    snap: FitSnapshot,
    qr: QueryResult,
    *,
    grad_p: Any,
    grad_v: Any,
    grad_a: Any,
    raise_on_error: bool,
) -> BackwardResult:
    nq = len(qr.requests)
    gp = _grad_array(grad_p, nq)
    gv = _grad_array(grad_v, nq)
    ga = _grad_array(grad_a, nq)

    obs_ids = tuple(o.id for o in snap.spec.observations)
    obs_grad = np.zeros(len(snap.spec.observations), dtype=float)
    con_ids = tuple(c.id if c.id is not None else i for i, c in enumerate(snap.spec.constraints))
    con_grad = np.zeros(len(snap.spec.constraints), dtype=float)
    reasons: list[str] = []
    joint_gradients: dict[int, np.ndarray] = {}

    requested_joints = set(q.joint for q in qr.requests)
    for joint in requested_joints:
        jf = snap.joint_fits.get(joint)
        idxs = [i for i, q in enumerate(qr.requests) if q.joint == joint]
        if jf is None:
            reasons.append(f"轨迹 {snap.trajectory_id!r} 没有关节 {joint} 的拟合结果")
            continue
        if not jf.ok or jf.lu is None:
            msg = f"轨迹 {snap.trajectory_id!r} 关节 {joint} 未达到可识别条件，反向不产生梯度"
            reasons.append(msg)
            if jf.unidentifiable is not None:
                reasons.extend(jf.unidentifiable.reasons)
            continue
        # 非零梯度不能交给侧别仍有歧义的分量；其他无歧义分量仍可正常反向。
        ambiguous_gradient = False
        component_grads = (gp, gv, ga)
        component_names = ("p", "v", "a")
        for i in idxs:
            for deriv in range(3):
                if component_grads[deriv][i] != 0.0 and qr.side_component_certain is not None \
                        and not qr.side_component_certain[i, deriv]:
                    ambiguous_gradient = True
                    reasons.append(
                        f"查询 {i} 的 {component_names[deriv]} 在边界 {qr.times[i]:g} 未声明唯一 side，"
                        "该分量反向无定义"
                    )
        if ambiguous_gradient and raise_on_error:
            err = AmbiguousSideError(
                "; ".join(reasons),
                trajectory_id=snap.trajectory_id,
                joint=joint,
                time=None,
            )
            raise err
        n = jf.n_coefficients
        gc = np.zeros(n, dtype=float)
        q_global = qr.q_matrix
        # q_global 行顺序与原始请求一致，每请求 p/v/a 三行。
        for i in idxs:
            base = 3 * i
            for deriv, grad in enumerate(component_grads):
                if grad[i] == 0.0:
                    continue
                if qr.side_component_certain is not None and not qr.side_component_certain[i, deriv]:
                    continue
                row = q_global[base + deriv]
                gc += float(grad[i]) * row.toarray().ravel()
        # KKT 对称；解 K x = gc。若结构上不严格对称，可改为分解 K.T。
        n_constraints = len(jf.selected_constraint_rows) if jf.selected_constraint_rows is not None else 0
        try:
            adj = jf.lu.solve(np.concatenate([gc, np.zeros(n_constraints)]))
        except Exception as exc:
            reasons.append(f"adjoint_solve_failed: {exc}")
            continue
        x = adj[:n]
        nu = adj[n:]
        joint_gradients[joint] = x

        # dL/dy_i = w_i * B_i x。
        assert jf.b_obs is not None and jf.w_obs is not None
        local_obs_indices = jf.obs_rows
        bx = jf.b_obs @ x
        for local_i, global_i in enumerate(local_obs_indices):
            obs_grad[global_i] += jf.w_obs[local_i] * float(bx[local_i])

        # 独立 KKT 约束乘子映射回用户精确约束；界面行没有可训练目标。
        selected = jf.selected_constraint_rows if jf.selected_constraint_rows is not None else np.zeros(0, dtype=int)
        sources = jf.constraint_sources
        for mult_pos, source_row in enumerate(selected):
            source = sources[int(source_row)]
            if source["kind"] == "hard":
                ci = int(source["index"])
                # KKT 中 b_sel 的顺序就是 selected 行顺序。
                con_grad[ci] += float(nu[mult_pos])
        # 未被选作主元的用户约束：在一致/冗余点上单独改动会导致不可行，
        # 因此不把乘子随机分给它；保持 0 并在原因中说明。
        selected_source_set = {int(x) for x in selected}
        for row_pos, src in enumerate(sources):
            if src["kind"] == "hard" and row_pos not in selected_source_set:
                ci = int(src["index"])
                reasons.append(f"constraint_{con_ids[ci]}_redundant_gradient_not_unique")

    ok = not reasons
    br = BackwardResult(
        ok=ok,
        trajectory_id=snap.trajectory_id,
        revision=snap.revision,
        snapshot_id=snap.snapshot_id,
        observation_ids=obs_ids,
        observation_gradients=obs_grad,
        constraint_ids=con_ids,
        constraint_gradients=con_grad,
        reasons=tuple(reasons),
        joint_gradients=joint_gradients,
    )
    if not ok and raise_on_error:
        br.raise_if_failed()
    return br


def _resolve_gradient_args(
    nq: int,
    grad_position: Any,
    grad_velocity: Any,
    grad_acceleration: Any,
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """兼容位置/关键字、``(nq,3)`` 张量和 dict 等常见反向调用形式。"""
    kwargs = dict(kwargs or {})
    seed = kwargs.pop("seed", kwargs.pop("cotangent", None))
    if seed is not None:
        seed_arr = np.asarray(seed, dtype=float).ravel()
        if seed_arr.size != 3:
            raise ContractError("seed 必须按 [position, velocity, acceleration] 给出三个分量")
        sp, sv, sa = seed_arr.tolist()
        grad_position = sp if grad_position is None else grad_position
        grad_velocity = sv if grad_velocity is None else grad_velocity
        grad_acceleration = sa if grad_acceleration is None else grad_acceleration
    alias = {
        "p": ("grad_p", "g_p", "dp", "dldp", "dL_dp", "position_grad", "grad_output_p"),
        "v": ("grad_v", "g_v", "dv", "dldv", "dL_dv", "velocity_grad", "grad_output_v"),
        "a": ("grad_a", "g_a", "da", "dlda", "dL_da", "acceleration_grad", "grad_output_a"),
    }
    gradients = kwargs.pop("gradients", None)
    if gradients is not None and grad_position is None:
        grad_position = gradients

    if isinstance(grad_position, Mapping):
        gmap = grad_position
        gp = gmap.get("p", gmap.get("position", gmap.get("grad_position")))
        gv = gmap.get("v", gmap.get("velocity", gmap.get("grad_velocity")))
        ga = gmap.get("a", gmap.get("acceleration", gmap.get("grad_acceleration")))
        return _grad_array(gp, nq), _grad_array(gv, nq), _grad_array(ga, nq)

    if (
        grad_velocity is None
        and grad_acceleration is None
        and grad_position is not None
        and not np.isscalar(grad_position)
    ):
        arr = np.asarray(grad_position, dtype=float)
        if arr.ndim == 1 and arr.size == 3:
            # 领域契约中的 seed 固定为 [dL/dp, dL/dv, dL/da]，对查询组广播。
            return (
                _grad_array(arr[0], nq),
                _grad_array(arr[1], nq),
                _grad_array(arr[2], nq),
            )
        if arr.ndim == 2 and arr.shape == (nq, 3):
            return _grad_array(arr[:, 0], nq), _grad_array(arr[:, 1], nq), _grad_array(arr[:, 2], nq)
        if arr.ndim == 2 and arr.shape == (3, nq):
            return _grad_array(arr[0], nq), _grad_array(arr[1], nq), _grad_array(arr[2], nq)

    values = [grad_position, grad_velocity, grad_acceleration]
    for component, names in alias.items():
        idx = "pva".index(component)
        for name in names:
            if name in kwargs:
                candidate = kwargs.pop(name)
                if values[idx] is not None:
                    raise TypeError(f"组件 {component} 的梯度被重复提供")
                values[idx] = candidate
    if kwargs:
        raise TypeError(f"未知的反向梯度参数：{sorted(kwargs)}")
    return _grad_array(values[0], nq), _grad_array(values[1], nq), _grad_array(values[2], nq)


def _grad_array(value: Any, nq: int) -> np.ndarray:
    if value is None:
        return np.zeros(nq, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(nq, float(arr))
    arr = arr.ravel()
    if arr.shape[0] != nq:
        # 允许 shape (nq,1)
        if arr.size == nq:
            arr = arr.reshape(nq)
        else:
            raise ContractError(f"梯度长度应为 {nq}，实际 {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# 批次算子、修订与对比
# ---------------------------------------------------------------------------


class BatchSplineOperator:
    """持有轨迹输入、所有旧快照和当前修订指针的批量算子。"""

    def __init__(self, trajectories: Iterable[TrajectorySpec | Mapping[str, Any]] | None = None, *, budget: QueryBudget | None = None):
        self.budget = budget or QueryBudget()
        self.store = RevisionStore(self)
        self.current: dict[Any, str] = {}
        self.failures: dict[Any, SplineKinematicsError] = {}
        self._specs: dict[Any, TrajectorySpec] = {}
        if trajectories is not None:
            self.add_many(trajectories)

    def add_many(self, trajectories: Iterable[TrajectorySpec | Mapping[str, Any]] | Mapping[Any, Any]) -> list[FitSnapshot]:
        if isinstance(trajectories, Mapping):
            iterable = [(tid, raw) for tid, raw in trajectories.items()]
        else:
            iterable = [(None, raw) for raw in trajectories]
        snaps = []
        for tid, raw in iterable:
            spec = raw if isinstance(raw, TrajectorySpec) else TrajectorySpec.from_record(raw, trajectory_id=tid or raw.get("id", "trajectory"))  # type: ignore[union-attr]
            if tid is not None:
                spec = replace(spec, trajectory_id=tid)
            self._specs[spec.trajectory_id] = spec
            try:
                snap = fit_trajectory(spec, budget=self.budget)
            except SplineKinematicsError as exc:
                self.failures[spec.trajectory_id] = exc
                continue
            self.failures.pop(spec.trajectory_id, None)
            self.store.add(snap)
            self.current[spec.trajectory_id] = snap.snapshot_id
            snaps.append(snap)
        return snaps

    def fit_revision(
        self,
        trajectory_id: Any,
        revision: Any = None,
        *,
        partition: PartitionSpec | Mapping[str, Any] | None = None,
        observations: Sequence[Any] | None = None,
        constraints: Sequence[Any] | None = None,
        regularization: float | None = None,
    ) -> FitSnapshot:
        old = self._specs.get(trajectory_id)
        if old is None and partition is None:
            raise SnapshotNotFoundError(f"未知轨迹 {trajectory_id!r}")
        if old is None:
            raise ContractError("新轨迹请先 add_many 或提供完整 TrajectorySpec")
        if partition is None:
            part = old.partition if revision is None else None
            if part is None:
                # 从 record 的 partitions 中按 rev 读取由调用方负责；这里要求显式 partition。
                raise ContractError("修订时必须给出 partition")
        else:
            part = PartitionSpec.from_raw(partition, revision) if not isinstance(partition, PartitionSpec) else partition
            if revision is not None and part.rev != revision:
                part = replace(part, rev=revision)
        new_spec = replace(
            old,
            partition=part,
            observations=tuple(Observation.from_raw(o) for o in observations) if observations is not None else old.observations,
            constraints=tuple(HardConstraint.from_raw(c, i) for i, c in enumerate(constraints)) if constraints is not None else old.constraints,
            regularization=old.regularization if regularization is None else regularization,
        )
        self._specs[trajectory_id] = new_spec
        snap = fit_trajectory(new_spec, budget=self.budget)
        self.store.add(snap)
        self.current[trajectory_id] = snap.snapshot_id
        return snap

    def add_record(self, record: Mapping[str, Any], trajectory_id: Any = "record", *, fit_all_revisions: bool = True) -> list[FitSnapshot]:
        partitions = record.get("partitions", [])
        if fit_all_revisions and partitions:
            snaps = []
            for p in partitions:
                spec = TrajectorySpec.from_record(record, trajectory_id=trajectory_id, partition=p)
                self._specs[trajectory_id] = spec
                snap = fit_trajectory(spec, budget=self.budget)
                self.store.add(snap)
                snaps.append(snap)
            self.current[trajectory_id] = snaps[-1].snapshot_id
            return snaps
        spec = TrajectorySpec.from_record(record, trajectory_id=trajectory_id)
        self._specs[trajectory_id] = spec
        snap = fit_trajectory(spec, budget=self.budget)
        self.store.add(snap)
        self.current[trajectory_id] = snap.snapshot_id
        return [snap]

    def get(self, trajectory_id: Any, revision: Any = None, snapshot_id: str | None = None) -> FitSnapshot:
        if snapshot_id is None and revision is None and trajectory_id in self.current:
            snapshot_id = self.current[trajectory_id]
        return self.store.get(trajectory_id, revision, snapshot_id)

    def query(
        self,
        queries_by_trajectory: Mapping[Any, Any] | Sequence[Any],
        *,
        trajectory_id: Any = None,
        revision: Any = None,
        snapshot_id: str | None = None,
        **query_kwargs: Any,
    ) -> QueryResult | BatchQueryResult:
        if trajectory_id is not None:
            snap = self.get(trajectory_id, revision, snapshot_id)
            return snap.query(queries_by_trajectory, budget=self.budget, **query_kwargs)  # type: ignore[arg-type]
        if not isinstance(queries_by_trajectory, Mapping):
            raise ContractError("批量查询必须是 {trajectory_id: queries}，或指定 trajectory_id")
        out = {}
        for tid, qs in queries_by_trajectory.items():
            out[tid] = self.get(tid).query(qs, budget=self.budget, **query_kwargs)
        return BatchQueryResult(out)

    def revise_and_compare(
        self,
        trajectory_id: Any,
        new_partition: PartitionSpec | Mapping[str, Any],
        *,
        new_revision: Any = None,
        observations: Sequence[Any] | None = None,
        constraints: Sequence[Any] | None = None,
        old_revision: Any = None,
    ) -> tuple[FitSnapshot, RevisionDiff]:
        old_snap = self.get(trajectory_id, old_revision)
        new_snap = self.fit_revision(
            trajectory_id,
            new_revision,
            partition=new_partition,
            observations=observations,
            constraints=constraints,
        )
        diff = compare_revisions(old_snap, new_snap, operator=self)
        return new_snap, diff


def compare_revisions(old: FitSnapshot, new: FitSnapshot, *, operator: BatchSplineOperator | None = None) -> RevisionDiff:
    op, np_ = old.spec.partition, new.spec.partition
    moved: list[tuple[int, float, float]] = []
    added: list[tuple[int, float]] = []
    removed: list[tuple[int, float]] = []
    for i, (a, b) in enumerate(zip(op.breaks, np_.breaks)):
        if not np.isclose(a, b, rtol=0, atol=1e-9):
            moved.append((i, a, b))
    old_set = list(op.breaks)
    new_set = list(np_.breaks)
    for bi, t in enumerate(np_.breaks):
        if not any(np.isclose(t, x, rtol=0, atol=1e-9) for x in op.breaks):
            added.append((bi, t))
    for bi, t in enumerate(op.breaks):
        if not any(np.isclose(t, x, rtol=0, atol=1e-9) for x in np_.breaks):
            removed.append((bi, t))
    changed_knots = [
        i for i, (a, b) in enumerate(zip(op.internal_knots, np_.internal_knots)) if tuple(a) != tuple(b)
    ]
    changed_if = [
        i + 1
        for i, (a, b) in enumerate(zip(op.interfaces, np_.interfaces))
        if a != b
    ]
    affected_old = sorted({idx for i, _, _ in moved if 0 < i < len(op.breaks) for idx in (i - 1, i)} |
                          {idx for i, _ in removed if 0 < i < len(op.breaks) for idx in (i - 1, i)})
    affected_new = sorted({idx for i, _, _ in moved if 0 < i < len(np_.breaks) for idx in (i - 1, i)} |
                          {idx for i, _ in added if 0 < i < len(np_.breaks) for idx in (i - 1, i)} |
                          set(changed_knots))
    coupled_new = []
    for bidx in range(1, np_.n_sections):
        t = np_.breaks[bidx]
        is_new = any(abs(t - x[1]) <= 1e-9 for x in added) or any(i == bidx for i, _, _ in moved)
        if is_new or bidx in changed_if:
            coupled_new.append((bidx - 1, bidx, np_.interfaces[bidx - 1]))
    reusable_tids: list[Any] = []
    reused_keys: list[tuple[Any, int]] = []
    if operator is not None:
        for tid, sid in operator.current.items():
            if tid != new.trajectory_id:
                reusable_tids.append(tid)
                snap = operator.store.snapshots[sid]
                reused_keys.extend((tid, j) for j in snap.joint_fits)
    notes = (
        "同一轨迹的界面等式会进入同一个稀疏 KKT；因此直接耦合对列在 directly_coupled_sections_new，"
        "p-v/p-only 链上的系数在数学上可能全局传播。其他轨迹/关节是可原样继续供应的独立单元。",
    )
    unchanged_old = tuple(i for i in range(old.spec.partition.n_sections) if i not in affected_old)
    unchanged_new = tuple(i for i in range(new.spec.partition.n_sections) if i not in affected_new)
    return RevisionDiff(
        trajectory_id=new.trajectory_id,
        old_revision=old.revision,
        new_revision=new.revision,
        old_snapshot_id=old.snapshot_id,
        new_snapshot_id=new.snapshot_id,
        moved_breaks=moved,
        added_breaks=added,
        removed_breaks=removed,
        changed_knot_sections=changed_knots,
        changed_interface_indices=changed_if,
        directly_coupled_sections_new=coupled_new,
        directly_affected_sections_old=affected_old,
        directly_affected_sections_new=affected_new,
        dependency_closure_joints=list(new.joint_fits.keys()),
        reusable_trajectory_ids=reusable_tids,
        reused_joint_keys=reused_keys,
        unchanged_old_sections=unchanged_old,
        unchanged_new_sections=unchanged_new,
        notes=notes,
    )


def query_batch(
    fits: Mapping[Any, FitSnapshot] | BatchFitResult,
    queries: Mapping[Any, Any],
    *,
    budget: QueryBudget | None = None,
) -> BatchQueryResult:
    if isinstance(fits, BatchFitResult):
        mapping = fits.snapshots
    else:
        mapping = fits
    out = {}
    for tid, qs in queries.items():
        out[tid] = mapping[tid].query(qs, budget=budget)
    return BatchQueryResult(out)


def backward_batch(
    query_results: Mapping[Any, QueryResult] | BatchQueryResult,
    gradients: Mapping[Any, Mapping[str, Any]],
) -> BatchBackwardResult:
    qmap = query_results.results if isinstance(query_results, BatchQueryResult) else query_results
    out = {}
    for tid, qr in qmap.items():
        g = gradients.get(tid, {})
        out[tid] = qr.backward(
            grad_position=g.get("p", g.get("position")),
            grad_velocity=g.get("v", g.get("velocity")),
            grad_acceleration=g.get("a", g.get("acceleration")),
        )
    return BatchBackwardResult(out)


def load_operator_from_record(path: str | Path | Mapping[str, Any], trajectory_id: Any = "record") -> BatchSplineOperator:
    if isinstance(path, Mapping):
        record = path
    else:
        with Path(path).open(encoding="utf-8") as handle:
            record = json.load(handle)
    op = BatchSplineOperator(budget=QueryBudget.from_scale(record.get("scale")))
    op.add_record(record, trajectory_id=trajectory_id, fit_all_revisions=True)
    return op


def query(
    fits_or_snapshot: Any,
    queries: Any = None,
    *,
    budget: QueryBudget | None = None,
    **kwargs: Any,
) -> Any:
    """单快照查询或 ``{trajectory_id: queries}`` 批量查询的统一入口。"""
    if isinstance(fits_or_snapshot, FitSnapshot):
        if queries is None:
            raise ContractError("查询快照时必须给出查询时刻")
        return fits_or_snapshot.query(queries, budget=budget, **kwargs)
    if queries is None:
        raise ContractError("批量查询必须给出 {trajectory_id: queries}")
    return query_batch(fits_or_snapshot, queries, budget=budget)


def backward(query_result: Any, gradients: Any = None, **kwargs: Any) -> Any:
    """单个 :class:`QueryResult` 或批量结果的统一反向入口。"""
    if isinstance(query_result, QueryResult):
        if gradients is None:
            return query_result.backward(**kwargs)
        if isinstance(gradients, Mapping):
            return query_result.backward(**gradients, **kwargs)
        return query_result.backward(gradients, **kwargs)
    return backward_batch(query_result, gradients or {})


# 常见类型/函数命名别名。
SplineOperator = BatchSplineOperator
SplineBatchOperator = BatchSplineOperator
BatchSpline = BatchSplineOperator
SplineKinematics = BatchSplineOperator
BatchSplineFitter = BatchSplineOperator
SplineFitter = BatchSplineOperator
SplineKinematicsOperator = BatchSplineOperator
ContactSplineOperator = BatchSplineOperator
FittedSpline = FitSnapshot
SplineFit = FitSnapshot
TrajectoryFit = FitSnapshot
fit_splines = fit_batch
fit_spline_batch = fit_batch
fit_trajectories = fit_batch
fit = fit_batch
query_splines = query_batch
backward_splines = backward_batch

__all__ += [
    "SplineOperator", "SplineBatchOperator", "BatchSpline", "SplineKinematics",
    "BatchSplineFitter", "SplineFitter", "SplineKinematicsOperator", "ContactSplineOperator",
    "FittedSpline", "SplineFit", "TrajectoryFit", "fit_splines", "fit_spline_batch",
    "fit_trajectories", "fit", "query_splines", "query", "backward_splines", "backward",
]


# ---------------------------------------------------------------------------
# 命令行自检/导出示例
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="拟合接触分段三次 B 样条并查询 data/record.json")
    parser.add_argument("path", nargs="?", default=str(Path(__file__).resolve().parent / "data" / "record.json"))
    parser.add_argument("--revision", default=None, help="partitions 中的 rev；默认所有修订")
    parser.add_argument("--backward-check", action="store_true", help="用中心有限差分检查一组梯度")
    args = parser.parse_args(argv)

    op = load_operator_from_record(args.path)
    record = json.loads(Path(args.path).read_text(encoding="utf-8"))
    revisions = [args.revision] if args.revision is not None else [p.get("rev") for p in record.get("partitions", [])]
    output: dict[str, Any] = {"snapshots": []}
    for rev in revisions:
        snap = op.get("record", rev)
        q = snap.query(record.get("query", []))
        item = {"revision": rev, "snapshot_id": snap.snapshot_id, "query": q.to_dict()["queries"]}
        if args.backward_check:
            seed = record.get("seed", [1.0, 0.0, 0.0])
            bw = q.backward(seed=seed)
            item["backward_ok"] = bw.ok
            item["seed"] = seed
            item["observation_gradients"] = bw.observations
            item["constraint_gradients"] = bw.constraints
            item["backward_reasons"] = list(bw.reasons)
        output["snapshots"].append(item)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
