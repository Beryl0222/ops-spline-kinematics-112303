"""接触分段样条算子的行为与反向梯度测试。"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from spline_kinematics_input import load_record
import spline_kinematics as sk


def one_section_spec(tid="t", observations=(), constraints=(), reg=1e-3, joints=None):
    part = sk.PartitionSpec([0.0, 1.0], [[]], [], rev="test")
    return sk.TrajectorySpec(tid, tuple(observations), part, tuple(constraints), reg, "auto", joints)


class SampleRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = load_record()
        cls.op = sk.load_operator_from_record(cls.record)

    def test_old_revision_keeps_left_and_right_velocity_separate(self):
        snap = self.op.get("record", 1)
        left = snap.query_one(0.58, "left")
        right = snap.query_one(0.58, "right")
        self.assertTrue(left.ok)
        self.assertTrue(right.ok)
        self.assertAlmostEqual(left.p[0], right.p[0], places=8)
        self.assertGreater(left.v[0], 0.5)
        self.assertLess(right.v[0], 0.0)
        self.assertGreater(abs(left.v[0] - right.v[0]), 0.7)

    def test_missing_side_is_ambiguous_for_p_only_boundary(self):
        snap = self.op.get("record", 1)
        result = snap.query([[0.58, "auto"]])
        self.assertFalse(result.ok)
        self.assertFalse(result.side_certain[0])
        np.testing.assert_array_equal(result.side_component_certain[0], [True, False, False])
        self.assertTrue(np.isfinite(result.position[0, 0]))
        self.assertFalse(np.isfinite(result.velocity[0, 0]))
        with self.assertRaises(sk.AmbiguousSideError) as ctx:
            snap.query_one(0.58, "auto")
        self.assertEqual(ctx.exception.trajectory_id, "record")
        self.assertAlmostEqual(ctx.exception.boundary, 0.58)

    def test_pv_boundary_fixes_velocity_but_acceleration_still_needs_side(self):
        snap = self.op.get("record", 2)
        left = snap.query_one(0.67, "left")
        right = snap.query_one(0.67, "right")
        self.assertAlmostEqual(left.v[0], right.v[0], places=7)
        with self.assertRaises(sk.AmbiguousSideError):
            snap.query_one(0.67, "auto")

    def test_record_query_rows_match_sides(self):
        snap = self.op.get("record", 1)
        result = snap.query(self.record["query"][:2])
        self.assertEqual(result.sides_resolved[0], "left")
        self.assertEqual(result.sides_resolved[1], "right")
        self.assertEqual(result.basis_evaluations, 24)


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.record = load_record()
        self.spec = sk.TrajectorySpec.from_record(self.record, trajectory_id="grad", revision=1)
        self.snap = sk.fit_trajectory(self.spec)
        self.queries = [[0.27, "auto"], [0.82, "auto"]]
        self.q = self.snap.query(self.queries)

    def _objective(self, obs_values=None, con_values=None):
        obs = []
        for i, o in enumerate(self.spec.observations):
            value = o.value if obs_values is None else obs_values[i]
            obs.append(sk.Observation(o.id, o.t, o.kind, value, o.weight, o.joint, o.side))
        cons = []
        for i, c in enumerate(self.spec.constraints):
            value = c.value if con_values is None else con_values[i]
            cons.append(sk.HardConstraint(c.t, c.side, c.kind, value, c.joint, c.id))
        spec = replace(self.spec, observations=tuple(obs), constraints=tuple(cons))
        snap = sk.fit_trajectory(spec)
        q = snap.query(self.queries)
        return q.position[0, 0] + 2.0 * q.velocity[1, 0] - 0.3 * q.acceleration[0, 0]

    def test_observation_and_constraint_vjp_matches_finite_differences(self):
        bw = self.q.backward(
            grad_position=np.array([1.0, 0.0]),
            grad_velocity=np.array([0.0, 2.0]),
            grad_acceleration=np.array([-0.3, 0.0]),
        )
        self.assertTrue(bw.ok, bw.reasons)
        obs_values = [o.value for o in self.spec.observations]
        for i, obs in enumerate(self.spec.observations):
            up = obs_values.copy()
            dn = obs_values.copy()
            up[i] += 1e-6
            dn[i] -= 1e-6
            fd = (self._objective(up) - self._objective(dn)) / 2e-6
            self.assertAlmostEqual(bw.observation_gradients[i], fd, delta=2e-6, msg=obs.id)

        con_values = [c.value for c in self.spec.constraints]
        for i in range(len(self.spec.constraints)):
            up = con_values.copy()
            dn = con_values.copy()
            up[i] += 1e-6
            dn[i] -= 1e-6
            fd = (self._objective(con_values=up) - self._objective(con_values=dn)) / 2e-6
            self.assertAlmostEqual(bw.constraint_gradients[i], fd, delta=2e-6)

    def test_exact_velocity_constraint_gradient(self):
        part = sk.PartitionSpec([0.0, 1.0], [[]], [])
        cons = (sk.HardConstraint(0.5, "auto", "v", 0.7),)
        spec = sk.TrajectorySpec(
            "v",
            (sk.Observation("p0", 0.0, "p", 0.0, 1.0), sk.Observation("p1", 1.0, "p", 1.0, 1.0)),
            part,
            cons,
            1e-3,
        )
        snap = sk.fit_trajectory(spec)
        q = snap.query([[0.25, "auto"]])
        bw = q.backward(grad_velocity=1.0)
        self.assertTrue(bw.ok)

        def obj(target):
            c = (sk.HardConstraint(0.5, "auto", "v", target),)
            s = sk.TrajectorySpec("v", spec.observations, part, c, 1e-3)
            return sk.fit_trajectory(s).query([[0.25, "auto"]]).velocity[0, 0]

        fd = (obj(0.7 + 1e-6) - obj(0.7 - 1e-6)) / 2e-6
        self.assertAlmostEqual(bw.constraint_gradients[0], fd, places=6)


class ContractAndFailureTests(unittest.TestCase):
    def test_contradictory_boundary_constraints_report_trajectory_and_boundary(self):
        part = sk.PartitionSpec([0.0, 0.5, 1.0], [[0.25], [0.75]], ["p-only"])
        cons = (
            sk.HardConstraint(0.5, "left", "p", 1.0),
            sk.HardConstraint(0.5, "right", "p", 2.0),
        )
        bad = sk.TrajectorySpec("bad-id", (), part, cons, 0.01)
        with self.assertRaises(sk.ContradictoryConstraintError) as ctx:
            sk.fit_trajectory(bad)
        self.assertEqual(ctx.exception.trajectory_id, "bad-id")
        self.assertEqual(ctx.exception.boundary, 0.5)
        self.assertTrue(any("boundary" in str(x) for x in ctx.exception.involved) or ctx.exception.involved)

    def test_batch_keeps_independent_trajectory_after_contradiction(self):
        part = sk.PartitionSpec([0.0, 0.5, 1.0], [[0.25], [0.75]], ["p-only"])
        bad = sk.TrajectorySpec(
            "bad",
            (),
            part,
            (sk.HardConstraint(0.5, "left", "p", 1.0), sk.HardConstraint(0.5, "right", "p", 2.0)),
            0.01,
        )
        good = sk.TrajectorySpec(
            "good",
            (
                sk.Observation("p0", 0.0, "p", 0.0, 10),
                sk.Observation("v0", 0.4, "v", 0.2, 1),
                sk.Observation("p1", 1.0, "p", 0.2, 10),
            ),
            part,
            (sk.HardConstraint(0.0, "right", "p", 0.0), sk.HardConstraint(1.0, "left", "p", 0.2)),
            0.01,
        )
        result = sk.fit_batch([good, bad])
        self.assertFalse(result.ok)
        self.assertIn("bad", result.failures)
        self.assertIn("good", result.snapshots)
        q = result["good"].query_one(0.5, "left")
        self.assertTrue(q.ok)

    def test_insufficient_observations_return_shared_reason_and_nan_not_zero(self):
        part = sk.PartitionSpec([0.0, 0.5, 1.0], [[0.25], [0.75]], ["p-only"])
        spec = sk.TrajectorySpec(
            "thin", (sk.Observation("only", 0.2, "p", 0.1, 1.0),), part, (), 0.01
        )
        snap = sk.fit_trajectory(spec, raise_on_error=False)
        jf = snap.joint_fits[0]
        self.assertEqual(jf.status, "unidentifiable")
        self.assertTrue(jf.reasons)
        q = snap.query([[0.3, "auto"]])
        self.assertFalse(q.ok)
        self.assertTrue(np.all(np.isnan(q.position)))
        self.assertTrue(np.all(np.isnan(q.velocity)))
        self.assertTrue(np.all(np.isnan(q.acceleration)))
        self.assertTrue(any("regularized_identifiability_rank" in r for r in q.reasons[0]))
        bw = q.backward(grad_position=1.0)
        self.assertFalse(bw.ok)
        self.assertTrue(any("未达到可识别条件" in r for r in bw.reasons))
        np.testing.assert_array_equal(bw.observation_gradients, [0.0])
        with self.assertRaises(sk.UnidentifiableError):
            q.backward(grad_position=1.0, raise_on_error=True)

    def test_multi_joint_batch_uses_independent_components(self):
        cons = (
            sk.HardConstraint(0.0, "right", "p", 0.0, joint=0),
            sk.HardConstraint(1.0, "left", "p", 1.0, joint=0),
            sk.HardConstraint(0.0, "right", "p", 10.0, joint=1),
            sk.HardConstraint(1.0, "left", "p", 11.0, joint=1),
        )
        spec = one_section_spec("joints", constraints=cons, reg=0.01, joints=2)
        snap = sk.fit_trajectory(spec)
        q0 = snap.query_one(0.5, "auto", joint=0)
        q1 = snap.query_one(0.5, "auto", joint=1)
        self.assertAlmostEqual(q0.position[0, 0], 0.5, places=6)
        self.assertAlmostEqual(q1.position[0, 1], 10.5, places=6)


class RevisionAndBudgetTests(unittest.TestCase):
    def test_old_snapshot_cannot_receive_gradient_for_new_observation(self):
        record = load_record()
        op = sk.BatchSplineOperator()
        op.add_record(record, "A", fit_all_revisions=True)
        old = op.get("A", 1)
        old_q = old.query([[0.58, "left"]])
        old_bw = old_q.backward(grad_position=1.0)
        self.assertNotIn("future", old_bw.observations)

        new_obs = list(record["observations"]) + [["future", 0.62, "v", 0.1, 1.0]]
        new = op.fit_revision("A", 3, partition=record["partitions"][1], observations=new_obs)
        new_bw = new.query([[0.67, "left"]]).backward(grad_position=1.0)
        self.assertIn("future", new_bw.observations)
        self.assertTrue(new_bw.ok)
        with self.assertRaises(sk.ObservationNotInSnapshotError):
            old_bw.grad_for_observation("future")

    def test_revision_diff_identifies_moved_boundary_and_reuses_other_trajectory(self):
        record = load_record()
        op = sk.BatchSplineOperator()
        op.add_record(record, "A", fit_all_revisions=True)
        other = one_section_spec("B", [sk.Observation("p", 0.5, "p", 0.5, 1)], reg=0.01)
        op.add_many([other])
        _, diff = op.revise_and_compare("A", record["partitions"][1], new_revision=2, old_revision=1)
        self.assertEqual(diff.moved_breaks[0][1:], (0.58, 0.67))
        self.assertIn((0, 1, "p-v"), diff.directly_coupled_sections_new)
        self.assertIn("B", diff.reusable_trajectory_ids)
        self.assertIn(("B", 0), diff.reused_joint_keys)
        old_again = op.get("A", 1)
        self.assertAlmostEqual(old_again.query_one(0.58, "left").v[0], 0.7050091733877899, places=8)

    def test_budget_chunks_query_but_preserves_results(self):
        record = load_record()
        snap = sk.fit_trajectory(sk.TrajectorySpec.from_record(record, trajectory_id="budget", revision=1))
        times = np.linspace(0.05, 1.05, 9)
        queries = [[float(t), "auto"] for t in times]
        budget = sk.QueryBudget(scratch_bytes=2048, bytes_per_query_row=768)
        q = snap.query(queries, budget=budget)
        self.assertEqual(budget.chunk_size(), 2)
        self.assertEqual(q.chunk_sizes, (2, 2, 2, 2, 1))
        self.assertEqual(len(q.p), 9)

    def test_batch_fit_stats_declare_no_global_dense_matrix(self):
        result = sk.fit_batch(
            [one_section_spec(f"b{i}", [sk.Observation("p", 0.5, "p", 0.5, 1)], reg=0.1) for i in range(3)]
        )
        self.assertEqual(result.stats["n_trajectories"], 3)
        self.assertFalse(result.stats["dense_global_fit_matrix"])
        self.assertTrue(all(sp.issparse(jf.b_obs) for s in result.snapshots.values() for jf in s.joint_fits.values()))


if __name__ == "__main__":
    unittest.main()
