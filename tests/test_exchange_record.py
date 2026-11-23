"""检查交换数据中用于回查的事实。"""
import unittest

import numpy as np

from spline_kinematics_input import load_record


class ExchangeRecordTests(unittest.TestCase):
    def test_exchange_contract(self):
        record = load_record()
        self.assertTrue(record["partitions"][0]["breaks"][1] in [q[0] for q in record["query"]])
        self.assertTrue(record["partitions"][0]["interfaces"] != record["partitions"][1]["interfaces"])
        self.assertTrue(any(e[2] == "v" for e in record["observations"]))
        self.assertTrue(all(np.all(np.diff(p["breaks"]) > 0) for p in record["partitions"]))
