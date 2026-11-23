"""本地合成交换样本的读取入口。"""
import argparse
import json
from pathlib import Path

import numpy as np


SAMPLE = Path(__file__).resolve().parent / "data" / "record.json"
FIELDS = ('observations', 'partitions', 'hard_constraints', 'query', 'seed', 'regularization', 'scale')


def load_record(path=SAMPLE):
    """读取原始数据并检查顶层交换字段。"""
    with Path(path).open(encoding="utf-8") as handle:
        record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError("输入应为一个 JSON 对象")
    missing = set(FIELDS).difference(record)
    if missing:
        raise ValueError("缺少交换字段：" + "、".join(sorted(missing)))
    return record


def main():
    parser = argparse.ArgumentParser(description="查看本地交换样本的项目")
    parser.add_argument("path", nargs="?", type=Path, default=SAMPLE, help="JSON 记录路径")
    args = parser.parse_args()
    record = load_record(args.path)
    print(json.dumps({"记录": args.path.name, "输入项目": sorted(record), "NumPy": np.__version__},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
