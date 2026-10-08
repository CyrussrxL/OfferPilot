"""
分类系统评测 —— run_classification_eval.py

职责：
  对 GuardAgent 的三层分类链路（代码检测 → 向量相似度 → LLM 仲裁 → 关键词兜底）
  进行离线评测，输出可复现的量化指标，支撑简历数据与回归测试。

指标：
  - 总体 Accuracy
  - 每类 Precision / Recall / F1
  - 混淆矩阵
  - LLM 仲裁触发率（低置信度样本占比，反映分层策略的必要性）
  - 平均置信度

用法：
  python evaluation/run_classification_eval.py            # fast 模式（向量直出，无 LLM 仲裁）
  python evaluation/run_classification_eval.py --full     # full 模式（含 LLM 仲裁，慢）

设计理由：
  - fast 模式评估"向量层 + 规则层"的基础能力，跑一次约 1 分钟；
  - full 模式评估真实线上全链路（低置信度触发 LLM few-shot 仲裁）；
  - 评测集 100 条（25 × 4 类），与训练种子隔离（可做留出集验证）。
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger

CATEGORIES = ["coding", "career", "emotional", "chitchat"]


def load_testset():
    data_file = os.path.join(os.path.dirname(__file__), "data", "classification_testset.json")
    with open(data_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cases = []
    for category, texts in raw.items():
        for text in texts:
            cases.append({"text": text, "label": category})
    return cases


def run_eval(full_mode: bool):
    if not full_mode:
        settings.GUARD_LLM_ARBITRATION = False
        logger.info("评测模式: fast（禁用 LLM 仲裁，纯向量+规则层）")
    else:
        logger.info("评测模式: full（全链路，含 LLM 仲裁）")

    from companion_ai.agents.guard_agent import _classify_message

    cases = load_testset()
    confusion = defaultdict(lambda: defaultdict(int))
    arbitration_count = 0
    confidences = []
    correct = 0
    errors = []
    t0 = time.time()

    for i, case in enumerate(cases, 1):
        try:
            pred, conf, arb_used = _classify_message(case["text"])
        except Exception as e:
            logger.warning(f"第 {i} 条分类异常: {e}")
            pred, conf, arb_used = "chitchat", 0.0, False
        confusion[case["label"]][pred] += 1
        confidences.append(conf)
        if arb_used:
            arbitration_count += 1
        if pred == case["label"]:
            correct += 1
        else:
            errors.append(
                {"text": case["text"][:40], "label": case["label"], "pred": pred, "conf": round(conf, 2)}
            )
        if i % 20 == 0:
            logger.info(f"进度: {i}/{len(cases)}")

    elapsed = time.time() - t0
    accuracy = correct / len(cases)

    # 每类 PRF
    per_category = {}
    for cat in CATEGORIES:
        tp = confusion[cat][cat]
        fp = sum(confusion[other][cat] for other in CATEGORIES if other != cat)
        fn = sum(confusion[cat][other] for other in CATEGORIES if other != cat)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_category[cat] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return {
        "mode": "full" if full_mode else "fast",
        "total": len(cases),
        "accuracy": round(accuracy, 4),
        "per_category": per_category,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "arbitration_rate": round(arbitration_count / len(cases), 4),
        "avg_confidence": round(sum(confidences) / len(confidences), 4),
        "elapsed_sec": round(elapsed, 1),
        "errors": errors,
    }


def write_report(result):
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    mode = result["mode"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(report_dir, f"classification_report_{mode}_{timestamp}.md")

    lines = [
        "# GuardAgent 分类系统评测报告",
        "",
        f"- 模式: **{mode}**（{'全链路，含 LLM 仲裁' if mode == 'full' else '向量+规则层，禁用 LLM 仲裁'}）",
        f"- 评测集: {result['total']} 条（25 × 4 类，与种子隔离）",
        f"- **总体 Accuracy: {result['accuracy']:.2%}**",
        f"- 平均置信度: {result['avg_confidence']:.3f}",
        f"- LLM 仲裁触发率: {result['arbitration_rate']:.2%}",
        f"- 总耗时: {result['elapsed_sec']}s",
        "",
        "## 每类指标",
        "",
        "| 类别 | Precision | Recall | F1 | 样本数 |",
        "|---|---|---|---|---|",
    ]
    for cat in CATEGORIES:
        m = result["per_category"][cat]
        lines.append(f"| {cat} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {m['support']} |")

    lines += ["", "## 混淆矩阵（行=真实，列=预测）", "",
              "| 真实\\预测 | " + " | ".join(CATEGORIES) + " |",
              "|---|" + "---|" * len(CATEGORIES)]
    for true_cat in CATEGORIES:
        row = [str(result["confusion"].get(true_cat, {}).get(p, 0)) for p in CATEGORIES]
        lines.append(f"| {true_cat} | " + " | ".join(row) + " |")

    if result["errors"]:
        lines += ["", "## 错分样本（前 10 条）", ""]
        for e in result["errors"][:10]:
            lines.append(f"- [{e['label']} → {e['pred']}] (conf={e['conf']}) {e['text']}...")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写入: {filepath}")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="全链路模式（含 LLM 仲裁）")
    args = parser.parse_args()

    result = run_eval(full_mode=args.full)
    write_report(result)

    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("confusion", "errors")},
        ensure_ascii=False, indent=2,
    ))
