import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import yaml

# TODO: Extract DataSource into module
# TODO: Extract InferenceBackend into module
# TODO: Extract Metric into module


def load_config(path: str, overrides: List[str]) -> Dict[str, Any]:
    """Load YAML config and apply key=value overrides."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    for ov in overrides:
        if '=' not in ov:
            continue
        key, val = ov.split('=', 1)
        keys = key.split('.')
        target = cfg
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = yaml.safe_load(val)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def read_jsonl(path: str, max_samples: int) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"Data file not found: {path}")
        return []
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if 0 < max_samples <= len(records):
                break
    return records


def generate(inputs: List[str], cfg: Dict[str, Any]) -> List[str]:
    mode = cfg.get('mode', 'reverse')
    outputs = []
    for x in inputs:
        if mode == 'reverse':
            outputs.append(x[::-1])
        elif mode == 'echo':
            outputs.append(x)
        elif mode == 'uppercase':
            outputs.append(x.upper())
        else:
            outputs.append("")
    return outputs


def to_label(text: str) -> str:
    return 'yes' if 'yes' in text.lower() else 'no'


def label_to_int(x: Any) -> int:
    return 1 if str(x).strip().lower() in {'yes', '1', 'true'} else 0


def compute_metrics(preds: List[Any], golds: List[Any]) -> Dict[str, float]:
    n = len(golds)
    exact = sum(p == g for p, g in zip(preds, golds)) / n if n else 0.0
    tp = sum(label_to_int(p) and label_to_int(g) for p, g in zip(preds, golds))
    fp = sum(label_to_int(p) and not label_to_int(g) for p, g in zip(preds, golds))
    fn = sum((not label_to_int(p)) and label_to_int(g) for p, g in zip(preds, golds))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {'exact_match': exact, 'f1_binary': f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal evaluation script")
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('overrides', nargs='*', help='key=value overrides')
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    seed = cfg.get('seed', 0)
    set_seed(seed)

    data_cfg = cfg.get('data', {})
    records = read_jsonl(data_cfg.get('path', ''), data_cfg.get('max_samples', 0))
    if not records:
        print('No data samples found.')
        return

    text_key = data_cfg.get('text_key', 'input')
    label_key = data_cfg.get('label_key', 'label')
    inputs = [r[text_key] for r in records]
    golds = [r[label_key] for r in records]

    gen_cfg = cfg.get('gen', {})
    template = gen_cfg.get('template', '{x}')
    prompts = [template.format(x=inp) for inp in inputs]
    raw_outputs = generate(inputs, gen_cfg)
    preds = [to_label(o) for o in raw_outputs]
    matches = [p == g for p, g in zip(preds, golds)]

    metrics = compute_metrics(preds, golds)

    report_cfg = cfg.get('report', {})
    out_dir = report_cfg.get('out_dir', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    # logging
    if report_cfg.get('save_log', True):
        log_path = os.path.join(out_dir, 'run.log')
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Loaded %d samples', len(records))
        logging.info('Generation mode: %s', gen_cfg.get('mode'))
        logging.info('Outputs saved to %s', out_dir)

    if report_cfg.get('save_samples', True):
        df = pd.DataFrame({
            'id': list(range(len(records))),
            'input': inputs,
            'gold': golds,
            'prompt': prompts,
            'raw_output': raw_outputs,
            'pred_label': preds,
            'match': matches,
        })
        df.to_csv(os.path.join(out_dir, 'samples.csv'), index=False)

    if report_cfg.get('save_metrics', True):
        payload = {
            'metrics': metrics,
            'meta': {
                'n_samples': len(records),
                'seed': seed,
            },
            'config_dump': cfg,
        }
        with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
