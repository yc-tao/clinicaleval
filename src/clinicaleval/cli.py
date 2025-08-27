import argparse
import json
import logging
import os
import random
import subprocess
import tempfile
from typing import Any, Dict, List

import pandas as pd
import yaml


def load_config(path: str, overrides: List[str]) -> Dict[str, Any]:
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


def read_json(path: str, max_samples: int) -> List[Dict[str, Any]]:
    """Read a JSON file containing a list of records."""
    if not os.path.exists(path):
        print(f"Data file not found: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        records = data
    else:
        # If it's not a list, assume it's a single record
        records = [data]
    
    if 0 < max_samples <= len(records):
        records = records[:max_samples]
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
        elif mode == 'yes_no_classification':
            # For the actual implementation, this would call an LLM
            # For now, we'll return a placeholder that can be processed by to_label
            outputs.append("yes")
        else:
            outputs.append("")
    return outputs


def to_label(text: str) -> str:
    return 'yes' if 'yes' in text.lower() else 'no'


def label_to_int(x: Any) -> int:
    # Handle both string labels and integer labels
    if isinstance(x, int):
        return x
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


def run_lighteval(cfg: Dict[str, Any]) -> int:
    le_cfg = cfg.get('lighteval', {})
    backend = le_cfg.get('backend', 'vllm')
    model_path = le_cfg.get('model_path') or le_cfg.get('model_name')
    if not model_path:
        raise ValueError("lighteval.model_path (or model_name) must be set in config")

    model_yaml: Dict[str, Any] = {
        'model_parameters': {
            'model_name': model_path,
        }
    }
    if isinstance(le_cfg.get('model_parameters'), dict):
        model_yaml['model_parameters'].update(le_cfg['model_parameters'])
    if isinstance(le_cfg.get('generation_parameters'), dict):
        model_yaml['model_parameters']['generation_parameters'] = le_cfg['generation_parameters']

    report_cfg = cfg.get('report', {})
    out_dir = report_cfg.get('out_dir', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile('w', suffix='.yaml', prefix='lighteval_model_', dir=out_dir, delete=False) as tf:
        yaml.safe_dump(model_yaml, tf, sort_keys=False)
        model_yaml_path = tf.name

    task_string = le_cfg.get('tasks', 'leaderboard|boolq|0|0')

    cmd = [
        'lighteval',
        backend,
        model_yaml_path,
        task_string,
    ]

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Running lighteval: %s', ' '.join(cmd))
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError as e:
        raise RuntimeError("`lighteval` CLI not found. Please install: pip install lighteval vllm") from e


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal evaluation script")
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('overrides', nargs='*', help='key=value overrides')
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    seed = cfg.get('seed', 0)
    set_seed(seed)

    le_cfg = cfg.get('lighteval', {})
    if isinstance(le_cfg, dict) and le_cfg.get('enabled'):
        rc = run_lighteval(cfg)
        if rc != 0:
            raise SystemExit(rc)
        return

    data_cfg = cfg.get('data', {})
    data_path = data_cfg.get('path', '')
    
    # Determine file format and load accordingly
    if data_path.endswith('.jsonl'):
        records = read_jsonl(data_path, data_cfg.get('max_samples', 0))
    elif data_path.endswith('.json'):
        records = read_json(data_path, data_cfg.get('max_samples', 0))
    else:
        print(f"Unsupported file format: {data_path}")
        return
        
    if not records:
        print('No data samples found.')
        return

    text_key = data_cfg.get('text_key', 'input')
    label_key = data_cfg.get('label_key', 'label')
    inputs = [r[text_key] for r in records]
    raw_golds = [r[label_key] for r in records]
    
    # Convert integer labels to yes/no format if needed
    golds = []
    for gold in raw_golds:
        if isinstance(gold, int):
            golds.append('yes' if gold == 1 else 'no')
        else:
            golds.append(str(gold).lower())

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


