"""Verify the trusted bundled production artifact; never fit or select a model."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify():
    import joblib
    root = Path(__file__).resolve().parents[1]
    path = root / 'models/production.joblib'
    manifest = json.loads((root / 'models/manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Production artifact checksum mismatch')
    bundle = joblib.load(path)
    for key in ['model_version', 'selected_model', 'dataset_sha256', 'threshold']:
        if bundle['metadata'][key] != manifest[key]:
            raise ValueError(f'Artifact metadata mismatch: {key}')
    if bundle['metadata']['selected_model'] != 'LightGBM':
        raise ValueError('This lean runtime is packaged for the frozen LightGBM model')
    if 'torch' in sys.modules:
        raise ValueError('Runtime unexpectedly imported PyTorch')
    print(json.dumps({'status': 'ok', **manifest}, indent=2))


if __name__ == '__main__':
    verify()
