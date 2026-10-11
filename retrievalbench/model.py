"""Pinned MiniLM ONNX inference: masked mean pooling, then L2 normalization."""
import os
import threading
from functools import lru_cache

import numpy as np

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_FILE = 'onnx/model.onnx'
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    kwargs = dict(repo_id=MODEL_ID, revision=MODEL_REVISION, token=False,
                  local_files_only=os.environ.get('RETRIEVALBENCH_OFFLINE') == '1')
    try:
        tokenizer = Tokenizer.from_file(hf_hub_download(filename='tokenizer.json', **kwargs))
        path = hf_hub_download(filename=MODEL_FILE, **kwargs)
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(path, opts, providers=['CPUExecutionProvider'])
        return tokenizer, session
    except Exception as exc:
        raise RuntimeError('MiniLM could not be loaded. Run retrievalbench download with internet access, '
                           'or choose TF-IDF. No fallback was silently substituted. ' + str(exc)) from exc


def encode_texts(texts, ids):
    tokenizer, session = _load()
    vectors, truncated = [], []
    # The shared tokenizer is stateful; serialize batch work across web requests.
    with _lock:
        for start in range(0, len(texts), 16):
            batch = texts[start:start + 16]
            tokens = tokenizer.encode_batch(batch)
            truncated.extend(identifier for identifier, tok in zip(ids[start:start + 16], tokens) if tok.overflowing)
            inputs = {'input_ids': np.array([x.ids for x in tokens], dtype=np.int64),
                      'attention_mask': np.array([x.attention_mask for x in tokens], dtype=np.int64),
                      'token_type_ids': np.array([x.type_ids for x in tokens], dtype=np.int64)}
            feed = {x.name: inputs[x.name] for x in session.get_inputs()}
            hidden = session.run(None, feed)[0]
            mask = inputs['attention_mask'][..., None]
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if not np.isfinite(pooled).all() or np.any(norms == 0):
                raise RuntimeError('MiniLM returned invalid embeddings.')
            vectors.append(pooled / norms)
    return np.vstack(vectors), {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
        'model_file': MODEL_FILE, 'max_tokens': 256, 'pooling': 'attention-mask mean + L2',
        'truncated_ids': truncated, 'provider': 'CPUExecutionProvider'}
