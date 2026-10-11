"""Validated paragraph corpora and deterministic retrieval evaluation."""
import hashlib
import json
import re
import time
import platform
import importlib.metadata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_corpus(files):
    files = list(files)
    if not 1 <= len(files) <= 100 or sum(len(b) for _, b in files) > 5*1024*1024:
        raise ValueError('Supply 1–100 documents, at most 5 MiB total.')
    passages, names = [], set()
    for name, raw in sorted(files):
        if not isinstance(name, str) or not re.fullmatch(r'[\w .-]{1,160}\.(txt|md)', name) or name in names:
            raise ValueError('Use unique plain .txt/.md filenames without directory paths.')
        names.add(name)
        try: text = raw.decode('utf-8-sig').replace('\r\n','\n').replace('\r','\n')
        except UnicodeError as exc: raise ValueError(f'{name}: expected UTF-8 text.') from exc
        chunks = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not chunks or any(len(p)>20000 for p in chunks):
            raise ValueError(f'{name}: empty file or paragraph exceeds 20,000 characters.')
        for i, chunk in enumerate(chunks, 1):
            passages.append({'id': f'{name}#p{i:03}', 'document': name, 'text': chunk})
        if len(passages)>500: raise ValueError('Corpus exceeds 500 paragraphs.')
    return {'sha256': digest(passages), 'passages': passages, 'document_count': len(names)}


def parse_questions(data, corpus):
    if len(data)>1024*1024: raise ValueError('Questions exceed 1 MiB.')
    try: obj=json.loads(data)
    except (ValueError,UnicodeError) as exc: raise ValueError('Questions must be UTF-8 JSON.') from exc
    description = obj.get('description', 'User-supplied annotations; provenance unspecified.') if isinstance(obj,dict) else 'User-supplied annotations; provenance unspecified.'
    if not isinstance(description,str) or len(description)>2000: raise ValueError('Description must be a string of at most 2000 characters.')
    fingerprint = obj.get('corpus_sha256') if isinstance(obj,dict) else None
    rows = obj.get('questions') if isinstance(obj,dict) else obj
    if fingerprint is not None and fingerprint != corpus['sha256']:
        raise ValueError('Corpus fingerprint mismatch. Recheck passage annotations.')
    if not isinstance(rows,list) or not 1<=len(rows)<=100:
        raise ValueError('Supply 1–100 annotated questions.')
    known={p['id'] for p in corpus['passages']}; seen=set(); clean=[]
    for row in rows:
        if not isinstance(row,dict): raise ValueError('Each question must be an object.')
        qid=row.get('id'); question=row.get('question'); relevant=row.get('relevant_passage_ids')
        if not isinstance(qid,str) or not qid.strip() or len(qid)>100 or qid in seen:
            raise ValueError('Question IDs must be unique nonempty strings, at most 100 characters.')
        if not isinstance(question,str) or not question.strip() or len(question)>2000:
            raise ValueError('Question text must contain 1–2000 characters.')
        if not isinstance(relevant,list) or not relevant or any(not isinstance(x,str) for x in relevant):
            raise ValueError('Every question needs a nonempty relevant_passage_ids list.')
        if len(set(relevant))!=len(relevant) or not set(relevant)<=known:
            raise ValueError('Relevant IDs must be unique and present in this corpus.')
        seen.add(qid); clean.append({'id':qid,'question':question.strip(),'relevant_passage_ids':relevant})
    return {'description':description,'questions':clean,'sha256':digest(clean), 'warnings': [] if fingerprint else
            ['No corpus fingerprint supplied; verify that paragraph IDs still refer to the labelled text.']}


def validate_k(k):
    if type(k) is not int or not 1<=k<=50: raise ValueError('k must be an integer from 1 to 50.')


def rank_scores(passages, scores, k):
    validate_k(k); scores=np.asarray(scores,dtype=float)
    if scores.shape!=(len(passages),) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and match the passage count.')
    order=sorted((i for i,s in enumerate(scores) if s>0),key=lambda i:(-scores[i],passages[i]['id']))[:k]
    return [{'passage_id':passages[i]['id'],'document':passages[i]['document'],
             'text':passages[i]['text'],'score':float(scores[i]),'rank':r} for r,i in enumerate(order,1)]


def query_metrics(ranked, relevant, k):
    validate_k(k)
    if not relevant: raise ValueError('At least one relevant passage is required.')
    ids=[r['passage_id'] for r in ranked[:k]]; gold=set(relevant); hits=gold.intersection(ids)
    rr=next((1/r for r,x in enumerate(ids,1) if x in gold),0.0)
    return {'recall_at_k':len(hits)/len(gold),'reciprocal_rank_at_k':rr,
            'hit_at_k':int(bool(hits)),'missed_relevant_ids':sorted(gold-set(ids))}


def run_benchmark(corpus, questions, methods=('tfidf','minilm'), k=3):
    validate_k(k)
    if not methods or len(set(methods))!=len(methods) or set(methods)-{'tfidf','minilm'}:
        raise ValueError('Choose unique methods from tfidf and minilm.')
    passages=corpus['passages']; texts=[p['text'] for p in passages]; outputs={}
    for method in methods:
        started=time.perf_counter(); metadata={}; load_ms=0
        if method=='tfidf':
            vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=20000)
            matrix=vectorizer.fit_transform(texts)
            def search(q): return (vectorizer.transform([q]) @ matrix.T).toarray()[0], []
            metadata={'ngram_range':[1,2],'max_features':20000,'fitted_on':'corpus only'}
        else:
            from .model import _load, encode_texts
            _load(); load_ms=(time.perf_counter()-started)*1000; started=time.perf_counter()
            matrix,metadata=encode_texts(texts,[p['id'] for p in passages])
            def search(q):
                v, info=encode_texts([q],['query']); return (v @ matrix.T)[0], info['truncated_ids']
        index_ms=(time.perf_counter()-started)*1000
        warm=time.perf_counter(); search(questions['questions'][0]['question']); warmup_ms=(time.perf_counter()-warm)*1000
        rows=[]; durations=[]
        for q in questions['questions']:
            t=time.perf_counter(); scores,truncated=search(q['question']); ranked=rank_scores(passages,scores,k)
            elapsed=(time.perf_counter()-t)*1000; durations.append(elapsed)
            rows.append({**q,'results':ranked,'metrics':query_metrics(ranked,q['relevant_passage_ids'],k),
                         'latency_ms':elapsed,'query_truncated':bool(truncated)})
        outputs[method]={'summary':{'recall_at_k':float(np.mean([r['metrics']['recall_at_k'] for r in rows])),
            'mrr_at_k':float(np.mean([r['metrics']['reciprocal_rank_at_k'] for r in rows])),
            'hit_rate_at_k':float(np.mean([r['metrics']['hit_at_k'] for r in rows]))},
            'timing':{'load_ms':load_ms,'index_ms':index_ms,'warmup_ms':warmup_ms,
                'query_ms_median':float(np.median(durations)),'query_ms_p95':float(np.percentile(durations,95))},
            'metadata':metadata,'queries':rows}
    return {'schema_version':1,'dataset_description':questions['description'],'k':k,'corpus_sha256':corpus['sha256'],'questions_sha256':questions['sha256'],
        'passage_count':len(passages),'question_count':len(questions['questions']),
        'warnings':questions['warnings'],'methods':outputs,
        'environment':{'python':platform.python_version(),'platform':platform.platform(),
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','scikit-learn','onnxruntime','tokenizers']}},
        'protocol':'One warmup per method; single timed call per query including encoding, scoring and ranking. Nonpositive scores excluded.'}
