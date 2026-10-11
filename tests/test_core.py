import json
import numpy as np
import pytest
from retrievalbench.core import build_corpus,parse_questions,rank_scores,query_metrics,run_benchmark

def corpus():
    return build_corpus([('guide.md',b'Password reset emails expire in fifteen minutes.\n\nCSV downloads contain every row.'),
                         ('setup.txt',b'Install the application using pip.')])

def questions(c=None):
    c=c or corpus()
    return parse_questions(json.dumps({'corpus_sha256':c['sha256'],'questions':[
        {'id':'q1','question':'When does the password reset expire?','relevant_passage_ids':['guide.md#p001']},
        {'id':'q2','question':'zzzzunseentoken','relevant_passage_ids':['setup.txt#p001']}
    ]}).encode(),c)

def test_paragraph_ids_bom_crlf_and_determinism():
    c=build_corpus([('b.md',b'\xef\xbb\xbf one\r\n\r\n two '),('a.txt',b'Three')])
    assert [p['id'] for p in c['passages']]==['a.txt#p001','b.md#p001','b.md#p002']
    assert c['passages'][1]['text']=='one'
    assert c==build_corpus([('a.txt',b'Three'),('b.md',b'\xef\xbb\xbf one\r\n\r\n two ')])
    assert c['sha256']!=build_corpus([('b.md',b'Changed'),('a.txt',b'Three')])['sha256']

@pytest.mark.parametrize('files',[
 [],[('x.pdf',b'abc')],[('../x.txt',b'abc')],[('x.txt',b'\xff')],
 [('x.txt',b' ')],[('x.txt',b'a'),('x.txt',b'b')],[('x.txt',b'a'*20001)],
 [('x.txt',b'x\n\n'*501)],[('x.txt',b'x'*(5*1024*1024+1))]])
def test_bad_corpora(files):
    with pytest.raises(ValueError):build_corpus(files)

def test_annotation_validation_and_stale_fingerprint():
    c=corpus();q=questions(c);assert len(q['questions'])==2 and not q['warnings']
    good={'id':'q','question':'why','relevant_passage_ids':['guide.md#p001']}
    for item in [dict(good,relevant_passage_ids=[]),dict(good,relevant_passage_ids=['missing']),
                 dict(good,relevant_passage_ids=['guide.md#p001']*2),dict(good,question=' ')]:
        with pytest.raises(ValueError):parse_questions(json.dumps([item]).encode(),c)
    with pytest.raises(ValueError):parse_questions(json.dumps([good,good]).encode(),c)
    with pytest.raises(ValueError):parse_questions(json.dumps({'corpus_sha256':'stale','questions':[good]}).encode(),c)
    assert parse_questions(json.dumps([good]).encode(),c)['warnings']

def test_ranking_excludes_zero_scores_and_ties_use_ids():
    p=corpus()['passages']
    result=rank_scores(p,[.5,.5,0],3)
    assert [r['passage_id'] for r in result]==['guide.md#p001','guide.md#p002']
    assert rank_scores(p,[0,0,0],3)==[]
    with pytest.raises(ValueError):rank_scores(p,[np.nan,1,2],2)
    with pytest.raises(ValueError):rank_scores(p,[1,2,3],0)
    with pytest.raises(ValueError):rank_scores(p,[1,2],1)

def test_metrics_hand_calculation_and_no_results():
    ranked=[{'passage_id':'a','score':.9},{'passage_id':'b','score':.8},{'passage_id':'c','score':.7}]
    m=query_metrics(ranked,['b','c'],2)
    assert m=={'recall_at_k':.5,'reciprocal_rank_at_k':.5,'hit_at_k':1,'missed_relevant_ids':['c']}
    assert query_metrics([],['a'],3)['reciprocal_rank_at_k']==0

def test_tfidf_benchmark_measures_no_match_as_miss():
    report=run_benchmark(corpus(),questions(),methods=['tfidf'],k=1)
    result=report['methods']['tfidf']
    assert result['summary']['recall_at_k']==.5
    assert result['summary']['mrr_at_k']==.5
    assert result['queries'][1]['results']==[]
    assert result['queries'][1]['metrics']['missed_relevant_ids']==['setup.txt#p001']
    assert result['timing']['query_ms_median']>=0

def test_empty_vocabulary_has_actionable_error():
    c=build_corpus([('x.txt',b'!!!')])
    q=parse_questions(b'[{"id":"q","question":"word","relevant_passage_ids":["x.txt#p001"]}]',c)
    with pytest.raises(ValueError,match='vocabulary'):run_benchmark(c,q,methods=['tfidf'])

@pytest.mark.model
def test_real_model_finds_paraphrase():
    c=build_corpus([('x.txt',b'To recover your account, request a password reset email.\n\nCharts can be exported in SVG format.')])
    q=parse_questions(b'[{"id":"q","question":"I forgot my login secret. How can I regain access?","relevant_passage_ids":["x.txt#p001"]}]',c)
    r=run_benchmark(c,q,methods=['minilm'],k=1)
    assert r['methods']['minilm']['queries'][0]['results'][0]['passage_id']=='x.txt#p001'
    assert r['methods']['minilm']['summary']['mrr_at_k']==1

def test_dataset_provenance_survives_evaluation():
    c=corpus()
    q=parse_questions(json.dumps({'description':'Synthetic fixture only','questions':questions(c)['questions']}).encode(),c)
    assert run_benchmark(c,q,['tfidf'])['dataset_description']=='Synthetic fixture only'
    with pytest.raises(ValueError,match='Description'):
        parse_questions(json.dumps({'description':[], 'questions':q['questions']}).encode(),c)

@pytest.mark.model
def test_model_truncation_is_disclosed():
    from retrievalbench.model import encode_texts
    vectors, info=encode_texts(['word '*400],['long-passage'])
    assert vectors.shape==(1,384)
    assert np.isclose(np.linalg.norm(vectors[0]),1)
    assert info['truncated_ids']==['long-passage']
