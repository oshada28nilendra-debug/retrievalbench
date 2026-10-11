import io
import pytest
from retrievalbench.web import create_app

@pytest.fixture
def client():
    return create_app().test_client()

def post(client,path,data):
    return client.post(path,data=data,headers={'X-RetrievalBench':'local'})

def test_home_and_request_guards(client):
    response=client.get('/'); assert response.status_code==200
    assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
    assert client.get('/',headers={'Host':'evil.example'}).status_code==403
    assert client.post('/api/run').status_code==403

def test_sample_baseline(client):
    r=post(client,'/api/run',{'sample':'1','methods':'tfidf','k':'3'})
    assert r.status_code==200 and r.json['question_count']==16
    assert r.json['dataset'].startswith('Synthetic')

def test_import_manifest_and_evaluate(client):
    r=post(client,'/api/manifest',{'documents':(io.BytesIO(b'Hello world'),'x.md')})
    assert r.status_code==200 and r.json['passages'][0]['id']=='x.md#p001'
    r=post(client,'/api/run',{'documents':(io.BytesIO(b'Hello world'),'x.md'),
        'questions':(io.BytesIO(b'[{"id":"q","question":"hello","relevant_passage_ids":["x.md#p001"]}]'),'q.json'),
        'methods':'tfidf','k':'1'})
    assert r.status_code==200 and r.json['methods']['tfidf']['summary']['mrr_at_k']==1

@pytest.mark.parametrize('data',[{}, {'sample':'1','k':'x'}, {'sample':'1','k':'0'}, {'sample':'1','methods':'bad'}])
def test_bad_requests(client,data):
    assert post(client,'/api/run',data).status_code==400

def test_html_is_text_not_markup(client):
    r=post(client,'/api/manifest',{'documents':(io.BytesIO(b'<script>alert(1)</script>'),'x.md')})
    assert r.json['passages'][0]['text']=='<script>alert(1)</script>'

def test_oversized_upload(client):
    r=client.post('/api/manifest',data=b'x'*(7*1024*1024+1),headers={'X-RetrievalBench':'local'},content_type='application/octet-stream')
    assert r.status_code==413
