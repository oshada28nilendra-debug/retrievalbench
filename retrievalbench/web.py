"""Loopback-only, stateless local evaluation UI."""
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import HTTPException
from .core import build_corpus, parse_questions, run_benchmark

SAMPLE=Path(__file__).parent/'sample'

def sample_inputs():
    return build_corpus([(p.name,p.read_bytes()) for p in sorted((SAMPLE/'documents').glob('*.md'))]), (SAMPLE/'questions.json').read_bytes()

def create_app():
    app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=7*1024*1024
    @app.before_request
    def guard():
        if request.host.split(':')[0] not in {'localhost','127.0.0.1'}:
            return jsonify(error='Use localhost or 127.0.0.1.'),403
        if request.method=='POST' and request.headers.get('X-RetrievalBench')!='local':
            return jsonify(error='Missing local request header.'),403
    @app.after_request
    def headers(response):
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'"
        response.headers['X-Content-Type-Options']='nosniff'; return response
    @app.errorhandler(ValueError)
    def invalid(e): return jsonify(error=str(e)),400
    @app.errorhandler(RuntimeError)
    def unavailable(e): return jsonify(error=str(e)),503
    @app.errorhandler(HTTPException)
    def http_error(e): return jsonify(error=e.description),e.code
    @app.get('/')
    def index(): return render_template('index.html')
    def uploaded():
        return build_corpus([(f.filename,f.read()) for f in request.files.getlist('documents')])
    @app.post('/api/manifest')
    def manifest(): return jsonify(uploaded())
    @app.post('/api/run')
    def run():
        if request.form.get('sample')=='1': corpus,raw=sample_inputs()
        else:
            corpus=uploaded(); f=request.files.get('questions')
            if not f: raise ValueError('Upload annotated questions JSON.')
            raw=f.read()
        try: k=int(request.form.get('k','3'))
        except ValueError as exc: raise ValueError('k must be an integer.') from exc
        methods=request.form.getlist('methods') or ['tfidf','minilm']
        report=run_benchmark(corpus,parse_questions(raw,corpus),methods,k)
        report['dataset']='Synthetic product documentation demo' if request.form.get('sample')=='1' else 'User supplied'
        return jsonify(report)
    return app
