import argparse
import json
from pathlib import Path
from .core import build_corpus,parse_questions,run_benchmark

def main():
    parser=argparse.ArgumentParser(description='Local passage retrieval evaluation')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('download'); serve=sub.add_parser('serve'); serve.add_argument('--port',type=int,default=8766)
    for name in ['manifest','evaluate']:
        p=sub.add_parser(name);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        if name=='evaluate':
            p.add_argument('--questions',type=Path,required=True);p.add_argument('--k',type=int,default=3)
            p.add_argument('--methods',nargs='+',choices=['tfidf','minilm'],default=['tfidf','minilm'])
    args=parser.parse_args()
    try:
        if args.command=='download':
            from .model import _load
            _load();print('Pinned MiniLM model cached.');return
        if args.command=='serve':
            from .web import create_app
            create_app().run(host='127.0.0.1',port=args.port,debug=False,threaded=False);return
        if not args.corpus.is_dir(): raise ValueError('--corpus must be a directory.')
        corpus=build_corpus([(p.name,p.read_bytes()) for p in args.corpus.iterdir() if p.suffix in {'.md','.txt'} and p.is_file()])
        result=corpus if args.command=='manifest' else run_benchmark(corpus,parse_questions(args.questions.read_bytes(),corpus),args.methods,args.k)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(args.output)
    except (ValueError,RuntimeError,OSError) as exc: parser.exit(2,f'Error: {exc}\n')

if __name__=='__main__':main()
