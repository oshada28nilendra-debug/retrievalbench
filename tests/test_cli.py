import json
import subprocess
import sys

def test_cli_manifest_and_error(tmp_path):
    corpus=tmp_path/'docs';corpus.mkdir();(corpus/'test.md').write_text('A reproducible passage.')
    output=tmp_path/'manifest.json'
    p=subprocess.run([sys.executable,'-m','retrievalbench.cli','manifest','--corpus',str(corpus),'--output',str(output)],capture_output=True,text=True)
    assert p.returncode==0 and json.loads(output.read_text())['passages'][0]['id']=='test.md#p001'
    p=subprocess.run([sys.executable,'-m','retrievalbench.cli','manifest','--corpus',str(tmp_path/'missing'),'--output',str(output)],capture_output=True,text=True)
    assert p.returncode==2 and '--corpus must be a directory' in p.stderr
