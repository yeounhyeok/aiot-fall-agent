"""Offline GPU pilot for demo cases only. Usage: python run_qwen_smoke.py SNAPSHOT OUTDIR"""
import json, sys, time, hashlib
from pathlib import Path
import transformers, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from harness import rows, validate, score


def main():
    snapshot, out = map(Path, sys.argv[1:])
    cases_path = Path('examples/cases.jsonl')
    cases = rows(cases_path); validate(cases)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda:0'}).eval()
    rules = ('한국어 낙상 확인 장치의 오프라인 시뮬레이션입니다. 센서와 대화를 종합해 다음 호출 1개를 JSON 객체로만 출력하세요. '
             '허용 함수/인자: raise_alert(reason="help_requested"), ask_checkin(reason="unclear_reply"), close_event(reason="false_alarm"). '
             '출력 형식: {"name":"함수명","arguments":{"reason":"사유"}}. 설명이나 코드 펜스는 출력하지 마세요.')
    pred = []; times = []; settings = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=96)
    with torch.inference_mode(), (out/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for c in cases:
            msg = [{'role':'system','content':rules},{'role':'user','content':json.dumps({k:c[k] for k in ['sensor','question','reply']},ensure_ascii=False)}]
            text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inp = tok(text, return_tensors='pt').to('cuda:0'); torch.cuda.synchronize(); t=time.monotonic()
            ids = model.generate(**inp, **settings)[0,inp.input_ids.shape[1]:]; torch.cuda.synchronize(); times.append(round(time.monotonic()-t,3))
            raw = tok.decode(ids,skip_special_tokens=True).strip()
            try: call = json.loads(raw)
            except ValueError: call = None
            p={'id':c['id'],'call':call,'raw':raw};pred.append(p);f.write(json.dumps(p,ensure_ascii=False)+'\n');f.flush()
    report = dict(model='Qwen/Qwen3-4B', revision=snapshot.name, protocol='untrained-on-task demo; non-thinking; batch=1', seed=42, settings=settings, torch=torch.__version__,transformers=transformers.__version__, cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(), peak_allocated_GiB=round(torch.cuda.max_memory_allocated()/2**30,2),generation_seconds=times,score=score(cases,pred))
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__': main()
