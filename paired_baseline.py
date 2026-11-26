"""Timed, matched OFF/ON pilot. Only final actions are scored and saved."""
import json, sys, time, statistics, hashlib
from pathlib import Path
import torch, transformers
from harness import rows, validate, score

snap, data, out = map(Path, sys.argv[1:]); out.mkdir(exist_ok=False)
cases = rows(data); validate(cases)
tok = transformers.AutoTokenizer.from_pretrained(snap, local_files_only=True)
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(snap, local_files_only=True, dtype=torch.bfloat16, device_map={'':'cuda:0'}).eval()
system = ('한국어 낙상 확인 장치의 모의 실험입니다. 직전 질문에 대한 답을 센서 상태와 함께 해석하세요. 도움 요청은 raise_alert(reason="help_requested"), 다시 듣기 요청·불명확한 답·센서와 답의 충돌은 ask_checkin(reason="unclear_reply"), 명확한 오경보나 정상 움직임과 일치하는 안심 응답은 close_event(reason="false_alarm")입니다. 최종 답변은 {"name":"함수명","arguments":{"reason":"사유"}} JSON 하나입니다.')
settings=dict(do_sample=True,temperature=0.7,top_p=0.8,top_k=20,max_new_tokens=1024)
endtok=tok.convert_tokens_to_ids('</think>'); assert tok.convert_ids_to_tokens(endtok)=='</think>'
results={False:[],True:[]}; stats={}
with torch.inference_mode():
    warm=tok('준비',return_tensors='pt').to('cuda:0'); model.generate(**warm,max_new_tokens=2,do_sample=False)
    for i,c in enumerate(cases):
        msg=[{'role':'system','content':system},{'role':'user','content':json.dumps({k:c[k] for k in ['sensor','question','reply']},ensure_ascii=False)}]
        for flag in ((False,True) if i%2==0 else (True,False)):
            prompt=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=flag); assert prompt.rstrip().endswith('<think>') == flag
            x=tok(prompt,return_tensors='pt').to('cuda:0');torch.manual_seed(42+i);torch.cuda.synchronize();t=time.monotonic();y=model.generate(**x,**settings)[0,x.input_ids.shape[1]:];torch.cuda.synchronize();sec=time.monotonic()-t
            ids=y.tolist();closed=endtok in ids;final=ids[ids.index(endtok)+1:] if closed else ([] if flag else ids);text=tok.decode(final,skip_special_tokens=True).strip()
            try: action=json.loads(text)
            except ValueError: action=None
            r=dict(id=c['id'],family=c['family'],call=action,final=text,tokens=len(ids),cap=len(ids)==settings['max_new_tokens'],seconds=round(sec,3),exact=action==c['gold'],think_closed=closed); results[flag].append(r)
            with (out/('on.jsonl' if flag else 'off.jsonl')).open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
for flag,p in results.items():
    stats['on' if flag else 'off']={**score(cases,p),'pairs_both':sum(all(r['exact'] for r in p if r['family']==g) for g in set(c['family'] for c in cases)),'median_seconds':statistics.median(r['seconds'] for r in p),'median_tokens':statistics.median(r['tokens'] for r in p),'cap_hits':sum(r['cap'] for r in p),'invalid':sum(not isinstance(r['call'],dict) for r in p),'wrong':[r['id'] for r in p if not r['exact']]}
report=dict(model='Qwen/Qwen3.5-2B',revision=snap.name,host='vi005',gpu=torch.cuda.get_device_name(),dtype='BF16',torch=torch.__version__,transformers=transformers.__version__,settings=settings,seed='42+case_index',cases_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),stats=stats);(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(stats),flush=True)
