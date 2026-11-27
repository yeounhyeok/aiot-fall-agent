"""Bounded single-event controller. Keep logs free of speech and sensor traces."""

QUESTIONS = {'initial_check': '괜찮으세요? 지금 상태를 말씀해 주세요.', 'clarify_status': '잘 듣지 못했어요. 지금 괜찮으신가요?', 'offer_help': '도움이 필요하신가요?'}


def route(decision, transcript):
    # Actions are allowlisted, never executable names supplied by a model.
    if not isinstance(decision, dict) or set(decision) != {'evidence','call','next_prompt'}:
        return None
    ev, call, prompt = (decision[k] for k in ('evidence','call','next_prompt'))
    if not isinstance(ev, str) or not ev.strip() or ev not in transcript:
        return None
    for name, reason in [('raise_alert','help_requested'), ('close_event','false_alarm'), ('ask_checkin','unclear_reply')]:
        if call == {'name':name, 'arguments':{'reason':reason}}:
            if name == 'ask_checkin':
                return prompt if prompt in ('clarify_status','repeat_last','offer_help') else None
            return name if prompt is None else None
    return None


def check_event(sensor, io, *, max_turns=4, log=print, alerts=False):
    """Return one outcome. An unresolved episode is never called a false alarm."""
    if not 1 <= max_turns <= 10:
        raise ValueError('max_turns outside 1..10')
    question, history = 'initial_check', []
    for turn in range(1, max_turns + 1):
        io.say(question)  # finish speaker playback before opening the microphone
        text = io.listen().strip()
        observation = sensor.snapshot()  # fresh after this reply, not the triggering window
        history.append({'sensor':observation,'device':QUESTIONS[question],'transcript':text})
        history = history[-2:]
        decision = None
        if text:
            try:
                decision = io.decide(history)
            except (ValueError, RuntimeError, OSError):
                log('policy_unavailable')
        action = route(decision,text) or 'clarify_status'
        log(f'turn={turn} route={action}')
        if action == 'raise_alert':
            # Speaker and webhook are independent. Failed playback must not suppress the alert.
            try:
                io.say(action)
            finally:
                if alerts:
                    try:
                        status, _ = io.notify(decision)
                    except (ValueError, RuntimeError, OSError):
                        status = 'unconfirmed'
                    log('notification='+status)
                else:
                    log('notification=disabled')
            return action
        if action == 'close_event':
            io.say(action)
            return action
        if action != 'repeat_last':
            question = action
    log('unresolved: turn limit reached; no close or external dispatch')
    return 'unresolved'


def monitor(sensor, io, *, one=False, max_turns=4, alerts=False, log=print):
    while True:
        log('monitoring')
        sensor.wait_candidate()
        log('fall_candidate')
        result = check_event(sensor,io,max_turns=max_turns,alerts=alerts,log=log)
        sensor.rearm()  # do not open a second episode for the same sustained score
        log('rearmed after '+result)
        if one:
            return result
