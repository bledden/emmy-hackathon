"""Path-sensitive ownership of synchronous CTA barrier resources.

This is a compile-time proof, not emitted synchronization. A full-CTA barrier
ends preceding phases only after checking its own ID cannot overlap one.
State maps an ID to the set of possible last participant cohorts since such
a rendezvous. Branch joins preserve every possibility, including no change.
"""


def join(*states):
    result = {}
    for state in states:
        for identifier, groups in state.items():
            result[identifier] = result.get(identifier, frozenset()) | groups
    return result


def validate(events, state=None):
    state = dict(state or {})
    for event in events:
        tag = event[0]
        if tag == 'barrier':
            _, identifier, cohort, full = event
            previous = state.get(identifier, frozenset())
            if previous - {cohort}:
                raise ValueError('barrier ID reused by different participant groups without a proven full-CTA rendezvous')
            if full:
                state = {}
            else:
                state[identifier] = frozenset({cohort})
        elif tag == 'choice':
            state = join(*(validate(arm, state) for arm in event[1]))
        elif tag == 'loop':
            _, body, may_skip, repetitions = event
            if repetitions == 0:
                continue
            entry = state
            state = validate(body, entry)
            if repetitions > 1:
                # Each ID's transfer is S -> C or S -> S union C. Once the
                # first iteration has established C, another iteration checks
                # the backedge and reaches the same exit state. No unrolling
                # proportional to the runtime trip count is needed.
                second = validate(body, state)
                assert second == state, 'non-idempotent barrier transfer'
            if may_skip:
                state = join(entry, state)
        else:
            raise AssertionError('unknown barrier event')
    return state
