# Tests (rebuild in progress)

This tree is the declarative contract suite rebuilt around **per-family declarations**: a contributor states a
function's whole testing contract by calling one per-family suite function (`suite_pnl` and its twins) with closed
vocabularies plus a hand-written reference oracle, and the checks in `support/rungs.py` are derived from that single
declaration. It runs alongside the live `tests/` suite during the rebuild; the normative suite law — the authoring
canon, the three-question criterion, and the structural freeze — is written here at cutover, when `tests_new/`
replaces `tests/`. Until then, `tests/README.md` remains the suite's normative document.
