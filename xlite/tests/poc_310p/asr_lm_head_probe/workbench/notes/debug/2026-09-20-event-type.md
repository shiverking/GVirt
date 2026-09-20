# 2026-09-20 — LM Head event_t compilation failure

Symptom: CANN 9.1 beta1 rejects eight event_t member initializers of the
form `= 0` in asr_m200_lm_head_fp16.cpp; no device execution occurred.

Static analysis: Init assigns every event from FetchEventID before Process
uses it. The integer defaults are unnecessary and invalid for event_t.
The existing attention MMAD probe also owns a TPipe for dynamic events;
the new LM Head omitted that member.

Fix: remove integer defaults, add the TPipe owner, preserve all eight
dynamic event assignments and the existing SetFlag/WaitFlag schedule.
Add an integer-initializer check to the existing ASR source gate.

Verification: source checks only on this host; CANN compilation and M=1/8/20
device validation remain required via test_asr_lm_head_probe.sh.
No device visibility or mapping changes.

Lesson: event_t is not an implicitly integer-initializable handle. Acquire
events from an initialized pipe, with every assignment preceding use.
