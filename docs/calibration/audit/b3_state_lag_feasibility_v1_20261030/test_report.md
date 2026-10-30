# A1 synthetic and historical parity test run

- Result: PASS
- Command: `D:\anaconda3\envs\torch\python.exe -m compileall -q experiments/calibration/stage1a_b3_mechanism_ablation.py tests/calibration/test_b3_mechanism_ablation.py`
- Compile result: exit 0
- Command: `D:\anaconda3\envs\torch\python.exe -m unittest discover -s tests/calibration -p test_b3_mechanism_ablation.py -v`
- Test result: 14 tests passed, including the frozen A0 verifier and exact historical A2 parity.
- Pytest is not installed in the supplied Python environment; no package was installed. The same test module is standard-library `unittest` compatible.
- Scope: deterministic synthetic recurrence, lag mapping, causal order, frozen Delta, checkpoint restore/rejection, evidence/hash fail-closed behavior, unchanged historical source check, and metadata-only frozen A0/A2 verification. No formal outcome evaluation was run.

See `unittest_output.txt` for the captured test output.
