"""SWE-bench-derived RCA evaluation pipeline for RootTrace.

The pipeline turns local SWE-bench Verified public metadata into RootTrace
incidents, runs each case in a disposable workspace bounded to the base
commit, and evaluates file localization against gold non-test files after
the RCA run completes. Gold data is evaluator-only and never enters the RCA
runtime.
"""
