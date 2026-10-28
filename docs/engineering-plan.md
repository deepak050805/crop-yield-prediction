# Engineering plan

Starting revision: `5968a14aa51f80ec56d6425ba3fc68f6e4b0d2dd` (`main`). The working tree was clean.

Baseline verification: all 5 backend tests and 9 frontend tests pass. The application
currently loads a CSV and pickle at import time; HTTP handling, validation, and
prediction share one module. Existing year-separated evaluation and the crop-mean
baseline must remain visible and comparable.

## Priorities and acceptance criteria

1. Separate application construction, HTTP validation, data access, and prediction
   without breaking existing imports or response fields. Test independent app
   instances and startup failures.
2. Reject ambiguous, oversized, malformed, and non-finite requests. Use consistent
   JSON errors and safe production logging. Add liveness and readiness endpoints.
3. Validate the dataset and inference schema, verify artifact integrity, and keep
   temporal preprocessing and selection independent of held-out years. Publish
   reproducible metadata and group-level comparisons without accuracy claims.
4. Fix observable UI failure modes, harden CSV export, and improve keyboard,
   chart, and small-screen behavior with deterministic browser-level tests.
5. Verify dependency installation, production container startup, health checks,
   local development, and automated checks. Document any unavailable verification.
6. Run regression, security, data, and evaluation checks; record actual outcomes,
   remaining deployment prerequisites, and the new commit count.

Implementation is divided into independently reviewable changes with their
relevant tests. Git operations use explicit paths and a shared commit lock.
Existing history is preserved; nothing is pushed. The requested 100 commits is
conditional on useful work and must not justify empty or artificial changes.

## Scope

Keep Flask, pandas, scikit-learn, Jinja, plain JavaScript, and Chart.js. Do not add
authentication, databases, application caches, or background infrastructure.
Historical observations and predictions remain distinct. The undocumented data
provenance and marginal improvement over a crop-average baseline remain explicit
limitations, regardless of engineering improvements.
