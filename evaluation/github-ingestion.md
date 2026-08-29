# GitHub ingestion evaluation feasibility

## Recommendation

Use `pytest-dev/pytest` as the primary real-GitHub evaluation repository. It
is a Python project with a native GitHub Issues and Pull Requests workflow,
and its contribution guide uses GitHub pull requests and issue numbers in
bug-fix changelog entries. This makes it practical to construct cases with an
issue, a resolving pull request, a pre-fix revision, and changed files through
the GitHub REST API.

Use `cli/cli` only as a secondary ingestion smoke-test source. It has a strong
GitHub-native issue/PR workflow, but it is a Go repository and therefore falls
outside RootTrace's supported Python-repository RCA scope.

Do not use `django/django` as the primary source. Django is Python and has many
pull requests, but its contributor documentation identifies Trac as the ticket
tracker. GitHub pull requests can also have no linked ticket, so an automatic
Issue -> PR dataset would have avoidable linkage gaps.

## Dataset construction

Build a 20-50 case development benchmark from merged bug-fix pull requests:

1. List closed issues and merged pull requests from `pytest-dev/pytest`.
2. Keep a pull request only when its body/title or timeline has one
   unambiguous issue reference, such as `Fixes #123` or a closing timeline
   event. Discard ambiguous, duplicate, documentation-only, and multi-issue
   cases unless manually adjudicated.
3. Fetch the PR detail, comments/reviews, commits, files, and linked issue.
4. Require a merged PR with a non-empty `base.sha`, a non-empty
   `merge_commit_sha`, and at least one non-test Python source file in the
   changed-file set.
5. Persist only bounded public metadata in the input split:
   `repo`, `issue_url`, `issue_number`, `pr_url`, `pr_number`, `base_commit`,
   `merge_commit`, `ground_truth_files`, `issue_title`, and `issue_body`.
   Keep raw patches and test metadata evaluator-only.

Reuse the existing evaluation harness, disposable case workspace, and
localization metrics. A GitHub-case adapter should produce `IncidentInput` and
record `url_parse`, `fetch`, `normalization`, repository-preparation, and
end-to-end ingestion outcomes beside the existing case results. Do not add a
second benchmark runner.

## Ground truth and revision policy

For a merged PR, ground truth is the deduplicated set of non-test source files
returned by the PR files endpoint. The diff remains evaluator-only. The RCA
workspace is always created at the PR's `base.sha`, which is the pre-change
revision. PR head and changed-file metadata are evidence, but the head or
merge commit is never the analyzed checkout.

Reject cases when the base SHA cannot be fetched, the merge is not verifiable,
Issue/PR linkage is ambiguous, or repository preparation cannot prove a
base-only disposable checkout. Record each rejection reason for auditability.

## Operational constraints

Public unauthenticated REST reads share a 60-request-per-hour IP limit, while
authenticated requests generally have a 5,000-request-per-hour limit. Dataset
construction should use authentication when available, bounded pagination,
and a response cache outside the RootTrace package. Rate-limit responses must
fail a case explicitly instead of being retried indefinitely.

References:

- [GitHub REST API rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
- [GitHub REST API issue endpoints](https://docs.github.com/en/rest/issues/issues)
- [GitHub REST API pull request endpoints](https://docs.github.com/en/rest/pulls/pulls)
- [GitHub REST API timeline endpoints](https://docs.github.com/en/rest/issues/timeline)
- [pytest contribution guide](https://github.com/pytest-dev/pytest/blob/main/CONTRIBUTING.rst)
- [Django ticket triaging documentation](https://github.com/django/django/blob/main/docs/internals/contributing/triaging-tickets.txt)
