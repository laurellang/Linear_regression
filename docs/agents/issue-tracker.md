# Issue tracker: GitHub

Specs and implementation tasks for this repository live in GitHub Issues at
`laurellang/machine-learning-hw1`. Use the `gh` CLI from the repository root.

## Conventions

- Create: `gh issue create --title "..." --body-file <file>`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open --json number,title,labels,assignees`.
- Discuss: `gh issue comment <number> --body-file <file>`.
- Label: `gh issue edit <number> --add-label <label>` or `--remove-label <label>`.
- Claim: `gh issue edit <number> --add-assignee @me`.
- Close: `gh issue close <number>` after acceptance criteria are met.
- Link a child task to a parent spec with `gh issue create --parent <parent>`
  when supported. Otherwise, put `Part of #<parent>` in the task body.
- Express task dependencies using GitHub's native `blocked by` relation when
  supported. Otherwise, put `Blocked by: #<number>` in the task body. A task
  is ready when every blocking issue is closed.

Verify the target with `gh repo view --json nameWithOwner` before creating
issues. Keep each task independently verifiable. Collaborators claim an issue,
commit directly to `main`, and record the verification command and result in
the issue. No pull request is required for collaborator work.

## Pull requests as a triage surface

PRs as a request surface: no. This team normally commits directly to `main`.
Do not treat a PR as an incoming feature request for `/triage`.

## Skill operations

When a skill says "publish to the issue tracker", create a GitHub issue.
When it says "fetch the relevant ticket", read that issue and its comments.
For `/wayfinder`, use a parent issue labelled `wayfinder:map`, child issues for
decision tickets, blocking relations for dependencies, and assignment to claim
work. Record each resolved decision in its ticket before closing it.
