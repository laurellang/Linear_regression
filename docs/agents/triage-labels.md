# Triage labels

The skills' canonical triage roles map directly to GitHub labels.

| Skill role | GitHub label | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Maintainer needs to evaluate the issue |
| `needs-info` | `needs-info` | Waiting for more information |
| `ready-for-agent` | `ready-for-agent` | Fully specified and ready for an agent |
| `ready-for-human` | `ready-for-human` | Requires human implementation or judgment |
| `wontfix` | `wontfix` | Will not be actioned |

Each triaged issue should have one category (`bug` or `enhancement`) and one
state label. Do not apply triage labels to tickets created by `/to-tickets`
except the specified `ready-for-agent` state.
