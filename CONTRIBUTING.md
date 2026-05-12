# Contributing

Thank you for your interest in contributing to MLCLI!

## Setup

```bash
git clone https://github.com/your-org/mlcli.git
cd mlcli
pip install -e ".[dev]"
pre-commit install --hook-type commit-msg
```

## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/). All commit messages must follow the format:

```
<type>(<scope>): <description>

[optional body]
```

**Types:**

| Type | When to use |
|---|---|
| `feat` | New feature (triggers minor version bump) |
| `fix` | Bug fix (triggers patch bump) |
| `perf` | Performance improvement (triggers patch bump) |
| `refactor` | Code refactor with no behaviour change |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `chore` | Maintenance, dependencies, tooling |
| `ci` | CI/CD changes |

**Examples:**
```
feat(training): add cosine annealing with warm restarts
fix(checkpoint): prevent emergency save from overwriting best_model.pt
docs(readme): add grid search example
chore(deps): bump torch to 2.3.0
```

A breaking change adds `!` after the type or `BREAKING CHANGE:` in the footer, which triggers a major version bump.

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes and add/update tests where applicable
3. Ensure `ruff check src/` and `pytest` pass locally
4. Open a PR — CI will run lint + tests automatically

## Releases

Releases are fully automated. Merging to `main` triggers `python-semantic-release`, which:
- Reads commits since the last tag
- Bumps the version in `pyproject.toml` following semver
- Generates/updates `CHANGELOG.md`
- Creates a GitHub Release and git tag

No manual version bumping needed.
