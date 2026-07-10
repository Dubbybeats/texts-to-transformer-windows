# Contributing

Thank you for helping make local iMessage experimentation more accessible to Windows users.
Contributions do not need to be large. Clearer instructions, better error messages, accessibility
improvements, support for another NVIDIA card, and careful tests are all valuable.

## The most important rule

Never include real message data in an issue, commit, test, screenshot, log, pull request, release,
or example. This includes pseudonymized datasets and trained models. Use the synthetic fixtures in
`tests/fixtures/` when reproducing a problem.

## Good first contributions

- Clarify a confusing tutorial step.
- Improve a Windows or macOS error message.
- Add a synthetic regression test.
- Report whether setup worked on a different NVIDIA GPU.
- Improve keyboard navigation, readability, or screen-reader friendliness.
- Add a privacy-preserving diagnostic check.

## Development setup

```powershell
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The test suite must run without any real Messages database. Keep `work/` and `outputs/` empty of
tracked content.

Before committing, run:

```powershell
git diff --check
git ls-files work outputs
git ls-files "*.db" "*.npy" "*.safetensors"
```

The final two commands must print nothing.

## Opening an issue

Use the provided bug or feature template. Describe what you expected, what happened, your Windows
version, GPU model, Python version, and the non-private portion of `imessage-cuda doctor` output.
Remove usernames and personal file paths before posting.

## Pull requests

Keep changes focused and explain the user-facing reason for them. Add or update synthetic tests when
behavior changes. Update the beginner guide when a command, default, or required step changes.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
