# Privacy and safety

Your messages are deeply personal. This project is designed around one simple boundary: source
code may be shared, but your messages, derived dataset, tokenizer, checkpoints, and trained model
must remain private.

## The local data path

The Mac exporter opens `~/Library/Messages/chat.db` read-only and creates a consistent snapshot with
SQLite's backup API. Attachments are not opened or copied. The snapshot is transferred privately to
Windows, where its SHA-256 fingerprint and SQLite integrity are verified before processing.

All later steps run locally on the Windows PC. The project has no command that uploads data or sends
an iMessage.

## Protections in the pipeline

- Raw participant handles and chat identifiers are replaced with keyed HMAC pseudonyms before
  extracted JSONL is written.
- URLs, email addresses, and phone-number-shaped strings are redacted by default.
- Reactions, system events, deleted messages, and attachment-only rows are filtered.
- Ordinary commands print aggregate counts and metrics, not message bodies.
- Chronological splits and guard bands reduce train/test leakage.
- Training targets are restricted to outgoing `me` turns. Incoming text remains context.
- Memorization probes store aggregate overlap counts, never matching private passages.
- Tests use synthetic conversations only.
- `work/`, `outputs/`, databases, arrays, tokenizers, and model weights are excluded by `.gitignore`.

## Pseudonymized does not mean anonymous

Pseudonymization hides obvious database identities, but conversation wording can still identify
people or events. A trained model can also memorize text. Treat every item in `work/` and `outputs/`
as sensitive even after the privacy audit passes.

## Never share these files

```text
chat.db
manifest.json
work/
outputs/
*.db
*.npy
*.safetensors
optimizer.pt
```

Do not attach them to GitHub issues, Discord conversations, emails, or bug reports. If you need help,
share the aggregate command output after checking it for personal paths. The issue templates ask for
safe diagnostic information only.

## Device protection

Use FileVault on the Mac and BitLocker or Windows Device Encryption on the PC. Transfer snapshots by
USB or a trusted local method. Delete extra copies from removable media and temporary folders after
the Windows import succeeds.

If you use a shared Windows account, school computer, workplace computer, cloud-synchronized folder,
or remotely managed PC, do not process your messages there.

## Before publishing source code

Run:

```powershell
git status --short
git ls-files work outputs
git ls-files "*.db" "*.npy" "*.safetensors"
```

The last two commands must print nothing. If private data was ever committed, adding it to
`.gitignore` afterward is not enough because it remains in Git history. Stop, remove it from history,
and create a clean repository before publishing.

Privacy checks reduce risk, but no local language-model project can guarantee that trained weights
contain no memorized text. Keep the final model private.
