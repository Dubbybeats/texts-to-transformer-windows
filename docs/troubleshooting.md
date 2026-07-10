# Troubleshooting

Most setup problems are ordinary permission, path, driver, or download issues. Your messages are not
damaged when a command stops. Keep the original snapshot, read the full error, and work through the
matching section below.

Never paste private message text, `chat.db`, or files from `work/` and `outputs/` into a public issue.

## PowerShell says script execution is disabled

Run this in the same PowerShell window, then retry setup:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

The change lasts only until that PowerShell window closes.

## `uv` is not recognized

Install uv with:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close PowerShell completely, reopen it inside the project folder, and run `uv --version`. If it is
still missing, restart Windows and review the
[official uv installation options](https://docs.astral.sh/uv/getting-started/installation/).

## Setup appears stuck during a large download

PyTorch and the NVIDIA runtime can total several gigabytes. Leave the first setup running while
downloads continue. If it fails due to an interrupted connection, rerun `setup_windows.ps1`; uv will
reuse completed downloads.

## `cuda_available` is `false` on Windows

Update the NVIDIA graphics driver, delete the `.venv` folder inside the project, and rerun:

```powershell
uv sync
uv run imessage-cuda doctor
```

This project selects PyTorch's official CUDA 12.8 package on Windows. You do not need to install the
full CUDA developer toolkit. The doctor report must show `cuda_available: true` before real training.

If Task Manager does not show an NVIDIA GPU under **Performance**, the computer may not have a
supported card or its driver may be missing.

## The Mac says `Could not read Messages`

Give Terminal Full Disk Access:

```text
System Settings > Privacy & Security > Full Disk Access
```

Enable Terminal, quit it completely with `Command + Q`, reopen it, and rerun the exporter. Do not
loosen permissions on `~/Library/Messages` and do not manually copy the live database.

## The Mac says the Messages database was not found

Open Messages and confirm that the correct Apple Account is signed in and conversations are visible.
The normal database location is `~/Library/Messages/chat.db`.

## The snapshot reports that the source changed

A new message or iCloud synchronization changed the database during export. Wait briefly and run the
Mac exporter again. The refusal protects snapshot consistency.

## Windows rejects the snapshot hash

Transfer both `chat.db` and `manifest.json` again from the same Mac export folder. Do not edit, open,
rename internally, or recompress the database. A hash mismatch usually means the files came from
different export attempts or the transfer was incomplete.

## `prepare` stops on body recovery

Do not lower the recovery threshold immediately. Apple sometimes stores visible text in the typed
`attributedBody` field rather than the ordinary `text` column. The pipeline stops when too many
eligible messages cannot be decoded safely.

Open a bug report with aggregate extraction counts, your macOS version, and your Windows version.
Do not attach the database or paste messages.

If generated output contains strings such as `__kIMMessagePartAttributeName`, delete the derived
`work` and `outputs` folders, update the project, rerun preparation, and confirm the privacy audit
passes before retraining.

## The privacy audit fails

Do not continue to training. The report contains aggregate failure counts without message text.
Keep the verified snapshot, update or repair the pipeline, rebuild the derived dataset, and rerun:

```powershell
uv run imessage-cuda privacy-audit
```

Continue only when `passed` is `true`.

## The corpus has fewer than one million tokens

This is not an installation failure. The synced history simply does not contain enough unique text
for the project's from-scratch safety threshold. A much smaller corpus would encourage memorization.

Let Messages finish syncing older history on the Mac and export again if more conversations should
exist. Do not duplicate messages or lower the gate just to force training.

## CUDA runs out of memory

Close games, browsers using hardware acceleration, video tools, and other GPU-heavy applications.
Open the selected file in `configs/`, reduce `batch_size` by half, and rerun training. For example,
change `32` to `16`, then `8` if needed. Keep the architecture values unchanged.

## Training was interrupted

Resume from the atomic `last` checkpoint. Use the same config selected for the original run:

```powershell
uv run imessage-cuda train `
  --config configs/model-1m.yaml `
  --data work/tokens `
  --tokenizer outputs/tokenizer `
  --output outputs/runs/my-model `
  --resume-from outputs/runs/my-model/last `
  --device cuda
```

## The model repeats phrases or sounds incoherent

Small from-scratch models have narrow abilities. Try conservative generation settings:

```powershell
uv run imessage-cuda chat `
  --model outputs/final `
  --temperature 0.5 `
  --top-p 0.8 `
  --max-new-tokens 24 `
  --repetition-penalty 1.2
```

Sampling can make replies calmer and shorter, but it cannot add reasoning or knowledge the model
never learned.

## Asking for help safely

Use the GitHub bug template and include only:

- Windows version
- NVIDIA GPU model
- Python and PyTorch versions
- The command that failed
- Aggregate error output with usernames and personal paths removed

Never share conversation text, databases, `work/`, `outputs/`, tokenizers, checkpoints, or model
weights. A synthetic reproduction is always preferred.
