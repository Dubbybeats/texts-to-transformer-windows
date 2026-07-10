# Windows walkthrough: from zero to your first reply

Welcome. This guide assumes you have never trained a language model before. We will take one step at
a time, and every command is safe to copy as long as you adjust the example file paths when asked.

## What you are about to do

The Mac holds the iMessage database, so it creates one read-only snapshot. You move that snapshot to
your Windows PC, which verifies it, removes obvious identifiers, builds a private dataset, and trains
a tiny model on your NVIDIA GPU.

The project never sends an iMessage. At the end, you type a pretend incoming message into PowerShell
and receive a local reply suggestion.

## Part 1: check the Windows PC

### Check Windows and the NVIDIA GPU

Press `Windows key + R`, type `winver`, and press Enter. Windows 10 or 11, 64-bit, is supported.

Open Task Manager with `Ctrl + Shift + Esc`, choose **Performance**, and select **GPU**. Confirm that
an NVIDIA GPU appears. Update its driver through the NVIDIA app or NVIDIA's driver website before
continuing.

### Put the project in a private location

Download the repository with GitHub's green **Code** button and choose **Download ZIP**, or clone it
with Git if you are comfortable doing so. Extract it somewhere like:

```text
C:\Users\YourName\Documents\texts-to-transformer-windows
```

Avoid Desktop folders synchronized by OneDrive and avoid shared folders. Later steps create private
derived data inside this project.

### Open PowerShell in the correct folder

Open the extracted folder in File Explorer. Click the address bar, type `powershell`, and press Enter.
A blue or black PowerShell window should open with the project folder in its prompt.

### Install uv

`uv` installs the correct Python version and the project's dependencies. In PowerShell, run the
official Windows installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close PowerShell, open it again inside the project folder, and confirm installation:

```powershell
uv --version
```

You should see a version number. If PowerShell says `uv` is not recognized, restart Windows or use
the manual options on the [official uv installation page](https://docs.astral.sh/uv/getting-started/installation/).

### Run the Windows setup

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

The first setup can download several gigabytes of NVIDIA and PyTorch components. That is normal.
The policy change applies only to this PowerShell window.

At the end, look for these lines in the doctor report:

```text
"cuda_available": true
"safe_to_train_on_gpu": true
```

Also confirm that `gpu_name` identifies your NVIDIA card. Stop here and visit
[Troubleshooting](troubleshooting.md) if CUDA is false.

## Part 2: create the safe snapshot on the Mac

The Mac must already show the iMessage history you want to use. Let Messages finish syncing before
you begin.

Copy `tools/export_imessage_snapshot.py` from this project to the Mac. A USB drive is fine for this
small script.

Open **System Settings > Privacy & Security > Full Disk Access**. Add or enable Terminal, then quit
Terminal completely and reopen it.

Move the script to the Mac desktop. In Terminal, run:

```bash
cd ~/Desktop
python3 export_imessage_snapshot.py
```

You should see:

```text
Snapshot ready: /Users/your-name/Desktop/imessage-snapshot
Transfer the entire folder privately to the Windows PC.
```

The folder contains:

```text
imessage-snapshot/
  chat.db
  manifest.json
```

Do not open or edit `chat.db`. Move the whole folder to a USB drive or another private local transfer
method. Do not email it to yourself or post it in a cloud share.

See [Mac export guide](MAC_EXPORT.md) if Terminal reports a permission error.

## Part 3: import the snapshot on Windows

Connect the USB drive or locate the transferred folder. In the examples below it is on drive `D:`.
Your drive letter may be different.

Return to PowerShell in the project folder and run:

```powershell
uv run imessage-cuda import-snapshot `
  --database "D:\imessage-snapshot\chat.db" `
  --manifest "D:\imessage-snapshot\manifest.json"
```

A successful result includes:

```text
"import_verified": true
"quick_check": "ok"
```

The two checks mean the transfer did not alter the database and SQLite considers it structurally
healthy.

## Part 4: build the private dataset

Run:

```powershell
uv run imessage-cuda prepare --config configs/data.yaml
```

This extracts text without opening attachments, replaces participant identities with pseudonyms,
redacts obvious URLs, email addresses, and phone-number-shaped strings, groups conversations, and
creates chronological dataset splits.

Now run the privacy audit:

```powershell
uv run imessage-cuda privacy-audit
```

Continue only when it says:

```text
"passed": true
```

If it fails, do not panic and do not delete the original snapshot. Visit
[Troubleshooting](troubleshooting.md#the-privacy-audit-fails).

## Part 5: teach the project how you write

First, train the local tokenizer:

```powershell
uv run imessage-cuda train-tokenizer `
  --train work/splits/train.jsonl `
  --output outputs/tokenizer `
  --vocab-size 4096
```

Then count the available text and let the safety logic choose a model size:

```powershell
uv run imessage-cuda corpus-stats `
  --splits work/splits `
  --tokenizer outputs/tokenizer `
  --output work/tokens
```

Open `work\reports\model-selection.json` in Notepad. Look for `enough_tokens_to_train` and
`selected`.

If `enough_tokens_to_train` is `false`, your synced history contains fewer than one million training
tokens. The project intentionally stops there because a model trained on too little private text is
more likely to memorize it. This is a safety decision, not something you did wrong.

If `selected` is `model-1m`, use:

```powershell
uv run imessage-cuda train `
  --config configs/model-1m.yaml `
  --data work/tokens `
  --tokenizer outputs/tokenizer `
  --output outputs/runs/my-model `
  --device cuda
```

If `selected` is `model-7m`, change only the first path to `configs/model-7m.yaml`.

Training prints aggregate progress. A lower loss usually means the model is getting better at
predicting your outgoing text. Speed and GPU memory vary by graphics card and message history.

If the computer restarts, PowerShell closes, or training is interrupted, resume with:

```powershell
uv run imessage-cuda train `
  --config configs/model-1m.yaml `
  --data work/tokens `
  --tokenizer outputs/tokenizer `
  --output outputs/runs/my-model `
  --resume-from outputs/runs/my-model/last `
  --device cuda
```

Use the model config selected for your original run.

## Part 6: evaluate and export

Evaluate the best checkpoint:

```powershell
uv run imessage-cuda evaluate `
  --checkpoint outputs/runs/my-model/best `
  --data work/tokens `
  --output outputs/evaluation.json
```

Export the inference-only model:

```powershell
uv run imessage-cuda export `
  --checkpoint outputs/runs/my-model/best `
  --metrics outputs/evaluation.json `
  --output outputs/final
```

The exported model excludes the optimizer and raw source messages, but its weights remain private
because a language model can memorize training text.

## Part 7: generate your first reply

```powershell
uv run imessage-cuda chat --model outputs/final
```

At `other:`, try something natural:

```text
other: hey, are you free later?
```

The model will print a local `me:` suggestion. Nothing is sent anywhere. Type `/quit` to exit.

If the result is too long or chaotic, try:

```powershell
uv run imessage-cuda chat `
  --model outputs/final `
  --temperature 0.5 `
  --top-p 0.8 `
  --max-new-tokens 24 `
  --repetition-penalty 1.2
```

## Part 8: clean up extra copies

After Windows has successfully imported and verified the snapshot, delete extra copies from the USB
drive and any temporary transfer folder. Keep the project itself on a BitLocker-protected drive.

Never publish `work`, `outputs`, `chat.db`, `manifest.json`, model weights, or screenshots containing
private messages. The source code is safe to share; the data and trained model are not.

You made it. You safely moved a private iMessage history to Windows, trained a model locally, and
generated a reply without uploading a conversation to anyone else.
