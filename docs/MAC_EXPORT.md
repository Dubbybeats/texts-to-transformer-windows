# Mac export guide

The Windows PC handles training, but Apple stores iMessage history inside a protected macOS
database. This small export step creates a consistent read-only snapshot for the PC.

## Before exporting

Open Messages and confirm that the conversations you expect are visible. If Messages is still
downloading history from iCloud, let it finish first.

Copy `tools/export_imessage_snapshot.py` from the project to the Mac desktop.

## Give Terminal temporary access

Open **System Settings**, choose **Privacy & Security**, then choose **Full Disk Access**. Enable
Terminal. If Terminal is not listed, use the plus button to add it from
`Applications/Utilities/Terminal.app`.

Quit Terminal completely with `Command + Q`, then reopen it. Closing only the window is not enough
for the permission change to take effect.

## Run the exporter

```bash
cd ~/Desktop
python3 export_imessage_snapshot.py
```

The exporter:

1. Opens `~/Library/Messages/chat.db` in SQLite read-only mode.
2. Uses SQLite's backup API to create a consistent copy.
3. Runs `PRAGMA quick_check` against that copy.
4. Calculates a SHA-256 fingerprint.
5. Writes the fingerprint and non-message metadata to `manifest.json`.

It does not open attachments, change the Messages database, upload anything, or print conversation
text.

## Transfer the result

Move the entire `imessage-snapshot` folder to the Windows PC using a USB drive or another private
local method. Keep `chat.db` and `manifest.json` together because Windows uses the manifest to catch
an incomplete or altered transfer.

Avoid email, public cloud links, and messaging attachments. The snapshot contains private
conversations even though the later Windows pipeline pseudonymizes its derived dataset.

## Common problems

### `Messages database not found`

Open Messages and make sure the Mac is signed into the correct Apple Account. The project expects
the normal macOS location at `~/Library/Messages/chat.db`.

### `Could not read Messages`

Terminal does not yet have Full Disk Access. Recheck the toggle, quit Terminal with `Command + Q`,
reopen it, and run the command again.

### `Messages changed during export`

A message arrived or the database changed while the safety checks were running. Wait a moment and
run the exporter again. This refusal is intentional and protects the consistency of the snapshot.

### `python3: command not found`

Install a current Python 3 release from [python.org](https://www.python.org/downloads/macos/) or use
an existing Homebrew Python installation. The exporter uses only Python's standard library.

After the Windows import succeeds, you may remove Terminal's Full Disk Access and delete the desktop
snapshot if you no longer need it.
