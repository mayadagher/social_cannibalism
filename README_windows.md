# Basler camera recorder for Windows

This script records JPEGs from a Basler acA4112-30um, then packages the finished recording into timestamp-keyed HDF5 batches and uploads only those HDF5 files with rclone. The original JPEGs remain in the local run folder.

## 1. Install prerequisites

1. Install Basler's pylon software/runtime for Windows and confirm the camera is visible in pylon Viewer.
2. In the VS Code terminal, create and activate a Python virtual environment, then install the Python packages:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install pypylon opencv-python numpy h5py
   ```

3. Install rclone, configure the server remote with `rclone config`, and confirm the destination with `rclone lsd myremote:`. The destination in the config must use rclone syntax such as `myremote:locust-data`.
4. Copy `config.example.json` to `config.json` and edit it. Keep `config.json` private because it contains phone numbers and server paths. Notification API secrets are read from environment variables, not from the config or log.

## 2. Configure a recording

- `duration_seconds`: stop after this many seconds; pressing `q` in the preview stops sooner.
- `frame_rate_fps`: requested free-running frame rate. The sample is 10 fps.
- `exposure_us`: fixed exposure in microseconds. `10000` is only an example; choose a value suitable for the illumination and motion in the experiment.
- `frames_per_h5`: maximum number of frames in each HDF5 file. The last batch may be smaller.
- `jpeg_quality`: JPEG encoder quality from 1 through 100.
- `output_base_dir`: local parent directory. The script creates a timestamped `recording_...` folder for every run.
- `rclone.destination`: configured rclone remote and destination path.

Set `crop.mode` to one of these:

```json
"crop": { "mode": "none" }
```

```json
"crop": { "mode": "rectangle", "x": 100, "y": 100, "width": 1800, "height": 1800 }
```

```json
"crop": { "mode": "circle", "n_points": 8, "buffer_factor": 1.1 }
```

For a circle crop, the script grabs one calibration still before the timed recording starts. Click at least six points along the arena boundary in the calibration window. The least-squares fitted circle is expanded by `buffer_factor`; the square around that buffered circle is saved, and pixels outside the buffered circle are set to black. JPEG cannot store transparency. The selected points and fitted crop geometry are recorded in `recording.log`.

## 3. Run it

```powershell
python .\locust_camera_recorder.py .\config.json
```

The preview window must have focus for `q` to stop the recording. Image retrieval runs in order and a separate writer drains a lossless queue to disk. HDF5 packaging and rclone transfer begin only after acquisition and JPEG writing finish. An unbounded queue avoids dropping frames if disk writing briefly falls behind, but a sustained disk bottleneck can use substantial memory.

## Output layout

Each run folder contains:

- `jpegs/`: locally retained JPEG files, named with local computer time to millisecond precision (`YYYYMMDD_HHMMSS_mmm.jpg`). A sequence suffix is added only if two images receive the same millisecond timestamp.
- `h5/`: HDF5 batches; each top-level dataset name is the original JPEG timestamp and its value is the JPEG file's bytes as a one-dimensional `uint8` array. There is no extra HDF5 compression.
- `recording.log`: config values (with tokens redacted), camera model/serial and settings read back from the camera, crop geometry, counts, and completion/error messages.
- `rclone.log`: detailed rclone transfer log.

To decode a stored image in Python:

```python
import cv2
import h5py
import numpy as np

with h5py.File("batch.h5", "r") as h5:
    timestamp = next(iter(h5.keys()))
    jpeg_bytes = h5[timestamp][...].tobytes()
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
```

## Completion and error notifications

The script tries notification providers in the configured order and stops at the first successful send. The sample order is Signal, WhatsApp Cloud API, then Telegram. Configure at least one provider; if none can send, the recording/upload results are still logged, but the script returns a nonzero exit code to make the missing notification visible.

- **Signal:** install and link `signal-cli` on this Windows machine; set `account` to the linked account number and `recipient` to the destination number.
- **WhatsApp:** requires a Meta WhatsApp Cloud API account, phone number ID, an access token in the environment variable named by `access_token_env`, a recipient number, and the current Graph API version. Meta may restrict free-form messages outside an active customer-service window.
- **Telegram:** create a bot and set the environment variables named by `bot_token_env` and `chat_id_env`, for example in the VS Code PowerShell terminal:

  ```powershell
  $env:TELEGRAM_BOT_TOKEN = "..."
  $env:TELEGRAM_CHAT_ID = "..."
  ```

Do not put tokens in `config.json`. The script redacts credential values from its run log.

## Important camera behavior

The script explicitly selects `Mono8`, turns auto exposure off, requests continuous free-running acquisition with trigger mode off, and sets the requested frame rate and fixed exposure. It logs the camera's actual readback values. If the pylon driver rejects a value or the feature is not writable, the run stops with an error rather than silently using an unsupported setting.
