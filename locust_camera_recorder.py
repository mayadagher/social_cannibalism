"""Record JPEGs from a Basler camera, package them into HDF5, then upload via rclone.

Windows dependencies: pypylon (with Basler pylon Runtime installed), opencv-python,
numpy, and h5py. See README_windows.md for setup and the config format.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from pypylon import pylon


TIMESTAMP_RE = re.compile(r"^(\d{8}_\d{6}_\d{3})(?:_(\d+))?\.jpg$", re.IGNORECASE)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = ["output_base_dir", "duration_seconds", "frame_rate_fps", "exposure_us", "jpeg_quality", "frames_per_h5", "crop", "rclone", "notifications"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(missing)}")
    if float(cfg["duration_seconds"]) <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if float(cfg["frame_rate_fps"]) <= 0 or float(cfg["exposure_us"]) <= 0:
        raise ValueError("frame_rate_fps and exposure_us must be greater than zero")
    if not 1 <= int(cfg["jpeg_quality"]) <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if int(cfg["frames_per_h5"]) < 1:
        raise ValueError("frames_per_h5 must be at least 1")
    crop = cfg["crop"]
    mode = crop.get("mode")
    if mode not in {"none", "rectangle", "circle"}:
        raise ValueError("crop.mode must be 'none', 'rectangle', or 'circle'")
    if mode == "rectangle" and any(k not in crop for k in ("x", "y", "width", "height")):
        raise ValueError("rectangle crop requires x, y, width, and height")
    if mode == "circle":
        if int(crop.get("n_points", 0)) < 6:
            raise ValueError("circle crop n_points must be at least 6")
        if float(crop.get("buffer_factor", 1.1)) <= 0:
            raise ValueError("circle crop buffer_factor must be positive")
    if cfg["rclone"].get("destination", "").strip() == "":
        raise ValueError("rclone.destination must be a configured rclone remote:path")
    return cfg


def make_run_dir(cfg: dict[str, Any]) -> Path:
    base = Path(cfg["output_base_dir"]).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"recording_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"recording_{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir()
    (run_dir / "jpegs").mkdir()
    (run_dir / "h5").mkdir()
    return run_dir


def configure_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("camera_recorder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(run_dir / "recording.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def safe_config_for_log(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep settings and env-var names, never log notification credentials."""
    copied = json.loads(json.dumps(cfg))
    for key in ("bot_token", "access_token", "token", "api_key", "password"):
        for section in (copied.get("notifications", {}), copied.get("rclone", {})):
            if key in section:
                section[key] = "<redacted>"
    return copied


def open_camera(cfg: dict[str, Any], logger: logging.Logger):
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    logger.info("Camera opened: model=%s serial=%s", camera.GetDeviceInfo().GetModelName(), camera.GetDeviceInfo().GetSerialNumber())

    # This model is monochrome. Mono8 produces one 8-bit intensity channel per pixel.
    if camera.PixelFormat.IsWritable():
        camera.PixelFormat.SetValue("Mono8")
    if hasattr(camera, "TriggerMode") and camera.TriggerMode.IsWritable():
        camera.TriggerMode.SetValue("Off")
    if hasattr(camera, "AcquisitionMode") and camera.AcquisitionMode.IsWritable():
        camera.AcquisitionMode.SetValue("Continuous")
    if hasattr(camera, "ExposureAuto") and camera.ExposureAuto.IsWritable():
        camera.ExposureAuto.SetValue("Off")
    exposure = None
    exposure_feature = None
    for feature_name in ("ExposureTime", "ExposureTimeAbs"):
        feature = getattr(camera, feature_name, None)
        if feature is not None and feature.IsWritable():
            exposure, exposure_feature = feature, feature_name
            break
    if exposure is None:
        raise RuntimeError("Camera ExposureTime/ExposureTimeAbs setting is not writable")
    exposure.SetValue(float(cfg["exposure_us"]))

    rate_enabled = getattr(camera, "AcquisitionFrameRateEnable", None)
    if rate_enabled is not None and rate_enabled.IsWritable():
        rate_enabled.SetValue(True)
    rate_node = None
    rate_feature = None
    for feature_name in ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"):
        feature = getattr(camera, feature_name, None)
        if feature is not None and feature.IsWritable():
            rate_node, rate_feature = feature, feature_name
            break
    if rate_node is None:
        raise RuntimeError("Camera AcquisitionFrameRate/AcquisitionFrameRateAbs is not writable; check the camera's pylon feature set")
    rate_node.SetValue(float(cfg["frame_rate_fps"]))
    gain_node = getattr(camera, "Gain", None)
    gain_value = gain_node.GetValue() if gain_node is not None and gain_node.IsReadable() else "unavailable"
    trigger_value = camera.TriggerMode.GetValue() if hasattr(camera, "TriggerMode") else "unavailable"
    acquisition_mode = camera.AcquisitionMode.GetValue() if hasattr(camera, "AcquisitionMode") else "unavailable"
    logger.info(
        "Camera settings readback: PixelFormat=%s Width=%d Height=%d %s=%.3f us %s=%.6f fps TriggerMode=%s AcquisitionMode=%s Gain=%s",
        camera.PixelFormat.GetValue(), camera.Width.GetValue(), camera.Height.GetValue(),
        exposure_feature, exposure.GetValue(), rate_feature, rate_node.GetValue(),
        trigger_value, acquisition_mode, gain_value,
    )
    return camera


def fit_circle(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 6:
        raise ValueError("At least six points are required to fit the arena circle")
    # Algebraic least-squares circle fit: x²+y² + A*x + B*y + C = 0.
    mat = np.column_stack((pts[:, 0], pts[:, 1], np.ones(len(pts))))
    rhs = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    a, b, c = np.linalg.lstsq(mat, rhs, rcond=None)[0]
    cx, cy = -a / 2.0, -b / 2.0
    radius_sq = cx * cx + cy * cy - c
    if radius_sq <= 0:
        raise ValueError("Selected points did not produce a valid circle")
    return float(cx), float(cy), math.sqrt(float(radius_sq))


def click_circle_points(image: np.ndarray, n_points: int) -> list[tuple[float, float]]:
    h, w = image.shape[:2]
    scale = min(1.0, 1500.0 / max(h, w))
    display = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    points: list[tuple[float, float]] = []
    win = "Circle calibration: click arena boundary; press q to cancel"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((x / scale, y / scale))

    cv2.setMouseCallback(win, on_mouse)
    while len(points) < n_points:
        canvas = display.copy()
        for x, y in points:
            cv2.circle(canvas, (round(x * scale), round(y * scale)), 5, 255, -1)
        cv2.putText(canvas, f"Click {n_points} boundary points ({len(points)}/{n_points}); q cancels", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.imshow(win, canvas)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            cv2.destroyWindow(win)
            raise RuntimeError("Circle calibration cancelled")
    cv2.destroyWindow(win)
    return points


def resolve_crop(image: np.ndarray, cfg: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    c = cfg["crop"]
    h, w = image.shape[:2]
    mode = c["mode"]
    if mode == "none":
        return {"mode": mode}
    if mode == "rectangle":
        x, y, cw, ch = map(int, (c["x"], c["y"], c["width"], c["height"]))
        if x < 0 or y < 0 or cw <= 0 or ch <= 0 or x + cw > w or y + ch > h:
            raise ValueError(f"Rectangle crop {(x, y, cw, ch)} falls outside the {w}x{h} sensor image")
        result = {"mode": mode, "x": x, "y": y, "width": cw, "height": ch}
        logger.info("Resolved crop: %s", result)
        return result

    # Grab one still for interactive point selection. It is calibration only and is not saved as a video frame.
    points = click_circle_points(image, int(c["n_points"]))
    cx, cy, radius = fit_circle(points)
    buffered_radius = radius * float(c.get("buffer_factor", 1.1))
    x0, y0 = max(0, int(math.floor(cx - buffered_radius))), max(0, int(math.floor(cy - buffered_radius)))
    x1, y1 = min(w, int(math.ceil(cx + buffered_radius))), min(h, int(math.ceil(cy + buffered_radius)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Fitted buffered circle falls outside the camera image")
    result = {
        "mode": "circle", "center_x": cx, "center_y": cy,
        "radius_pixels": radius, "buffer_factor": float(c.get("buffer_factor", 1.1)),
        "buffered_radius_pixels": buffered_radius, "x": x0, "y": y0,
        "width": x1 - x0, "height": y1 - y0, "mask_outside_circle_black": True,
        "clicked_boundary_points_xy": [[float(x), float(y)] for x, y in points],
    }
    logger.info("Resolved circle crop from %d clicked points: %s", len(points), result)
    return result


def apply_crop(image: np.ndarray, crop: dict[str, Any]) -> np.ndarray:
    if crop["mode"] == "none":
        return image
    x, y = int(crop["x"]), int(crop["y"])
    out = image[y:y + int(crop["height"]), x:x + int(crop["width"])].copy()
    if crop["mode"] == "circle":
        yy, xx = np.ogrid[:out.shape[0], :out.shape[1]]
        cx, cy = crop["center_x"] - x, crop["center_y"] - y
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 > crop["buffered_radius_pixels"] ** 2
        out[mask] = 0  # JPEG has no alpha channel, so masked pixels are black.
    return out


def make_frame_name(now: datetime, last_ms: int, collision_index: int) -> tuple[str, int, int]:
    current_ms = int(now.timestamp() * 1000)
    stem = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    if current_ms == last_ms:
        collision_index += 1
        stem += f"_{collision_index:03d}"
    else:
        collision_index = 0
    return stem + ".jpg", current_ms, collision_index


def capture(camera, cfg: dict[str, Any], run_dir: Path, logger: logging.Logger) -> list[Path]:
    # Retrieve/grab runs in this thread; the camera's buffers are drained promptly and disk I/O
    # is delegated to a lossless, unbounded queue so the producer never drops frames on a full queue.
    q: queue.Queue[Any] = queue.Queue()
    sentinel = object()
    errors: list[BaseException] = []
    saved: list[Path] = []
    crop: dict[str, Any] | None = None
    stop = threading.Event()

    # Finish crop setup before the timed recording begins, so clicks never pause acquisition.
    if cfg["crop"]["mode"] == "circle":
        calibration = camera.GrabOne(5000, pylon.TimeoutHandling_ThrowException)
        try:
            if not calibration.GrabSucceeded():
                raise RuntimeError(f"Pylon calibration grab failed: {calibration.ErrorCode} {calibration.ErrorDescription}")
            still = calibration.Array.copy()
        finally:
            calibration.Release()
    else:
        still = np.empty((int(camera.Height.GetValue()), int(camera.Width.GetValue())), dtype=np.uint8)
    crop = resolve_crop(still, cfg, logger)
    logger.info("Resolved crop saved for every frame: %s", crop)

    def writer():
        try:
            while True:
                item = q.get()
                try:
                    if item is sentinel:
                        return
                    filename, image = item
                    if not errors:
                        try:
                            ok = cv2.imwrite(str(run_dir / "jpegs" / filename), image, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg_quality"])])
                            if not ok:
                                raise OSError(f"OpenCV failed to write {filename}")
                            saved.append(run_dir / "jpegs" / filename)
                        except BaseException as e:
                            errors.append(e)
                            stop.set()
                finally:
                    q.task_done()
        except BaseException as e:
            errors.append(e)
            stop.set()

    writer_thread = threading.Thread(target=writer, name="jpeg-writer", daemon=True)
    writer_thread.start()
    duration = float(cfg["duration_seconds"])
    last_ms, collision_index = -1, 0
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    started = time.monotonic()
    deadline = started + duration
    logger.info("Recording started; duration limit %.3f s. Press q in preview to stop early.", duration)
    cv2.namedWindow("Basler recording (q to stop)", cv2.WINDOW_NORMAL)
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            result = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if result is None:
                continue
            try:
                if not result.GrabSucceeded():
                    raise RuntimeError(f"Pylon grab failed: {result.ErrorCode} {result.ErrorDescription}")
                now = datetime.now()  # Computer wall-clock timestamp, captured immediately after retrieve.
                image = result.Array.copy()
            finally:
                result.Release()

            frame = apply_crop(image, crop)
            filename, last_ms, collision_index = make_frame_name(now, last_ms, collision_index)
            q.put((filename, frame))
            preview = frame
            if preview.shape[1] > 1500:
                preview = cv2.resize(preview, (1500, round(preview.shape[0] * 1500 / preview.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imshow("Basler recording (q to stop)", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Recording stopped early by q key.")
                break
            if errors:
                raise RuntimeError("JPEG writer failed") from errors[0]
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        cv2.destroyAllWindows()
        q.put(sentinel)
        q.join()
        writer_thread.join()
    if errors:
        raise RuntimeError("JPEG writer failed") from errors[0]
    elapsed = time.monotonic() - started
    logger.info("Recording stopped: %.3f s, %d JPEG frames saved; writer queue drained.", elapsed, len(saved))
    if not saved:
        raise RuntimeError("No frames were captured")
    return sorted(saved)


def timestamp_sort_key(path: Path):
    m = TIMESTAMP_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected JPEG filename (timestamp missing): {path.name}")
    return (m.group(1), int(m.group(2) or 0))


def package_h5(jpeg_paths: list[Path], cfg: dict[str, Any], run_dir: Path, logger: logging.Logger) -> list[Path]:
    batch_size = int(cfg["frames_per_h5"])
    h5_paths: list[Path] = []
    total = len(jpeg_paths)
    for start in range(0, total, batch_size):
        batch = jpeg_paths[start:start + batch_size]
        first_stamp = timestamp_sort_key(batch[0])[0]
        last_stamp = timestamp_sort_key(batch[-1])[0]
        out_path = run_dir / "h5" / f"frames_{start + 1:08d}_{start + len(batch):08d}_{first_stamp}_to_{last_stamp}.h5"
        with h5py.File(out_path, "w") as h5:
            h5.attrs["description"] = "Each top-level HDF5 dataset is named by the computer capture timestamp and stores one JPEG file as uint8 bytes."
            h5.attrs["frame_count"] = len(batch)
            h5.attrs["jpeg_quality"] = int(cfg["jpeg_quality"])
            h5.attrs["hdf5_compression"] = "none"
            h5.attrs["timestamp_timezone"] = "local computer time"
            for jpeg in batch:
                key = jpeg.stem
                data = np.fromfile(jpeg, dtype=np.uint8)
                if data.size == 0:
                    raise OSError(f"JPEG is empty: {jpeg}")
                h5.create_dataset(key, data=data, dtype="u1")
        h5_paths.append(out_path)
        logger.info("Created HDF5 batch %s containing %d/%d frames.", out_path.name, len(batch), total)
    logger.info("HDF5 packaging complete: %d files, %d frames; no additional HDF5 compression applied.", len(h5_paths), total)
    return h5_paths


def rclone_copy(cfg: dict[str, Any], run_dir: Path, logger: logging.Logger) -> None:
    rconf = cfg["rclone"]
    executable = str(rconf.get("executable", "rclone"))
    if shutil.which(executable) is None and not Path(executable).is_file():
        raise FileNotFoundError(f"Could not find rclone executable: {executable}")
    log_path = run_dir / "rclone.log"
    cmd = [executable, "copy", str(run_dir / "h5"), str(rconf["destination"]), "--log-file", str(log_path), "--log-level", "INFO", "--log-format", "date,time,microseconds", "--stats", "30s"]
    if rconf.get("transfers") is not None:
        cmd.extend(["--transfers", str(int(rconf["transfers"]))])
    logger.info("Starting rclone upload to destination %s", rconf["destination"])
    logger.info("rclone command: rclone copy %s %s [logging/progress flags]", run_dir / "h5", rconf["destination"])
    completed = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.stdout.strip():
        logger.info("rclone stdout: %s", completed.stdout.strip().replace("\n", " | "))
    if completed.stderr.strip():
        logger.info("rclone stderr: %s", completed.stderr.strip().replace("\n", " | "))
    logger.info("rclone exit code: %d; detailed transfer log: %s", completed.returncode, log_path)
    if completed.returncode != 0:
        raise RuntimeError(f"rclone copy failed with exit code {completed.returncode}; see {log_path}")


def env_value(name: str | None) -> str | None:
    return os.environ.get(name) if name else None


def send_signal(provider: dict[str, Any], message: str) -> None:
    exe = str(provider.get("executable", "signal-cli"))
    account, recipient = provider.get("account"), provider.get("recipient")
    if not account or not recipient or (shutil.which(exe) is None and not Path(exe).is_file()):
        raise RuntimeError("Signal is not configured: set executable/account/recipient and install/link signal-cli")
    proc = subprocess.run([exe, "-a", str(account), "send", "-m", message, str(recipient)], capture_output=True, text=True, timeout=60, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"signal-cli failed ({proc.returncode}): {proc.stderr.strip()}")


def send_whatsapp(provider: dict[str, Any], message: str) -> None:
    token = env_value(provider.get("access_token_env"))
    phone_id, recipient, version = provider.get("phone_number_id"), provider.get("recipient"), provider.get("graph_api_version")
    if not token or not phone_id or not recipient or not version:
        raise RuntimeError("WhatsApp Cloud API is not configured (token env, phone_number_id, recipient, graph_api_version required)")
    url = f"https://graph.facebook.com/{version}/{phone_id}/messages"
    body = json.dumps({"messaging_product": "whatsapp", "to": recipient, "type": "text", "text": {"body": message}}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"WhatsApp API returned HTTP {response.status}")


def send_telegram(provider: dict[str, Any], message: str) -> None:
    token = env_value(provider.get("bot_token_env"))
    chat_id = env_value(provider.get("chat_id_env")) or provider.get("chat_id")
    if not token or not chat_id:
        raise RuntimeError("Telegram is not configured (bot token env and chat ID required)")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
        if response.status < 200 or response.status >= 300 or not payload.get("ok"):
            raise RuntimeError("Telegram sendMessage returned an unsuccessful response")


def notify(cfg: dict[str, Any], message: str, logger: logging.Logger) -> bool:
    settings = cfg.get("notifications", {})
    providers = settings.get("providers", ["signal_cli", "whatsapp_cloud", "telegram"])
    functions = {"signal_cli": send_signal, "whatsapp_cloud": send_whatsapp, "telegram": send_telegram}
    errors: list[str] = []
    for name in providers:
        if name not in functions:
            errors.append(f"Unknown notification provider {name!r}")
            continue
        try:
            functions[name](settings.get(name, {}), message)
            logger.info("Notification sent through %s.", name)
            return True
        except (OSError, RuntimeError, urllib.error.URLError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning("Notification provider %s failed: %s", name, e)
            errors.append(f"{name}: {e}")
    logger.error("No notification provider succeeded. %s", "; ".join(errors) or "No providers configured")
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python locust_camera_recorder.py config.json", file=sys.stderr)
        return 2
    config_path = Path(sys.argv[1]).expanduser().resolve()
    try:
        cfg = load_config(config_path)
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    run_dir = make_run_dir(cfg)
    logger = configure_logger(run_dir)
    logger.info("Run folder: %s", run_dir)
    logger.info("Configuration file: %s", config_path)
    logger.info("Effective parameters:\n%s", json.dumps(safe_config_for_log(cfg), indent=2, ensure_ascii=False))
    camera = None
    try:
        camera = open_camera(cfg, logger)
        jpegs = capture(camera, cfg, run_dir, logger)
        h5_files = package_h5(jpegs, cfg, run_dir, logger)
        rclone_copy(cfg, run_dir, logger)
        logger.info("Run completed successfully. Frames=%d HDF5_files=%d; local JPEGs retained at %s", len(jpegs), len(h5_files), run_dir / "jpegs")
        if not notify(cfg, f"Camera recording completed: {len(jpegs)} frames in {len(h5_files)} HDF5 file(s). Run folder: {run_dir}", logger):
            logger.error("Recording and upload succeeded, but no success notification could be delivered.")
            return 1
        return 0
    except Exception as e:
        logger.exception("Run failed: %s", e)
        notify(cfg, f"Camera recording FAILED. {type(e).__name__}: {e}. See local run log: {run_dir / 'recording.log'}", logger)
        return 1
    finally:
        if camera is not None:
            try:
                if camera.IsOpen():
                    camera.Close()
            except Exception:
                logger.exception("Failed to close camera cleanly")
        logger.info("Run log: %s", run_dir / "recording.log")


if __name__ == "__main__":
    raise SystemExit(main())
