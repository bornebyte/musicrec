"""ffmpeg-based decoding: robust against broken MP3 headers, any container, fast seeking."""
import subprocess
import numpy as np


def probe_duration(path) -> float:
    try:
        from mutagen import File as MFile
        f = MFile(str(path))
        if f is not None and f.info is not None and getattr(f.info, "length", 0):
            return float(f.info.length)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip() or 0)
    except Exception:
        return 0.0


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    return np.frombuffer(r.stdout, dtype="<f4").copy(), r.stderr.decode(errors="ignore")[:200]


def decode(path, start: float, dur: float, sr: int) -> np.ndarray:
    common = ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    base = ["ffmpeg", "-v", "error", "-nostdin", "-err_detect", "ignore_err"]
    s = f"{max(start, 0):.2f}"
    # 1) fast input seek; 2) accurate output seek (slower, survives broken frames); 3) whole-file decode
    for cmd in (base + ["-ss", s, "-t", f"{dur:.2f}", "-i", str(path)] + common,
                base + ["-i", str(path), "-ss", s, "-t", f"{dur:.2f}"] + common,
                base + ["-i", str(path), "-t", f"{dur:.2f}"] + common):
        y, err = _run(cmd)
        if y.size:
            return y
    raise RuntimeError(f"ffmpeg produced no audio: {err}")
