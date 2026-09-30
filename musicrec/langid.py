"""Per-song spoken/sung language identification with SpeechBrain's VoxLingua107 ECAPA model.
Gives (a) a 256-d language embedding and (b) a probability over 107 languages, averaged over 3 windows."""
import sys
import numpy as np

from . import config as C
from .audio import decode


class LangID:
    def __init__(self):
        import torch
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError:                                   # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier
        torch.set_num_threads(1)
        self.torch = torch
        self.m = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir=str(C.MODEL_DIR / "voxlingua107"),
            run_opts={"device": "cpu"})
        self.labels = self.m.hparams.label_encoder.ind2lab    # {idx: 'hi: Hindi'}

    def analyse(self, path, duration: float):
        W = C.LID_WINDOW
        if duration <= W + 1:
            starts = [0.0]
        else:
            starts = [min(max(duration * f - W / 2, 0), duration - W) for f in (0.2, 0.5, 0.75)][:C.LID_WINDOWS]
        n = W * C.LID_SR
        sigs = []
        for s in starts:
            try:
                y = decode(path, s, W, C.LID_SR)
            except Exception:
                continue
            if y.size < C.LID_SR * 2 or np.sqrt(np.mean(y ** 2)) < 1e-4:
                continue                                        # silence / too short
            y = np.pad(y, (0, max(0, n - y.size)))[:n]
            sigs.append(y)
        if not sigs:
            return None
        torch = self.torch
        x = torch.from_numpy(np.stack(sigs).astype(np.float32))
        with torch.no_grad():
            emb = self.m.encode_batch(x)                        # (B,1,256)
            logp = self.m.mods.classifier(emb).squeeze(1)       # (B,107) log-probs
        probs = torch.exp(logp).mean(0).numpy()
        probs = probs / probs.sum()
        e = emb.squeeze(1).mean(0).numpy()
        e = e / (np.linalg.norm(e) + 1e-9)
        top = []
        for i in np.argsort(-probs)[:5]:
            code, _, name = self.labels[int(i)].partition(": ")
            top.append({"code": code, "name": name or code, "p": round(float(probs[i]), 4)})
        return {"emb": e.astype(np.float32), "top": top}
