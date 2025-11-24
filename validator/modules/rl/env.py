import numpy as np


def _infer_schema(Info: np.ndarray):
    m = Info.shape[1]
    start = 3
    V = (m - start) // 4
    return start, V


def _split_info(Info: np.ndarray):
    N, M = Info.shape
    start, V = _infer_schema(Info)

    qty = Info[:, 0].astype(np.float32)
    duration = Info[:, 1].astype(np.float32)

    cols = [start + 4 * j + k for j in range(V) for k in range(4)]
    block = Info[:, cols].astype(np.float32).reshape(N, V, 4)
    fill_rate = np.clip(block[:, :, 0], 0.0, 1.0)
    rebate_bps = block[:, :, 1]
    punish = block[:, :, 2]
    latest_vol = np.maximum(block[:, :, 3], 0.0)

    return {
        "qty": qty,
        "duration": duration,
        "fill_rate": fill_rate,
        "rebate_bps": rebate_bps,
        "punish": punish,
        "latest_vol": latest_vol,
        "V": V,
        "start_idx": start,
    }


class EnvLite:
    def __init__(
        self, X: np.ndarray, Info: np.ndarray, batch_size: int = 64, seed: int = 7
    ):
        assert X.shape[0] == Info.shape[0], "X and Info must have same number of rows"
        self.X_all = X.astype(np.float32, copy=False)
        self.Info_all = Info.astype(np.float32, copy=False)
        self.N = self.X_all.shape[0]

        parsed = _split_info(self.Info_all)
        self.qty_all = parsed["qty"]
        self.duration_all = parsed["duration"]
        self.fill_all = parsed["fill_rate"]
        self.rebate_all = parsed["rebate_bps"]
        self.punish_all = parsed["punish"]
        self.vol_all = parsed["latest_vol"]
        self.V = parsed["V"]

        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

        self.idx = None
        self.X_b = None
        self.Info_b = None
        self.qty_b = None
        self.duration_b = None
        self.fill_b = None
        self.rebate_b = None
        self.punish_b = None
        self.vol_b = None

    def reset(self):
        if self.N <= self.batch_size:
            self.idx = self.rng.choice(self.N, size=self.batch_size, replace=True)
        else:
            self.idx = self.rng.choice(self.N, size=self.batch_size, replace=False)

        self.X_b = self.X_all[self.idx]
        self.Info_b = self.Info_all[self.idx]

        self.qty_b = self.qty_all[self.idx]
        self.duration_b = self.duration_all[self.idx]
        self.fill_b = self.fill_all[self.idx, :]
        self.rebate_b = self.rebate_all[self.idx, :]
        self.punish_b = self.punish_all[self.idx, :]
        self.vol_b = self.vol_all[self.idx, :]

        return self.X_b, self.Info_b

    @staticmethod
    def _row_normalize(probs: np.ndarray) -> np.ndarray:
        probs = np.maximum(probs, 0.0)
        s = probs.sum(axis=1, keepdims=True)
        out = np.divide(
            probs, s, out=np.zeros_like(probs, dtype=np.float32), where=(s > 0)
        )
        return out

    def step(self, action: np.ndarray) -> np.ndarray:
        assert self.X_b is not None, "Call reset() before step()."
        current_batch_size = self.X_b.shape[0]
        assert action.shape == (
            current_batch_size,
            self.V,
        ), f"action must be {(current_batch_size, self.V)}, but got {action.shape}"
        probs = self._row_normalize(action.astype(np.float32))

        alloc = probs * self.qty_b[:, None]

        cap_window = 1.0 + 0.6 * (1.0 - np.exp(-0.9 * self.duration_b[:, None]))
        capacity = self.vol_b * cap_window * np.clip(self.fill_b, 0.0, 1.0)
        filled = np.minimum(alloc, capacity)
        unfilled = alloc - filled

        reward = ((self.rebate_b / 10000.0) * filled + self.punish_b * unfilled).sum(
            axis=1
        )
        reward = reward / np.log1p(self.qty_b[:])

        return reward.astype(np.float32)

