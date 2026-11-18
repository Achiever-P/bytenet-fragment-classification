import numpy as np, torch
from torch.utils.data import Dataset
from bytenet.byte2image import byte2image, byte_stats

class FragmentDataset(Dataset):
    def __init__(self, X, y, n_gram=4, train=True):
        self.X, self.y, self.train = X, y, train
        self.imgs = np.stack([byte2image(x, n=n_gram) for x in X]).astype(np.float32)
        self.stats = np.stack([byte_stats(x) for x in X])
    def __len__(self): return len(self.X)
    def _aug(self, img):
        if self.train and np.random.rand() < 0.5: img = img[:, ::-1].copy()
        if self.train and np.random.rand() < 0.5:
            h, w = img.shape; eh, ew = max(1, h // 6), max(1, w // 3)
            y0, x0 = np.random.randint(0, max(1, h - eh)), np.random.randint(0, max(1, w - ew))
            img[y0:y0+eh, x0:x0+ew] = np.random.randint(0, 256)
        return img
    def __getitem__(self, i):
        img = self._aug(self.imgs[i].copy()) / 255.0
        img = (img - img.mean()) / (img.std() + 1e-6)
        return (torch.from_numpy(self.X[i].astype(np.float32) / 255.0),
                torch.from_numpy(img).unsqueeze(0).float(),
                torch.from_numpy(self.stats[i]), torch.tensor(self.y[i], dtype=torch.long))

def mixup(raw, img, st, labels, k, alpha=0.8):
    lam = np.random.beta(alpha, alpha); p = torch.randperm(raw.size(0))
    oh = torch.zeros(labels.size(0), k); oh.scatter_(1, labels.unsqueeze(1), 1.0)
    return (lam*raw + (1-lam)*raw[p], lam*img + (1-lam)*img[p], lam*st + (1-lam)*st[p], lam*oh + (1-lam)*oh[p])
