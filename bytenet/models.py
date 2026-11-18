"""ByteNet (dual-branch) plus the proposed extension: a position-invariant statistics branch and gated fusion."""
import torch, torch.nn as nn

class SideBranch(nn.Module):
    """side='byte': paper's single FC over the raw byte sequence (BBFE).
       side='hist': proposed statistics branch over a 256-bin histogram + entropy."""
    def __init__(self, side, sector_size, out_dim=128):
        super().__init__()
        self.side = side
        if side == "byte":
            self.net = nn.Sequential(nn.Linear(sector_size, out_dim), nn.ReLU(inplace=True))
        else:
            self.net = nn.Sequential(nn.Linear(257, out_dim), nn.ReLU(inplace=True),
                                     nn.Linear(out_dim, out_dim), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class NgramEmbedding(nn.Module):
    def __init__(self, img_w, k=32, c1=16):
        super().__init__()
        self.wide = nn.Conv2d(1, k, (1, img_w))
        self.proj = nn.Conv2d(1, c1, 7, padding=3)
        self.bn, self.act, self.pool = nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2)
    def forward(self, x):
        e = self.wide(x).squeeze(-1).permute(0, 2, 1).unsqueeze(1)  # (B,1,H,K)
        return self.pool(self.act(self.bn(self.proj(e))))

class PatchEmbedding(nn.Module):
    def __init__(self, img_h, img_w, patch=8, c1=16):
        super().__init__()
        self.proj = nn.Conv2d(1, c1, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, c1, img_h // patch, img_w // patch))
    def forward(self, x): return self.proj(x) + self.pos

class ResNetBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        o = self.act(self.b1(self.c1(x))); o = self.b2(self.c2(o)); return self.act(o + x)

class PoolFormerBlock(nn.Module):
    def __init__(self, c, r=2):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(1, c), nn.GroupNorm(1, c)
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.mlp = nn.Sequential(nn.Conv2d(c, c*r, 1), nn.GELU(), nn.Conv2d(c*r, c, 1))
    def forward(self, x):
        z = (self.pool(self.n1(x)) - self.n1(x)) + x
        return self.mlp(self.n2(z)) + z

class Stage(nn.Module):
    def __init__(self, cin, cout, depth, kind, last):
        super().__init__()
        blk = ResNetBlock if kind == "resnet" else PoolFormerBlock
        self.inp = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.blocks = nn.Sequential(*[blk(cout) for _ in range(depth)])
        self.down = nn.AdaptiveAvgPool2d(1) if last else nn.Sequential(
            nn.Conv2d(cout, cout, 3, stride=2, padding=1), nn.ReLU(inplace=True))
    def forward(self, x): return self.down(self.blocks(self.inp(x)))

class ByteNet(nn.Module):
    def __init__(self, sector_size, img_h, img_w, n_classes, variant="resnet", side="byte",
                 fusion="concat", channels=(16, 32, 64, 128), depths=(1, 1, 1, 1), d=128, skip_image=False):
        super().__init__()
        self.fusion, self.skip_image, self.side_kind = fusion, skip_image, side
        self.side = SideBranch(side, sector_size, d)
        self.embed = NgramEmbedding(img_w, c1=channels[0]) if variant == "resnet" else PatchEmbedding(img_h, img_w, c1=channels[0])
        stages, cin = [], channels[0]
        for i, (c, dp) in enumerate(zip(channels, depths)):
            stages.append(Stage(cin, c, dp, variant, i == len(channels) - 1)); cin = c
        self.stages = nn.ModuleList(stages)
        self.c4 = channels[-1]
        if fusion == "gate":
            self.pb, self.pi = nn.Linear(d, d), nn.Linear(self.c4, d)
            self.gate = nn.Linear(2 * d, 2 * d)
            self.cls = nn.Linear(2 * d, n_classes)
        else:
            self.cls = nn.Linear(d + self.c4, n_classes)
        self.d = d
    def forward(self, raw, img, stats):
        s = self.side(raw if self.side_kind == "byte" else stats)
        if self.skip_image:
            x = torch.zeros(s.shape[0], self.c4, device=s.device)
        else:
            x = self.embed(img)
            for st in self.stages: x = st(x)
            x = x.flatten(1)
        if self.fusion == "gate":
            pb, pi = torch.relu(self.pb(s)), torch.relu(self.pi(x))
            g = torch.sigmoid(self.gate(torch.cat([pb, pi], 1)))
            f = torch.cat([g[:, :self.d] * pb, g[:, self.d:] * pi], 1)
        else:
            f = torch.cat([s, x], 1)
        return self.cls(f)
