"""Model definition for the deployed SLR-GAT forecaster.

Ported verbatim from the research notebook (NB06 Cell 3) so that
`model.pt` loads with strict state-dict matching. Do not "tidy" this file:
any change to layer names or ordering breaks weight loading.

Only the graph bodies are kept — the deployed configuration is `gat`.
The recurrent and hybrid bodies were evaluated in the thesis and are not
deployed, so they are deliberately absent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, LayerNorm


class MLP(nn.Module):
    """No graph, no sequence. The control model from the benchmark."""

    def __init__(s, fin, h=48, dp=0.2, layers=2, **kw):
        super().__init__()
        mods, din = [], fin
        for _ in range(layers):
            mods += [nn.Linear(din, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dp)]
            din = h
        s.net, s.out_dim = nn.Sequential(*mods), h

    def forward(s, x, ei, ea=None, et=None, g=None):
        return s.net(x)


class GNN(nn.Module):
    """Attention over running + headway edges.

    Edge features entering attention are [edge_type_embedding | gap | log_gap],
    so `edge_dim = etype_dim + 2`. GCN receives no edge features at all, which
    is why headway rewiring is inoperative under it.
    """

    def __init__(s, fin, kind="gatv2", h=48, heads=4, layers=2, dp=0.2,
                 n_etypes=2, etype_dim=8, **kw):
        super().__init__()
        s.kind, s.dp = kind, dp
        s.etype_emb = nn.Embedding(n_etypes, etype_dim)
        edim = etype_dim + 2
        s.convs, s.norms = nn.ModuleList(), nn.ModuleList()
        din = fin
        for _ in range(layers):
            if kind == "gcn":
                s.convs.append(GCNConv(din, h)); dout = h
            elif kind == "gat":
                s.convs.append(GATConv(din, h, heads=heads, concat=True,
                                       dropout=dp, edge_dim=edim)); dout = h * heads
            else:
                s.convs.append(GATv2Conv(din, h, heads=heads, concat=True,
                                         dropout=dp, edge_dim=edim)); dout = h * heads
            s.norms.append(LayerNorm(dout)); din = dout
        s.out_dim = din

    def forward(s, x, ei, ea=None, et=None, g=None):
        eattr = None
        if s.kind != "gcn" and ea is not None and ea.numel() > 0:
            eattr = torch.cat([s.etype_emb(et), ea], dim=1)
        h = x
        for conv, norm in zip(s.convs, s.norms):
            h = conv(h, ei) if s.kind == "gcn" else conv(h, ei, eattr)
            h = F.relu(norm(h))
            h = F.dropout(h, p=s.dp, training=s.training)
        return h


class Head(nn.Module):
    """Regression head. `hurdle` was disabled in the thesis (86.5% of Tier A
    stops are delayed), so `cls` is None in the deployed model."""

    def __init__(s, body, hurdle=True):
        super().__init__()
        s.body, s.hurdle = body, hurdle
        s.reg = nn.Linear(body.out_dim, 1)
        s.cls = nn.Linear(body.out_dim, 1) if hurdle else None

    def forward(s, x, ei, ea=None, et=None, g=None):
        h = s.body(x, ei, ea, et, g)
        logit = s.cls(h).squeeze(-1) if s.hurdle else None
        return s.reg(h).squeeze(-1), logit


def build_model(cfg):
    """Rebuild the architecture described by config.json."""
    t = cfg["tcfg"]
    kw = dict(h=t["HIDDEN"], heads=t["HEADS"], layers=t["LAYERS"], dp=t["DROPOUT"])
    fin = len(cfg["features"])
    arch = cfg["arch"]
    if arch == "mlp":
        body = MLP(fin, **kw)
    else:
        body = GNN(fin, kind=arch, n_etypes=cfg.get("n_etypes", 2),
                   etype_dim=cfg.get("etype_dim", 8), **kw)
    return Head(body, hurdle=t["HURDLE"])
