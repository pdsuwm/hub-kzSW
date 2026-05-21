"""
第五周作业：基于 Transformer 的单向语言模型 + 文本生成
"""

import math, glob, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ────────── 数据 ──────────

def load_corpus(pattern="*.txt"):
    return "".join(open(p, encoding="utf-8", errors="ignore").read() for p in glob.glob(pattern))

def build_vocab(text):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c

class CharDataset(Dataset):
    def __init__(self, text, c2i, seq_len):
        ids = [c2i[c] for c in text if c in c2i]
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self): return max(0, len(self.data) - self.seq_len)
    def __getitem__(self, i):
        return self.data[i:i+self.seq_len], self.data[i+1:i+self.seq_len+1]


# ────────── 模型 ──────────

class CausalSelfAttention(nn.Module):
    """多头因果自注意力：用下三角掩码屏蔽未来位置，实现单向语言模型"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # 拆分多头
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # 注意力分数 + 因果掩码
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        # 聚合并合并多头
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks  = nn.Sequential(*[Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln      = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        return self.head(self.ln(self.blocks(h)))


# ────────── 文本生成 ──────────

@torch.no_grad()
def generate(model, c2i, i2c, prompt, max_new=200, temperature=0.8, top_k=40, device="cpu"):
    """
    自回归生成：每步把已有序列输入模型，取最后位置的输出预测下一个字符。
    使用 top-k 采样：只从概率最高的 k 个字符中随机选，避免生成低质量词。
    """
    model.eval()
    ids = [c2i[c] for c in prompt if c in c2i] or [0]
    ctx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new):
        ctx_crop = ctx[:, -model.seq_len:]          # 超长时裁剪
        logits = model(ctx_crop)[0, -1] / temperature
        # top-k：保留最高 k 个，其余置 -inf
        top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_vals[-1]] = float("-inf")
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ctx = torch.cat([ctx, next_id.unsqueeze(0)], dim=1)

    return "".join(i2c.get(i, "?") for i in ctx[0].tolist())


# ────────── 训练 ──────────

def train():
    # 超参数
    SEQ_LEN    = 64
    BATCH_SIZE = 128
    EPOCHS     = 20
    D_MODEL    = 128
    N_HEADS    = 4
    N_LAYERS   = 2
    LR         = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    text = load_corpus("*.txt")
    assert text, "未找到 .txt 文件"
    print(f"语料大小: {len(text):,}  ", end="")

    c2i, i2c = build_vocab(text)
    print(f"词表大小: {len(c2i)}")

    # 划分训练/验证集
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * 0.95)
    train_ds = CharDataset("\n".join(lines[:split]), c2i, SEQ_LEN)
    val_ds   = CharDataset("\n".join(lines[split:]), c2i, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = TransformerLM(len(c2i), D_MODEL, N_HEADS, N_LAYERS, SEQ_LEN).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_ppl, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):
        # 训练
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x).reshape(-1, len(c2i)), y.reshape(-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x).reshape(-1, len(c2i)), y.reshape(-1)).item()
        val_loss /= len(val_loader)
        val_ppl = math.exp(val_loss)

        mark = " *" if val_ppl < best_ppl else ""
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_ppl={val_ppl:.2f}{mark}")

    print(f"\n最佳 val PPL: {best_ppl:.2f}")
    model.load_state_dict(best_state)

    # ── 文本生成演示 ──
    print("\n===== 文本生成 =====")
    for prompt in ["黄金", "中国证券", "经济", "市场"]:
        print(f"\n[prompt: {prompt}]")
        print(generate(model, c2i, i2c, prompt, device=device))


if __name__ == "__main__":
    train()
