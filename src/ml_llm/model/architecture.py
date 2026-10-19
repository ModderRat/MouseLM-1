from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ml_llm.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.dim % cfg.n_heads == 0

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = cfg.dropout

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(cfg.context, cfg.context)
            ).view(1, 1, cfg.context, cfg.context),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        att = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        att = att.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf"),
        )

        att = F.softmax(att, dim=-1)

        att = F.dropout(
            att,
            p=self.dropout,
            training=self.training,
        )

        out = att @ v

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(
            cfg.dim,
            4 * cfg.dim,
        )

        self.fc2 = nn.Linear(
            4 * cfg.dim,
            cfg.dim,
        )

        self.dropout = nn.Dropout(
            cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.fc2(
                F.gelu(
                    self.fc1(x)
                )
            )
        )


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg

        self.token_emb = nn.Embedding(
            cfg.vocab_size,
            cfg.dim,
        )

        self.pos_emb = nn.Embedding(
            cfg.context,
            cfg.dim,
        )

        self.drop = nn.Dropout(
            cfg.dropout,
        )

        self.blocks = nn.ModuleList(
            [
                Block(cfg)
                for _ in range(cfg.n_layers)
            ]
        )

        self.feedback_gate = nn.Linear(
            2 * cfg.dim,
            cfg.dim,
        )

        self.feedback_ln = nn.LayerNorm(cfg.dim)

        self.ln_f = nn.LayerNorm(cfg.dim)

        self.head = nn.Linear(
            cfg.dim,
            cfg.vocab_size,
            bias=False,
        )

        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        B, T = idx.shape

        pos = torch.arange(
            T,
            device=idx.device,
        )

        x0 = self.drop(
            self.token_emb(idx)
            + self.pos_emb(pos)
        )

        x = x0

        # Activation checkpointing trades a small amount of extra compute
        # for substantially lower peak memory during backward. This is
        # especially important on unified-memory Apple MPS systems.
        #
        # It does not change the model architecture, parameter shapes, or
        # checkpoint format, so existing checkpoints remain compatible.
        use_checkpoint = (
            self.training
            and x.device.type == "mps"
            and os.environ.get(
                "ML_LLM_MPS_CHECKPOINT",
                "1",
            ) != "0"
        )

        for loop_idx in range(self.cfg.n_loops):
            for block in self.blocks:
                if use_checkpoint:
                    x = checkpoint(
                        block,
                        x,
                        use_reentrant=False,
                    )
                else:
                    x = block(x)

            if loop_idx < self.cfg.n_loops - 1:
                gate = torch.sigmoid(
                    self.feedback_gate(
                        torch.cat([x, x0], dim=-1)
                    )
                )

                x = self.feedback_ln(
                    gate * x + (1 - gate) * x0
                )

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None

        if targets is not None:
            flat_logits = logits.view(
                -1,
                logits.size(-1),
            )

            flat_targets = targets.view(-1)

            # MPS-safe cross entropy.
            #
            # Avoid F.cross_entropy() because the MPS
            # implementation enters the NLL-loss path.
            #
            # Everything stays on the same device and
            # autograd remains intact.

            log_probs = F.log_softmax(
                flat_logits,
                dim=-1,
            )

            safe_targets = flat_targets.clamp_min(0)

            selected = log_probs.gather(
                1,
                safe_targets.unsqueeze(1),
            ).squeeze(1)

            valid = flat_targets.ne(-1)

            selected = selected * valid.to(
                selected.dtype
            )

            valid_count = valid.sum().clamp_min(1)

            loss = -selected.sum() / valid_count

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    seen_logits = logits[b, seen]
                    logits[b, seen] = torch.where(
                        seen_logits > 0,
                        seen_logits / repetition_penalty,
                        seen_logits * repetition_penalty,
                    )

            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1),
                    ),
                )

                logits[
                    logits < v[:, [-1]]
                ] = float("-inf")

            probs = F.softmax(
                logits,
                dim=-1,
            )

            next_id = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (
                    idx,
                    next_id,
                ),
                dim=1,
            )

        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
