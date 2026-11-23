"""Компактные состояния AdamW для 4 ГБ VRAM (experiments/lfm2_train_profile.md,
приоритет 5).

torch.optim.AdamW(fused=True) создаёт состояния через zeros_like(p) — для fp16
параметров это fp16-состояния m и v. m (линейный по градиенту) в fp16 безопасен,
а v = g^2 — нет: при |g| ~ 1e-4 значение v ~ 1e-8 лежит на пять порядков ниже
нижней нормали fp16 (~6e-5) и флашится в ноль (или субнормал) — denom = sqrt(v)
схлопывается в eps, и шаг параметра становится ~lr*m/eps, на порядки больше
нормального (наблюдалось как скачок loss на 100-шаговом файнтюне).

AdamWFused16 держит m в fp16 (zeros_like(p)) и v в fp32 (+0.46 ГБ на 230M
параметров; суммарно состояния 1.38 ГБ вместо 0.92 fp16 / 1.84 fp32) — v не
теряет порядок, а фузированный кернел adamw.slang принимает смешанные типы
(T=fp16 для param/grad/m, U=fp32 для v) и остаётся одним диспетчем на параметр.
"""

import torch


class AdamWFused16(torch.optim.AdamW):
    """AdamW(fused=True) с компактными состояниями: m fp16, v fp32.

    Параметры — fp16 (или fp32) на vulkan; состояния создаются сразу (не лениво),
    чтобы переопределить dtype v. Шаг идёт через фузированный кернел
    (torch.ops._fused_adamw_ -> adamw.slang), 1 диспетч на параметр.
    """

    def __init__(self, params, **kwargs):
        kwargs.setdefault("fused", True)
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {
                    "step": torch.zeros((), dtype=torch.float32, device=p.device),
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p, dtype=torch.float32),
                }
