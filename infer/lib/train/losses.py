import torch


def feature_loss(
    fmap_r: list[list[torch.Tensor]], fmap_g: list[list[torch.Tensor]]
) -> torch.Tensor:
    if not fmap_r or not fmap_r[0]:
        return torch.tensor(0.0)
    loss = torch.zeros((), device=fmap_g[0][0].device)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: list[torch.Tensor], disc_generated_outputs: list[torch.Tensor]
) -> tuple[torch.Tensor, list[float], list[float]]:
    if not disc_real_outputs or not disc_generated_outputs:
        return torch.tensor(0.0), [], []
    loss = torch.zeros((), device=disc_generated_outputs[0].device)
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1.0 - dr).pow(2))
        g_loss = torch.mean(dg.pow(2))
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: list[torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if not disc_outputs:
        return torch.tensor(0.0), []
    loss = torch.zeros((), device=disc_outputs[0].device)
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1.0 - dg).pow(2))
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p).pow(2)) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
