import torch

class Pk :

    def __init__(self, x) :
        signal_ndim = x.dim() - 2
        signal_size = x.shape[-signal_ndim:]
        kmax = min(s for s in signal_size) // 2
        even = x.shape[-1] % 2 == 0

        try:
            x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
            p = x.real.square() + x.imag.square()
        except AttributeError:
            x = torch.rfft(x, signal_ndim)
            p = x.square().sum(dim=-1)

        p = p.mean(dim=0)
        p = p.sum(dim=0)
        del x

        k = [torch.arange(d, dtype=torch.float32, device=p.device)
             for d in p.shape]
        k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]]
        k = torch.meshgrid(*k, indexing='ij')
        k = torch.stack(k, dim=0)
        k = k.norm(p=2, dim=0)

        N = torch.full_like(p, 2, dtype=torch.int32)
        N[..., 0] = 1
        if even:
            N[..., -1] = 1

        k = k.flatten()
        p = p.flatten()
        N = N.flatten()

        kbin = k.ceil().to(torch.int32)
        k = torch.bincount(kbin, weights=k * N)
        p = torch.bincount(kbin, weights=p * N)
        N = torch.bincount(kbin, weights=N).round().to(torch.int32)
        del kbin

        # drop k=0 mode and cut at kmax (smallest Nyquist)
        k = k[1:1+kmax]
        p = p[1:1+kmax]
        N = N[1:1+kmax]

        self.k = (k / N).cpu().numpy()
        self.p = (p / N).cpu().numpy()

class XPk :

    def __init__(self, x, y) :
        signal_ndim = x.dim() - 2
        signal_size = x.shape[-signal_ndim:]
        kmax = min(s for s in signal_size) // 2
        even = x.shape[-1] % 2 == 0

        try:
            x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
            y = torch.fft.rfftn(y, s=signal_size)  # new version broke BC
            p11 = x.real.square() + x.imag.square()
            p22 = y.real.square() + y.imag.square()
            p12 = (x * y).abs()
        except AttributeError:
            x = torch.rfft(x, signal_ndim)
            y = torch.rfft(y, signal_ndim)
            p11 = (x.real.square() + x.imag.square()).sum(dim=-1)
            p22 = (y.real.square() + y.imag.square()).sum(dim=-1)
            p12 = ((x * y).abs().square()).sum(dim=-1)

        p11 = p11.mean(dim=0)
        p11 = p11.sum(dim=0)
        p22 = p22.mean(dim=0)
        p22 = p22.sum(dim=0)
        p12 = p12.mean(dim=0)
        p12 = p12.sum(dim=0)
        del x

        k = [torch.arange(d, dtype=torch.float32, device=p11.device)
             for d in p11.shape]
        k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]]
        k = torch.meshgrid(*k, indexing='ij')
        k = torch.stack(k, dim=0)
        k = k.norm(p=2, dim=0)

        N = torch.full_like(p11, 2, dtype=torch.int32)
        N[..., 0] = 1
        if even:
            N[..., -1] = 1

        k = k.flatten()
        p11 = p11.flatten()
        p22 = p22.flatten()
        p12 = p12.flatten()
        N = N.flatten()

        kbin = k.ceil().to(torch.int32)
        k = torch.bincount(kbin, weights=k * N)
        p11 = torch.bincount(kbin, weights=p11 * N)
        p22 = torch.bincount(kbin, weights=p22 * N)
        p12 = torch.bincount(kbin, weights=p12 * N)
        N = torch.bincount(kbin, weights=N).round().to(torch.int32)
        del kbin

        # drop k=0 mode and cut at kmax (smallest Nyquist)
        k = k[1:1+kmax]
        p11 = p11[1:1+kmax]
        p22 = p22[1:1+kmax]
        p12 = p12[1:1+kmax]
        N = N[1:1+kmax]

        self.k = (k / N).cpu().numpy()
        self.p11 = (p11 / N).cpu().numpy()
        self.p22 = (p22 / N).cpu().numpy()
        self.p12 = (p12 / N).cpu().numpy()
        self.s = (1 - (p12 / (p11 * p22).sqrt())).cpu().numpy()
        self.te = ((p11 / p22).sqrt() - 1).cpu().numpy()


def power(x):
    """Compute power spectra of input fields

    Each field should have batch and channel dimensions followed by spatial
    dimensions. Powers are summed over channels, and averaged over batches.

    Power is not normalized. Wavevectors are in unit of the fundamental
    frequency of the input.
    """
    signal_ndim = x.dim() - 2
    signal_size = x.shape[-signal_ndim:]
    kmax = min(s for s in signal_size) // 2
    even = x.shape[-1] % 2 == 0

    try:
        x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
        P = x.real.square() + x.imag.square()
    except AttributeError:
        x = torch.rfft(x, signal_ndim)
        P = x.square().sum(dim=-1)

    P = P.mean(dim=0)
    P = P.sum(dim=0)
    del x

    k = [torch.arange(d, dtype=torch.float32, device=P.device)
         for d in P.shape]
    k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]]
    k = torch.meshgrid(*k, indexing='ij')
    k = torch.stack(k, dim=0)
    k = k.norm(p=2, dim=0)

    N = torch.full_like(P, 2, dtype=torch.int32)
    N[..., 0] = 1
    if even:
        N[..., -1] = 1

    k = k.flatten()
    P = P.flatten()
    N = N.flatten()

    kbin = k.ceil().to(torch.int32)
    k = torch.bincount(kbin, weights=k * N)
    P = torch.bincount(kbin, weights=P * N)
    N = torch.bincount(kbin, weights=N).round().to(torch.int32)
    del kbin

    # drop k=0 mode and cut at kmax (smallest Nyquist)
    k = k[1:1+kmax]
    P = P[1:1+kmax]
    N = N[1:1+kmax]

    k /= N
    P /= N

    return k, P, N


