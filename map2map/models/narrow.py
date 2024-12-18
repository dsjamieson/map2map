import torch
import torch.nn as nn

#def narrow_by(a, c):
#    """Narrow a by size c symmetrically on all edges.
#    """
#    ind = (slice(None),) * 2 + (slice(c, -c),) * (a.dim() - 2)
#    return a[ind]

@torch.jit.script
def narrow_by(a: torch.Tensor, c: int) -> torch.Tensor:
    """
    Narrow a tensor symmetrically on all edges by size c.
    """
    dims = a.dim()
    if dims == 3:
        return a[:, :, c:-c]
    elif dims == 4:
        return a[:, :, c:-c, c:-c]
    elif dims == 5:
        return a[:, :, c:-c, c:-c, c:-c]
    else:
        raise ValueError("narrow_by only supports tensors with 3, 4, or 5 dimensions.")

def narrow_cast(*tensors):
    """Narrow each tensor to the minimum length in each dimension.

    Try to be symmetric but cut more on the right for odd difference
    """
    dim_max = max(a.dim() for a in tensors)

    len_min = {d: min(a.shape[d] for a in tensors) for d in range(2, dim_max)}

    casted_tensors = []
    for a in tensors:
        for d in range(2, dim_max):
            width = a.shape[d] - len_min[d]
            half_width = width // 2
            a = a.narrow(d, half_width, a.shape[d] - width)

        casted_tensors.append(a)

    return casted_tensors


def narrow_like(a, b):
    """Narrow a to be like b.

    Try to be symmetric but cut more on the right for odd difference
    """
    for d in range(2, a.dim()):
        width = a.shape[d] - b.shape[d]
        half_width = width // 2
        a = a.narrow(d, half_width, a.shape[d] - width)
    return a
