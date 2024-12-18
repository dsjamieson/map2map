import numpy as np
from scipy.special import hyp2f1

def dis(x, undo=False, z=0.0, dis_std=6.0, Om=0.31, **kwargs):
    dis_norm = dis_std * D(z, Om)  # [Mpc/h]

    if not undo:
        dis_norm = 1 / dis_norm

    x *= dis_norm

def vel(x, undo=False, z=0.0, dis_std=6.0, Om=0.31, **kwargs):
    vel_norm = dis_std * D(z, Om) * H(z, Om) * f(z, Om) / (1 + z)  # [km/s]

    if not undo:
        vel_norm = 1 / vel_norm

    x *= vel_norm

def acc(x, undo=False, z=0.0, dis_std=6.0, Om=0.31, **kwargs):
    acc_norm = dis_std * D(z, Om) * H(z, Om) ** 2 * (growth_acc(z, Om) + dlogH_dloga(z, Om) * f(z, Om)) / (1 + z) # km / s^2

    if not undo:
        acc_norm = 1 / acc_norm

    x *= acc_norm


_a2f1 = 1
_b2f1 = 1/3
_c2f1 = 11/6

def growth_2f1(x) :
    return hyp2f1(_a2f1, _b2f1, _c2f1, x)

def D(z, Om=0.31) -> np.float64:
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero
    """
    a = 1 / (1 + z)
    OL = 1 - Om
    aa3 = -OL * a**3 / Om
    aa30 = -OL / Om
    return a * growth_2f1(aa3) / growth_2f1(aa30)

def d_growth_2f1(x) :
    return _a2f1 * _b2f1 / _c2f1 * hyp2f1(_a2f1 + 1, _b2f1 + 1, _c2f1 + 1, x)

def f(z, Om=0.31):
    """linear growth rate for flat LambdaCDM
    """
    a = 1 / (1 + z)
    OL = 1 - Om
    aa3 = -OL * a**3 / Om
    return 1 + 3 * aa3 * d_growth_2f1(aa3) / growth_2f1(aa3)

def dd_growth_2f1(x) :
    return _a2f1 * (_a2f1 + 1) * _b2f1 * (_b2f1 + 1) / _c2f1 / (_c2f1 + 1) * hyp2f1(_a2f1 + 2, _b2f1 + 2, _c2f1 + 2, x)

def growth_acc(z, Om=32) :
    """ d^2 D / (dloga)^2 / D
    """
    a = 1 / (1 + z)
    OL = 1 - Om
    aa3 = -OL * a**3 / Om
    return 1 + (12 * aa3 * d_growth_2f1(aa3) + 9 * (aa3) ** 2 * dd_growth_2f1(aa3)) / growth_2f1(aa3)

def H(z, Om=0.31):
    """Hubble in [h km/s/Mpc] for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1+z)
    return 100 * np.sqrt(Om / a**3 + OL)


def dlogH_dloga(z, Om=0.31):
    """ log-log derivative of Hubble w.r.t scale factor for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1+z)
    aa3 = OL * a**3 / Om
    return -1.5 / (1 + aa3)
