import math
import torch
from torch.distributions.kl import register_kl

from vmfmix.ive import ive
from vmfmix.hyperspherical_uniform import HypersphericalUniform
import math
import torch
from torch.distributions.kl import register_kl
import math
import torch
from torch.distributions.kl import register_kl
from .ive import ive

class VonMisesFisher(torch.distributions.Distribution):

    arg_constraints = {
        'loc': torch.distributions.constraints.real,
        'scale': torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):

        m = float(self.__m)
        scale = self.scale
        eps = torch.finfo(scale.dtype).tiny
        scale_safe = torch.clamp(scale, min=eps)

        s64 = scale_safe.to(torch.float64)
        nu_num = 0.5 * m
        nu_den = 0.5 * m - 1.0

        i_num = ive(nu_num, s64)
        i_den = ive(nu_den, s64)

        i_num = torch.where(torch.isfinite(i_num), i_num, torch.zeros_like(i_num))
        i_den = torch.where(
            torch.isfinite(i_den),
            i_den,
            torch.full_like(i_den, eps),
        )
        i_den = torch.clamp(i_den, min=eps)

        ratio = (i_num / i_den).to(scale.dtype)
        return self.loc * ratio

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]

        e1 = torch.zeros(self.__m, device=self.device, dtype=self.dtype)
        e1[0] = 1.0
        self.__e1 = e1

        super(VonMisesFisher, self).__init__(
            batch_shape=self.loc.size(),
            validate_args=validate_args,
        )

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        if self.__m == 3:
            w = self.__sample_w3(shape=shape)
        else:
            w = self.__sample_w_rej(shape=shape)

        v = torch.distributions.Normal(0, 1).sample(
            shape + torch.Size(self.loc.shape)
        ).to(self.device)
        v = v.transpose(0, -1)[1:].transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1.0 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), dim=-1)
        z = self.__householder_rotation(x)

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = 1 + torch.stack(
            [torch.log(u), torch.log(1 - u) - 2 * self.scale],
            dim=0,
        ).logsumexp(0) / self.scale
        return self.__w

    def __sample_w_rej(self, shape):
        m = float(self.__m)
        scale = self.scale

        c = torch.sqrt((4.0 * (scale ** 2)) + (m - 1.0) ** 2)
        b_true = (-2.0 * scale + c) / (m - 1.0)

        b_app = (m - 1.0) / (4.0 * torch.clamp(scale, min=1e-8))
        s = torch.min(
            torch.max(torch.tensor([0.0], device=self.device), scale - 10.0),
            torch.tensor([1.0], device=self.device),
        )
        b = b_app * s + b_true * (1.0 - s)

        a = (m - 1.0 + 2.0 * scale + c) / 4.0
        d = (4.0 * a * b) / (1.0 + b) - (m - 1.0) * math.log(m - 1.0)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape)
        return self.__w

    def __while_loop(self, b, a, d, shape):
        b, a, d = [
            e.repeat(*shape, *([1] * len(self.scale.shape)))
            for e in (b, a, d)
        ]
        w = torch.zeros_like(b, device=self.device)
        e = torch.zeros_like(b, device=self.device)
        bool_mask = torch.ones_like(b, dtype=torch.bool, device=self.device)

        shape = shape + torch.Size(self.scale.shape)
        m = float(self.__m)

        count = 0
        while bool_mask.any():
            e_ = torch.distributions.Beta(
                (m - 1.0) / 2.0, (m - 1.0) / 2.0
            ).sample(shape[:-1]).reshape(shape).to(self.device)
            u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)

            w_ = (1.0 - (1.0 + b) * e_) / (1.0 - (1.0 - b) * e_)
            t = (2.0 * a * b) / (1.0 - (1.0 - b) * e_)

            accept = ((m - 1.0) * t.log() - t + d) > torch.log(u)
            reject = ~accept

            w[bool_mask & accept] = w_[bool_mask & accept]
            e[bool_mask & accept] = e_[bool_mask & accept]

            bool_mask[bool_mask & accept] = reject[bool_mask & accept]
            count += 1

        return e, w

    def __householder_rotation(self, x):
        u = (self.__e1 - self.loc)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2.0 * (x * u).sum(-1, keepdim=True) * u
        return z

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):

        m = float(self.__m)
        scale = self.scale
        eps = torch.finfo(scale.dtype).tiny
        scale_safe = torch.clamp(scale, min=eps)

        s64 = scale_safe.to(torch.float64)
        nu_den = 0.5 * m - 1.0

        i_den = ive(nu_den, s64)
        i_den = torch.where(
            torch.isfinite(i_den),
            i_den,
            torch.full_like(i_den, eps),
        )
        i_den = torch.clamp(i_den, min=eps)

        log_ive = torch.log(i_den)  # double
        log_ive = log_ive.to(scale.dtype)
        log_scale = torch.log(scale_safe)
        m2 = m / 2.0

        log_c = (m2 - 1.0) * log_scale \
                - m2 * math.log(2.0 * math.pi) \
                - (scale_safe + log_ive)

        output = -log_c
        return output.view(*(output.shape[:-1]))

    def entropy(self):

        m = float(self.__m)
        scale = self.scale
        eps = torch.finfo(scale.dtype).tiny
        scale_safe = torch.clamp(scale, min=eps)

        s64 = scale_safe.to(torch.float64)
        nu_num = 0.5 * m
        nu_den = 0.5 * m - 1.0

        i_num = ive(nu_num, s64)
        i_den = ive(nu_den, s64)

        i_num = torch.where(torch.isfinite(i_num), i_num, torch.zeros_like(i_num))
        i_den = torch.where(
            torch.isfinite(i_den),
            i_den,
            torch.full_like(i_den, eps),
        )
        i_den = torch.clamp(i_den, min=eps)

        ratio = (i_num / i_den).to(scale.dtype)

        output = -scale_safe * ratio
        return output.view(*(output.shape[:-1])) + self._log_normalization()

    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return - vmf.entropy() + hyu.entropy()


