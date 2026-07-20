import torch
import torch.nn as nn
import math
import scipy.integrate as integrate
from typing import Any, Callable, List, Dict, Optional, Union, Tuple
from tqdm.auto import trange
import torchsde

# K-diffusion Sampler Helpers
def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma.view(-1, *([1] * (x.ndim - 1)))

def get_ancestral_step(sigma_from: float, sigma_to: float, eta: float = 1.0) -> Tuple[float, float]:
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.0
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def default_noise_sampler(x: torch.Tensor, seed: Optional[int] = None) -> Callable:
    if seed is not None:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None
    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)

class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""
    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = kwargs.pop("cpu", True)
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2 ** 63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        if self.cpu_tree:
            self.trees = [torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs) for s in seed]
        else:
            self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        if self.cpu_tree:
            w = torch.stack([tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device) for tree in self.trees]) * (self.sign * sign)
        else:
            w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]

class BrownianTreeNoiseSampler:
    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()

def sigma_to_half_log_snr(sigma: torch.Tensor, model_sampling: Any) -> torch.Tensor:
    return sigma.log().neg()

def offset_first_sigma_for_snr(sigmas: torch.Tensor, model_sampling: Any) -> torch.Tensor:
    return sigmas

@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: sigma.log().neg()
    old_denoised, old_old_denoised = None, None
    h_last, h_last_2 = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            l_s, l_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = l_t - l_s
            h_eta = h * (eta + 1)
            alpha_t = sigmas[i + 1] * l_t.exp()
            x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised
            if old_denoised is not None:
                r0 = h_last / h
                val_eta = -h_eta
                if old_old_denoised is None or sigmas[i + 1] == 0:
                    x = x + 0.5 * alpha_t * (val_eta.exp() - 1).neg() * (1 / r0) * (denoised - old_denoised)
                else:
                    r1 = h_last_2 / h
                    val_eta = -h_eta
                    x = x + alpha_t * ((val_eta.exp() - 1).neg() / (-h_eta) + 1) * (1 / r0) * (denoised - old_denoised)
            if eta > 0 and s_noise > 0:
                val_noise = torch.as_tensor(-2 * h * eta)
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (val_noise.exp() - 1).neg().sqrt() * s_noise
        old_old_denoised, old_denoised = old_denoised, denoised
        h_last_2, h_last = h_last, h
    return x

@torch.no_grad()
def sample_dpmpp_3m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    return sample_dpmpp_3m_sde(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)

@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + r * h
            val_eta = torch.as_tensor(-h * r)
            x_2 = (sigma_fn(s) / sigma_fn(t)) * x - (val_eta.exp() - 1) * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - ((-h).exp() - 1) * denoised_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * ((-2 * h * eta).exp() - 1).neg().sqrt() * s_noise
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    if len(sigmas) <= 1: return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            l_s, l_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = l_t - l_s
            h_eta = h * (eta + 1)
            alpha_t = sigmas[i + 1] * l_t.exp()
            val_eta = torch.as_tensor(-h * eta)
            val_h_eta = torch.as_tensor(-h_eta)
            x = sigmas[i + 1] / sigmas[i] * val_eta.exp() * x + alpha_t * (val_h_eta.exp() - 1).neg() * denoised
            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    val_eta_h = -h_eta
                    x = x + alpha_t * ((torch.as_tensor(val_eta_h).exp() - 1).neg() / (-h_eta) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    val_eta_m = -h_eta
                    x = x + 0.5 * alpha_t * (torch.as_tensor(val_eta_m).exp() - 1).neg() * (1 / r) * (denoised - old_denoised)
            if eta > 0 and s_noise > 0:
                val_noise = torch.as_tensor(-2 * h * eta)
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (val_noise.exp() - 1).neg().sqrt() * s_noise
        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    return sample_dpmpp_sde(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler, r)

@torch.no_grad()
def sample_dpmpp_2m_sde_gpu(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    return sample_dpmpp_2m_sde(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler, solver_type)

def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k: continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]

@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order: ds.pop(0)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            cur_order = min(i + 1, order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x

class PIDStepSizeController:
    def __init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs: self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept

class DPMSolver(nn.Module):
    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma): return -sigma.log()
    def sigma(self, t): return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache: return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None: self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, x, t_start, t_end, nfe, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x, seed=self.extra_args.get("seed", None)) if noise_sampler is None else noise_sampler
        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=x.device)
        orders = [3] * (m - 2) + [2, 1] if nfe % 3 == 0 else [3] * (m - 1) + [nfe % 3]
        for i in range(len(orders)):
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else: t_next_, su = t_next, 0.
            eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
            if orders[i] == 1: x, eps_cache = self.dpm_solver_1_step(x, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2: x, eps_cache = self.dpm_solver_2_step(x, t, t_next_, eps_cache=eps_cache)
            else: x, eps_cache = self.dpm_solver_3_step(x, t, t_next_, eps_cache=eps_cache)
            x = x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))
        return x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x, seed=self.extra_args.get("seed", None)) if noise_sampler is None else noise_sampler
        forward = t_end > t_start
        h_init = abs(h_init) * (1 if forward else -1)
        atol, rtol = torch.tensor(atol), torch.tensor(rtol)
        s, x_prev = t_start, x
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else: t_, su = t, 0.
            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            if pid.propose_step(error):
                x_prev, x, s = x_low, x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t)), t
        return x

@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1., noise_sampler=None):
    solver = DPMSolver(model, extra_args, info_callback=callback)
    return solver.dpm_solver_fast(x, solver.t(torch.tensor(sigma_max)), solver.t(torch.tensor(sigma_min)), n, eta, s_noise, noise_sampler)

@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
    solver = DPMSolver(model, extra_args, info_callback=callback)
    return solver.dpm_solver_adaptive(x, solver.t(torch.tensor(sigma_max)), solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)

@torch.no_grad()
def sample_deis(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=3):
    extra_args, x_next, t_steps, buffer_model = {} if extra_args is None else extra_args, x, sigmas, []
    s_in = x.new_ones([x.shape[0]])
    def get_rhoab_coeff(i, sigmas, order):
        prev_t = sigmas[[i - k for k in range(order + 1)]]
        t_cur, t_next = sigmas[i], sigmas[i + 1]
        if order == 1:
            c1 = ((t_next - prev_t[1])**2 - (t_cur - prev_t[1])**2) / (2 * (t_cur - prev_t[1]))
            c2 = (t_next - t_cur)**2 / (2 * (prev_t[1] - t_cur))
            return [c1, c2]
        def poly_int2(a, b, start, end, c):
             return ((end**3-start**3)/3 - (end**2-start**2)*(a+b)/2 + (end-start)*a*b) / ((c-a)*(c-b))
        if order == 2:
            return [poly_int2(prev_t[1], prev_t[2], t_cur, t_next, t_cur),
                    poly_int2(t_cur, prev_t[2], t_cur, t_next, prev_t[1]),
                    poly_int2(t_cur, prev_t[1], t_cur, t_next, prev_t[2])]
        return [0.0] * (order + 1)

    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur, t_next, x_cur = sigmas[i], sigmas[i + 1], x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur, order = (x_cur - denoised) / t_cur, min(max_order, i + 1)
        if t_next <= 0: x_next = denoised
        else:
            coeffs = get_rhoab_coeff(i, sigmas.cpu().numpy(), order)
            x_next = x_cur + coeffs[0] * d_cur + sum(c * b for c, b in zip(coeffs[1:], reversed(buffer_model)))
        buffer_model.append(d_cur)
        if len(buffer_model) > max_order: buffer_model.pop(0)
    return x_next

def compute_exponential_coeffs(s, t, solver_order, tau_t):
    tau_mul, h = 1 + tau_t ** 2, t - s
    p = torch.arange(solver_order, dtype=s.dtype, device=s.device)
    product_terms = (t ** p - s ** p * (-tau_mul * h).exp())
    depth = p.unsqueeze(1) - p.unsqueeze(0)
    log_fact = (p + 1).lgamma()
    coeff_mat = log_fact.unsqueeze(1) - log_fact.unsqueeze(0)
    if tau_t > 0: coeff_mat -= depth * math.log(tau_mul)
    signs = torch.where(depth % 2 == 0, 1.0, -1.0)
    coeff_mat = (coeff_mat.exp() * signs).tril()
    return coeff_mat @ product_terms

def compute_sa_b_coeffs(sigma_next, lambdas, l_s, l_t, tau_t):
    n = lambdas.shape[0]
    exp_ints = compute_exponential_coeffs(l_s, l_t, n, tau_t)
    vander = torch.vander(lambdas, n, increasing=True).T
    grads = torch.linalg.solve(vander, exp_ints)
    return sigma_next * l_t.exp() * grads

@torch.no_grad()
def sample_sa_solver(model, x, sigmas, extra_args=None, callback=None, disable=False, eta=1.0, predictor_order=3, corrector_order=4, use_pece=False):
    extra_args, s_in = {} if extra_args is None else extra_args, x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: sigma.log().neg()
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        l_s, l_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        tau = eta if i < len(sigmas) // 2 else 0.0
        curr_l = sigmas[max(0, i-predictor_order+1):i+1].log().neg()
        b_coeffs = compute_sa_b_coeffs(sigmas[i+1], curr_l, l_s, l_t, tau)
        x = (sigmas[i+1]/sigmas[i]) * x + (b_coeffs * denoised).sum()
        if sigmas[i+1] == 0: x = denoised
    return x

@torch.no_grad()
def sample_sa_solver_pece(model, x, sigmas, extra_args=None, callback=None, disable=False, eta=1.0):
    return sample_sa_solver(model, x, sigmas, extra_args, callback, disable, eta, use_pece=True)

@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    extra_args, seed = {} if extra_args is None else extra_args, extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0: x = denoised
        else:
            s_down, s_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            t, t_next = sigmas[i].log().neg(), s_down.log().neg()
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (s.neg().exp() / t.neg().exp()) * x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, s.neg().exp() * s_in, **extra_args)
            x = (t_next.neg().exp() / t.neg().exp()) * x - (-h).expm1() * denoised_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * s_up
    return x

@torch.no_grad()
def sample_ipndm(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=4):
    extra_args = {} if extra_args is None else extra_args
    s_in, x_next, buffer_model = x.new_ones([x.shape[0]]), x, []
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur, t_next, x_cur = sigmas[i], sigmas[i + 1], x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur, order = (x_cur - denoised) / t_cur, min(max_order, i+1)
        if t_next == 0: x_next = denoised
        elif order == 1: x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2: x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3: x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4: x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2): buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else: buffer_model.append(d_cur)
    return x_next

@torch.no_grad()
def sample_ipndm_v(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=4):
    extra_args, x_next, t_steps, buffer_model = {} if extra_args is None else extra_args, x, sigmas, []
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        t_cur, t_next, x_cur = sigmas[i], sigmas[i + 1], x_next
        denoised = model(x_cur, t_cur * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d_cur, order = (x_cur - denoised) / t_cur, min(max_order, i+1)
        if t_next == 0: x_next = denoised
        elif order == 1: x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:
            h_n, h_n_1 = (t_next - t_cur), (t_cur - t_steps[i-1])
            coeff1, coeff2 = (2 + (h_n / h_n_1)) / 2, -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:
            h_n, h_n_1, h_n_2 = (t_next - t_cur), (t_cur - t_steps[i-1]), (t_steps[i-1] - t_steps[i-2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1, coeff2, coeff3 = (2 + (h_n / h_n_1)) / 2 + temp, -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp, temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:
            h_n, h_n_1, h_n_2, h_n_3 = (t_next - t_cur), (t_cur - t_steps[i-1]), (t_steps[i-1] - t_steps[i-2]), (t_steps[i-2] - t_steps[i-3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1, coeff2 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2, -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2): buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else: buffer_model.append(d_cur)
    return x_next

def DDPMSampler_step(x, sigma, sigma_prev, noise, noise_sampler):
    sigma_t = torch.as_tensor(sigma)
    sigma_p = torch.as_tensor(sigma_prev)
    alpha_cumprod = 1 / ((sigma_t * sigma_t) + 1)
    alpha_cumprod_prev = 1 / ((sigma_p * sigma_p) + 1)
    alpha = (alpha_cumprod / alpha_cumprod_prev)
    mu = (1.0 / alpha).sqrt() * (x - (1 - alpha) * noise / (1 - alpha_cumprod).sqrt())
    if sigma_prev > 0:
        mu += ((1 - alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * noise_sampler(sigma, sigma_prev)
    return mu

def generic_step_sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, step_function=None):
    extra_args, seed = {} if extra_args is None else extra_args, extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = step_function(x / torch.sqrt(1.0 + sigmas[i] ** 2.0), sigmas[i], sigmas[i + 1], (x - denoised) / sigmas[i], noise_sampler)
        if sigmas[i + 1] != 0: x *= torch.sqrt(1.0 + sigmas[i + 1] ** 2.0)
    return x

@torch.no_grad()
def sample_ddpm(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    return generic_step_sampler(model, x, sigmas, extra_args, callback, disable, noise_sampler, DDPMSampler_step)

@torch.no_grad()
def sample_heunpp2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args, s_in, s_end = {} if extra_args is None else extra_args, x.new_ones([x.shape[0]]), sigmas[-1]
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0: x = x + torch.randn_like(x) * s_noise * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == s_end: x = x + d * dt
        elif sigmas[i + 2] == s_end:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1], model(x_2, sigmas[i + 1] * s_in, **extra_args))
            w = 2 * sigmas[0]
            w2 = sigmas[i+1]/w
            x = x + (d * (1 - w2) + d_2 * w2) * dt
        else:
            x_2 = x + d * dt
            d_2 = to_d(x_2, sigmas[i + 1], model(x_2, sigmas[i + 1] * s_in, **extra_args))
            dt_2 = sigmas[i + 2] - sigmas[i + 1]
            x_3 = x_2 + d_2 * dt_2
            d_3 = to_d(x_3, sigmas[i + 2], model(x_3, sigmas[i + 2] * s_in, **extra_args))
            w = 3 * sigmas[0]
            w2, w3 = sigmas[i+1]/w, sigmas[i+2]/w
            x = x + ((1 - w2 - w3) * d + w2 * d_2 + w3 * d_3) * dt
    return x

@torch.no_grad()
def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0):
    extra_args, s_in = {} if extra_args is None else extra_args, x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0: x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[ i]
            x = x + d * dt + torch.randn_like(x) * s_noise * (sigmas[i]**2 - sigmas[i+1]**2)**0.5
    return x

@torch.no_grad()
def sample_seeds_2(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., r=0.5):
    extra_args, s_in = {} if extra_args is None else extra_args, x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0: x = denoised
        else:
            s_down, s_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            sigma_mid = torch.as_tensor(sigmas[i]).log().lerp(torch.as_tensor(s_down).log(), r).exp()
            x_2 = x + to_d(x, sigmas[i], denoised) * (sigma_mid - sigmas[i])
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            x = x + to_d(x_2, sigma_mid, denoised_2) * (s_down - sigmas[i])
            x = x + torch.randn_like(x) * s_noise * s_up
    return x

@torch.no_grad()
def sample_seeds_3(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., r_1=1./3, r_2=2./3):
    extra_args, s_in = {} if extra_args is None else extra_args, x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0: x = denoised
        else:
            s_down, s_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            sigma_1 = sigmas[i].log().lerp(s_down.log(), r_1).exp()
            sigma_2 = sigmas[i].log().lerp(s_down.log(), r_2).exp()
            x_2 = x + to_d(x, sigmas[i], denoised) * (sigma_1 - sigmas[i])
            denoised_2 = model(x_2, sigma_1 * s_in, **extra_args)
            x_3 = x + (sigma_2 - sigmas[i]) / (sigma_1 - sigmas[i]) * (x_2 - x)
            denoised_3 = model(x_3, sigma_2 * s_in, **extra_args)
            x = x + to_d(x_3, sigma_2, denoised_3) * (s_down - sigmas[i])
            x = x + torch.randn_like(x) * s_noise * s_up
    return x

@torch.no_grad()
def res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., eta=1., cfg_pp=False):
    extra_args, s_in = {} if extra_args is None else extra_args, x.new_ones([x.shape[0]])
    old_denoised, old_sigma_down = None, None
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0 or old_denoised is None:
            x = x + to_d(x, sigmas[i], denoised) * (sigma_down - sigmas[i])
        else:
            t, t_next = sigmas[i].log().neg(), sigma_down.log().neg()
            h = t_next - t
            x = (t_next.neg().exp() / t.neg().exp()) * x - (h.exp() - 1.0) * denoised
        if sigmas[i + 1] > 0: x = x + torch.randn_like(x) * s_noise * sigma_up
        old_denoised, old_sigma_down = denoised, sigma_down
    return x

@torch.no_grad()
def sample_res_multistep(model, x, sigmas, extra_args=None, callback=None, disable=None):
    return res_multistep(model, x, sigmas, extra_args, callback, disable)

@torch.no_grad()
def sample_gradient_estimation(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2.):
    extra_args, s_in, old_d = {} if extra_args is None else extra_args, x.new_ones([x.shape[0]]), None
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = to_d(x, sigmas[i], denoised)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        dt = sigmas[i + 1] - sigmas[i]
        if sigmas[i + 1] == 0: x = denoised
        else:
            x = x + d * dt
            if i >= 1: x = x + (ge_gamma - 1) * (d - old_d) * dt
        old_d = d
    return x

@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_churn > 0 and s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x

@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            x = denoised
        else:
            d = to_d(x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            x = x + d * dt + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_churn > 0 and s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x

@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None: callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x
