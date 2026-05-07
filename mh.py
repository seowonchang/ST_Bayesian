"""
Metropolis-Hastings sampler for Bayesian hierarchical SRT model
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln as scipy_gammaln
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Optional
from numba import njit, prange
import math
import time


# log pmf terms that only depend on mu
@njit(cache=True)
def _nb_ll_mu_only(y, mu, phi, N):
    out = 0.0
    phii = max(phi, 1e-12)
    for i in range(N):
        mui = max(mu[i], 1e-12)
        out += -(phii + y[i]) * math.log(phii + mui) + y[i] * math.log(mui)
    return out


# full log pmf for a gene
@njit(cache=True)
def _nb_logpmf_gene(y, mu, phi, lgamma_y1):
    out = 0.0
    N = y.shape[0]
    phii = max(phi, 1e-12)
    lgamma_phi = math.lgamma(phii)
    log_phi = math.log(phii)
    for i in range(N):
        mui = max(mu[i], 1e-12)
        log_denom = math.log(phii + mui)
        out += (math.lgamma(y[i] + phii) - lgamma_phi - lgamma_y1[i]
                + phii * (log_phi - log_denom) + y[i] * (math.log(mui) - log_denom))
    return out


# recompute after gene j's params change
@njit(cache=True)
def _recompute_gene(j, beta, gamma, Z_aug, G_indptr, G_indices,
                    G_vals, G_rowsum, size_fac, theta, lam, mu, L_aug, N):
    # cross slice
    for i in range(N):
        val = 0.0
        for l in range(L_aug):
            val += beta[l, j] * gamma[l, j] * Z_aug[i, l]
        val = max(-30.0, min(30.0, val))
        theta[i, j] = math.exp(val)
    # within slice
    for i in range(N):
        s = 0.0
        for k in range(G_indptr[i], G_indptr[i + 1]):
            s += G_vals[k] * theta[G_indices[k], j]
        lam[i, j] = s / G_rowsum[i]
        mu[i, j] = size_fac[i] * lam[i, j]

    
# update gamma step
@njit(parallel=True, cache=True)
def _update_gamma_kuo_mallick(
    Y, mu, phi, beta, gamma, theta, lam, size_fac,
    Z_aug, G_indptr, G_indices, G_vals, G_rowsum,
    comp_ids, log_us, gene_starts, gene_counts, N, P, L_aug
):
    n_accept = 0
    for j in prange(P):
        cnt = gene_counts[j]
        if cnt == 0:
            continue
        start = gene_starts[j]
        phi_j = phi[j]
        for q in range(cnt):
            idx = start + q
            l = comp_ids[idx]
            if l == 0:
                continue
            old_g = gamma[l, j]
            ll_old = _nb_ll_mu_only(Y[:, j], mu[:, j], phi_j, N)
            gamma[l, j] = 1.0 - old_g
            _recompute_gene(j, beta, gamma, Z_aug, G_indptr, G_indices,
                            G_vals, G_rowsum, size_fac, theta, lam, mu, L_aug, N)
            ll_new = _nb_ll_mu_only(Y[:, j], mu[:, j], phi_j, N)
            if log_us[idx] < (ll_new - ll_old):
                n_accept += 1
            else:
                gamma[l, j] = old_g
                _recompute_gene(j, beta, gamma, Z_aug, G_indptr, G_indices,
                                G_vals, G_rowsum, size_fac, theta, lam, mu, L_aug, N)
    return n_accept


# update betas step
@njit(parallel=True, cache=True)
def _update_all_betas_parallel(
    Y, mu, phi, beta, gamma, theta, lam, size_fac,
    Z_aug, G_indptr, G_indices, G_vals, G_rowsum,
    eps, log_u, prior_var, N, P, L_aug
):
    n_accept = 0
    n_total = 0
    for j in prange(P):
        phi_j = phi[j]
        for l in range(L_aug):
            n_total += 1
            old_b = beta[l, j]
            new_b = old_b + eps[l, j]
            inv_2var = 0.5 / prior_var[l]
            if gamma[l, j] == 1.0:
                ll_old = _nb_ll_mu_only(Y[:, j], mu[:, j], phi_j, N)
                lp_old = -inv_2var * old_b * old_b
                beta[l, j] = new_b
                _recompute_gene(j, beta, gamma, Z_aug, G_indptr, G_indices,
                                G_vals, G_rowsum, size_fac, theta, lam, mu, L_aug, N)
                ll_new = _nb_ll_mu_only(Y[:, j], mu[:, j], phi_j, N)
                lp_new = -inv_2var * new_b * new_b
                if log_u[l, j] < (ll_new + lp_new) - (ll_old + lp_old):
                    n_accept += 1
                else:
                    beta[l, j] = old_b
                    _recompute_gene(j, beta, gamma, Z_aug, G_indptr, G_indices,
                                    G_vals, G_rowsum, size_fac, theta, lam, mu, L_aug, N)
            else:
                log_alpha = inv_2var * (old_b * old_b - new_b * new_b)
                if log_u[l, j] < log_alpha:
                    beta[l, j] = new_b
                    n_accept += 1
    return n_accept, n_total


# update phi step
@njit(parallel=True, cache=True)
def _update_phi_parallel(Y, mu, phi, lgamma_Y1, eps, log_u, a, b, N, P):
    n_accept = 0
    for j in prange(P):
        phi_old = phi[j]
        log_phi_old = math.log(phi_old)
        log_phi_new = log_phi_old + eps[j]
        phi_new = math.exp(log_phi_new)
        phi_old_s = max(phi_old, 1e-12)
        phi_new_s = max(phi_new, 1e-12)
        lg_phi_old = math.lgamma(phi_old_s)
        lg_phi_new = math.lgamma(phi_new_s)
        lp_old_val = math.log(phi_old_s)
        lp_new_val = math.log(phi_new_s)
        ll_old = 0.0
        ll_new = 0.0
        for i in range(N):
            yi = Y[i, j]
            mui = max(mu[i, j], 1e-12)
            log_mui = math.log(mui)
            ld_old = math.log(phi_old_s + mui)
            ll_old += (math.lgamma(yi + phi_old_s) - lg_phi_old - lgamma_Y1[i, j]
                       + phi_old_s * (lp_old_val - ld_old) + yi * (log_mui - ld_old))
            ld_new = math.log(phi_new_s + mui)
            ll_new += (math.lgamma(yi + phi_new_s) - lg_phi_new - lgamma_Y1[i, j]
                       + phi_new_s * (lp_new_val - ld_new) + yi * (log_mui - ld_new))
        prior_old = (a - 1.0) * log_phi_old - b * phi_old
        prior_new = (a - 1.0) * log_phi_new - b * phi_new
        log_jac = log_phi_new - log_phi_old
        if log_u[j] < (ll_new + prior_new) - (ll_old + prior_old) + log_jac:
            phi[j] = phi_new
            n_accept += 1
    return n_accept


# helpers
def build_knn_adjacency(coords, k):
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    N = coords.shape[0]
    rows = np.repeat(np.arange(N), k + 1)
    cols = idx.flatten()
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(N, N))

def compute_size_factors(Y):
    lib_sizes = Y.sum(axis=1).astype(np.float64)
    lib_sizes = np.maximum(lib_sizes, 1.0)
    log_geo_mean = np.mean(np.log(lib_sizes))
    return lib_sizes / np.exp(log_geo_mean)

def log_likelihood(Y, mu, phi, lgamma_Y1):
    total = 0.0
    for j in range(Y.shape[1]):
        total += _nb_logpmf_gene(Y[:, j], mu[:, j], phi[j], lgamma_Y1[:, j])
    return total

def log_posterior(Y, params, data):
    ll = log_likelihood(Y, params.mu, params.phi, data.lgamma_Y1)
    lp_phi = np.sum((0.1 - 1) * np.log(params.phi) - 0.1 * params.phi)
    lp_beta = 0.0
    for l in range(data.L_aug):
        lp_beta += np.sum(-0.5 * params.beta[l, :]**2 / data.prior_var[l])
    return ll + lp_phi + lp_beta


# slice data structure
@dataclass
class SliceData:
    Y: np.ndarray
    T: np.ndarray
    G: csr_matrix = None
    X_ref: np.ndarray = None
    T_ref: np.ndarray = None
    knn_ref_idx: np.ndarray = None
    X_agg: np.ndarray = None
    Z_aug: np.ndarray = None
    lgamma_Y1: np.ndarray = None
    prior_var: np.ndarray = None
    s: np.ndarray = None
    K_intra: int = 6
    K_cross: int = 6
    L: int = 10
    G_indptr: np.ndarray = None
    G_indices: np.ndarray = None
    G_vals: np.ndarray = None
    G_rowsum: np.ndarray = None

    @property
    def L_aug(self):
        return self.L + 1

    def __post_init__(self):
        self.Y = np.ascontiguousarray(self.Y, dtype=np.float64)
        self.s = compute_size_factors(self.Y)
        G = build_knn_adjacency(self.T, self.K_intra)
        self.G = G
        self.G_indptr = np.ascontiguousarray(G.indptr)
        self.G_indices = np.ascontiguousarray(G.indices)
        self.G_vals = np.ascontiguousarray(G.data)
        self.G_rowsum = np.asarray(G.sum(axis=1)).flatten()
        self.lgamma_Y1 = scipy_gammaln(self.Y + 1.0)

    def set_reference(self, Y_ref, T_ref, L=None, sigma_intercept=10.0):
        # normalize
        Y_ref = Y_ref.astype(np.float64)
        lib_sizes = Y_ref.sum(axis=1, keepdims=True)
        lib_sizes = np.maximum(lib_sizes, 1.0)
        Y_norm = Y_ref / lib_sizes * np.median(lib_sizes)
        Y_norm = np.log1p(Y_norm)
        gene_means = Y_norm.mean(axis=0, keepdims=True)
        gene_stds = Y_norm.std(axis=0, keepdims=True)
        gene_stds = np.maximum(gene_stds, 1e-8)
        Y_norm = (Y_norm - gene_means) / gene_stds
        # pca
        pca = PCA(n_components=self.L)
        self.X_ref = pca.fit_transform(Y_norm)
        # design matrix
        self.T_ref = T_ref
        tree = cKDTree(T_ref)
        _, self.knn_ref_idx = tree.query(self.T, k=self.K_cross)
        self.X_agg = self.X_ref[self.knn_ref_idx].sum(axis=1)
        N = self.Y.shape[0]
        ones = np.ones((N, 1), dtype=np.float64)
        self.Z_aug = np.ascontiguousarray(
            np.hstack([ones, self.X_agg]), dtype=np.float64)
        self.prior_var = np.ones(self.L + 1, dtype=np.float64)
        self.prior_var[0] = sigma_intercept ** 2


# parameters data structure
@dataclass
class Parameters:
    beta: np.ndarray
    gamma: np.ndarray
    phi: np.ndarray
    theta: np.ndarray = None
    lam: np.ndarray = None
    mu: np.ndarray = None

    @staticmethod
    def initialize(data, rng):
        N, P = data.Y.shape
        L_aug = data.L_aug
        beta = rng.normal(0, 0.1, size=(L_aug, P)).astype(np.float64)
        beta[0, :] = np.log(data.Y.mean(axis=0).clip(min=1e-3))
        gamma = rng.binomial(1, 0.5, size=(L_aug, P)).astype(np.float64)
        gamma[0, :] = 1.0
        phi = np.ones(P, dtype=np.float64)
        params = Parameters(beta, gamma, phi)
        params.theta = np.zeros((N, P), dtype=np.float64, order='F')
        params.lam = np.zeros((N, P), dtype=np.float64, order='F')
        params.mu = np.zeros((N, P), dtype=np.float64, order='F')
        params.recompute_all(data)
        return params

    def recompute_all(self, data):
        active_beta = self.beta * self.gamma
        log_theta = data.Z_aug @ active_beta
        np.clip(log_theta, -30, 30, out=log_theta)
        np.exp(log_theta, out=self.theta)
        result = data.G @ self.theta
        np.divide(result, data.G_rowsum[:, None], out=self.lam)
        np.multiply(data.s[:, None], self.lam, out=self.mu)


# metropolis hastings sampler
class MHSampler:
    def __init__(self, data, n_iter=2000, burn_in=500, thin=1, seed=42,
                 proposal_sd=None):
        self.data = data
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.rng = np.random.default_rng(seed)
        self.N, self.P = data.Y.shape
        self.L = data.L
        self.L_aug = data.L_aug
        self.params = Parameters.initialize(data, self.rng)
        defaults = {"beta": 0.3, "gamma": 5, "log_phi": 0.5}
        self.prop_sd = {**defaults, **(proposal_sd or {})}
        self.phi_a = 0.1
        self.phi_b = 0.1
        n_keep = max(1, (n_iter - burn_in) // thin)
        self.trace_beta = np.zeros((n_keep, self.L_aug, self.P))
        self.trace_gamma = np.zeros((n_keep, self.L_aug, self.P))
        self.trace_phi = np.zeros((n_keep, self.P))
        self.trace_lp = np.zeros(n_keep)
        self.accept = {"beta": 0, "gamma": 0, "phi": 0}
        self.total = {"beta": 0, "gamma": 0, "phi": 0}

    def _update_phi(self):
        eps = self.rng.normal(0, self.prop_sd["log_phi"], size=self.P)
        log_u = np.log(self.rng.random(self.P))
        n_acc = _update_phi_parallel(
            self.data.Y, self.params.mu, self.params.phi,
            self.data.lgamma_Y1, eps, log_u,
            self.phi_a, self.phi_b, self.N, self.P)
        self.accept["phi"] += n_acc
        self.total["phi"] += self.P

    def _update_gamma(self):
        n_props = max(1, self.L_aug * self.P // self.prop_sd["gamma"])
        flat_idx = self.rng.choice(self.L_aug * self.P, n_props, replace=False)
        comp_ids = (flat_idx // self.P).astype(np.int64)
        gene_ids = (flat_idx % self.P).astype(np.int64)
        log_us = np.log(self.rng.random(n_props))
        order = np.argsort(gene_ids)
        gene_ids, comp_ids, log_us = gene_ids[order], comp_ids[order], log_us[order]
        gene_starts = np.zeros(self.P, dtype=np.int64)
        gene_counts = np.zeros(self.P, dtype=np.int64)
        for k in range(n_props):
            gene_counts[gene_ids[k]] += 1
        cs = 0
        for j in range(self.P):
            gene_starts[j] = cs
            cs += gene_counts[j]
        n_acc = _update_gamma_kuo_mallick(
            self.data.Y, self.params.mu, self.params.phi,
            self.params.beta, self.params.gamma,
            self.params.theta, self.params.lam, self.data.s,
            self.data.Z_aug, self.data.G_indptr, self.data.G_indices,
            self.data.G_vals, self.data.G_rowsum,
            comp_ids, log_us, gene_starts, gene_counts,
            self.N, self.P, self.L_aug)
        self.accept["gamma"] += n_acc
        self.total["gamma"] += n_props

    def _update_beta(self):
        eps = self.rng.normal(0, self.prop_sd["beta"], size=(self.L_aug, self.P))
        log_u = np.log(self.rng.random((self.L_aug, self.P)))
        n_acc, n_tot = _update_all_betas_parallel(
            self.data.Y, self.params.mu, self.params.phi,
            self.params.beta, self.params.gamma,
            self.params.theta, self.params.lam, self.data.s,
            self.data.Z_aug, self.data.G_indptr, self.data.G_indices,
            self.data.G_vals, self.data.G_rowsum,
            eps, log_u, self.data.prior_var,
            self.N, self.P, self.L_aug)
        self.accept["beta"] += n_acc
        self.total["beta"] += n_tot

    def run(self, verbose=True):
        store_idx = 0
        t_start = time.perf_counter()
        for it in range(self.n_iter):
            self._update_phi()
            self._update_gamma()
            self._update_beta()
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                self.trace_beta[store_idx] = self.params.beta.copy()
                self.trace_gamma[store_idx] = self.params.gamma.copy()
                self.trace_phi[store_idx] = self.params.phi.copy()
                self.trace_lp[store_idx] = log_posterior(self.data.Y, self.params, self.data)
                store_idx += 1
            if verbose and (it + 1) % max(1, self.n_iter // 10) == 0:
                elapsed = time.perf_counter() - t_start
                lp = log_posterior(self.data.Y, self.params, self.data)
                n_obs = self.N * self.P
                print(f"Iter {it+1:>6d}/{self.n_iter}  "
                      f"lp/obs={lp / n_obs:+.4f}  "
                      f"g_bar={self.params.gamma[1:].mean():.3f}  "
                      f"[{elapsed:.1f}s elapsed]")
        if verbose:
            total = time.perf_counter() - t_start
            print(f"\nTotal: {total:.2f}s  ({total/self.n_iter*1000:.1f} ms/iter)")
            print("\n--- Acceptance rates ---")
            for key in self.accept:
                tot = self.total[key]
                rate = self.accept[key] / tot if tot > 0 else 0
                print(f"  {key:>8s}: {rate:.3f}  ({self.accept[key]}/{tot})")

    def posterior_mean_beta0(self):
        return self.trace_beta[:, 0, :].mean(axis=0)
    def posterior_mean_beta(self):
        return self.trace_beta[:, 1:, :].mean(axis=0)
    def posterior_mean_phi(self):
        return self.trace_phi.mean(axis=0)
    def pip(self):
        return self.trace_gamma[:, 1:, :].mean(axis=0)


# simulated data
def generate_synthetic_data(N=200, P=10, L=5, K_intra=6, K_cross=6, seed=0):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(N)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1
                    ).reshape(-1, 2)[:N].astype(float)
    T = grid + rng.normal(0, 0.1, grid.shape)
    T_ref = grid + rng.normal(0, 0.2, grid.shape)
    beta0_true = rng.normal(2, 0.5, size=P)
    gamma_true = rng.binomial(1, 0.3, size=(L, P)).astype(float)
    beta_true = rng.normal(0, 1, size=(L, P)) * gamma_true
    phi_true = rng.gamma(2, 1, size=P)
    X_ref = rng.normal(0, 1, size=(N, L))
    tree = cKDTree(T_ref)
    _, knn_idx = tree.query(T, k=K_cross)
    X_agg = X_ref[knn_idx].sum(axis=1)
    X_agg_std = X_agg.std(axis=0, keepdims=True)
    X_agg_std[X_agg_std < 1e-8] = 1.0
    X_agg = X_agg / X_agg_std
    G = build_knn_adjacency(T, K_intra)
    G_rowsum = np.asarray(G.sum(axis=1)).flatten()
    log_theta = beta0_true[None, :] + X_agg @ (beta_true * gamma_true)
    log_theta = np.clip(log_theta, -20, 20)
    theta = np.exp(log_theta)
    lam = (G @ theta) / G_rowsum[:, None]
    Y = rng.negative_binomial(n=phi_true[None, :],
                               p=phi_true[None, :] / (phi_true[None, :] + lam))
    data = SliceData(Y=Y, T=T, K_intra=K_intra, K_cross=K_cross, L=L)
    data.set_reference(Y_ref=rng.poisson(5, (N, P)), T_ref=T_ref, L=L)
    data.X_ref = X_ref
    data.knn_ref_idx = knn_idx
    data.X_agg = X_agg
    ones = np.ones((N, 1), dtype=np.float64)
    data.Z_aug = np.ascontiguousarray(np.hstack([ones, X_agg]), dtype=np.float64)
    data.s = np.ones(N, dtype=np.float64)
    truth = {"beta0": beta0_true, "beta": beta_true,
             "gamma": gamma_true, "phi": phi_true}
    return data, truth
