import numpy as np
from scipy.special import gammaln
from sklearn.linear_model import Ridge, Lasso
import warnings
import time


def _mom_phi(Y, mu):
    mu = np.maximum(mu, 1e-8)
    P = Y.shape[1]
    phi = np.ones(P)
    for j in range(P):
        sample_var = np.var(Y[:, j])
        mean_mu = np.mean(mu[:, j])
        excess = sample_var - mean_mu
        if excess > 1e-8:
            phi[j] = mean_mu ** 2 / excess
        else:
            phi[j] = 100.0
    return np.clip(phi, 1e-4, 1e4)


def nb_loglik(Y, mu, phi):
    mu = np.maximum(mu, 1e-12)
    phi2 = np.maximum(phi[None, :], 1e-12)
    ll = (gammaln(Y + phi2) - gammaln(phi2) - gammaln(Y + 1)
          + phi2 * np.log(phi2 / (phi2 + mu))
          + Y * np.log(mu / (phi2 + mu)))
    return ll.sum()


def mse_per_spot(Y, mu):
    """MSE on log(y+1) vs log(mu+1), averaged over genes per spot."""
    log_Y = np.log1p(np.maximum(Y, 0).astype(np.float64))
    log_mu = np.log1p(np.maximum(mu, 0))
    per_spot = np.mean((log_Y - log_mu) ** 2, axis=1)
    return per_spot.mean(), per_spot


def mae_per_spot(Y, mu):
    """MAE on log(y+1) vs log(mu+1), averaged over genes per spot."""
    log_Y = np.log1p(np.maximum(Y, 0).astype(np.float64))
    log_mu = np.log1p(np.maximum(mu, 0))
    per_spot = np.mean(np.abs(log_Y - log_mu), axis=1)
    return per_spot.mean(), per_spot


def evaluate(Y, mu_hat, phi_hat, label, truth=None):
    N, P = Y.shape
    ll = nb_loglik(Y, mu_hat, phi_hat)
    mse_val, mse_spots = mse_per_spot(Y, mu_hat)
    mae_val, mae_spots = mae_per_spot(Y, mu_hat)
    results = {
        "method": label,
        "nb_loglik": ll,
        "nb_loglik_per_obs": ll / (N * P),
        "mse": mse_val,
        "mse_std": mse_spots.std(),
        "mae": mae_val,
        "mae_std": mae_spots.std(),
    }
    return results


def knn_average(data):
    Y = data.Y
    G = data.G.toarray()
    G_rowsum = G.sum(axis=1, keepdims=True)
    G_norm = G / G_rowsum
    mu_hat = G_norm @ Y.astype(np.float64)
    phi_hat = _mom_phi(Y, mu_hat)
    return mu_hat, phi_hat, None


def ridge_log_counts(data, alpha=1.0, seed=42):
    Y = data.Y
    N, P = Y.shape
    Z = data.Z_aug
    log_Y = np.log(Y + 1.0)
    beta_hat = np.zeros((Z.shape[1], P))
    for j in range(P):
        model = Ridge(alpha=alpha, fit_intercept=False, random_state=seed)
        model.fit(Z, log_Y[:, j])
        beta_hat[:, j] = model.coef_
    log_mu_hat = Z @ beta_hat
    mu_hat = np.exp(log_mu_hat) - 1.0
    mu_hat = np.maximum(mu_hat, 1e-6)
    phi_hat = _mom_phi(Y, mu_hat)
    return mu_hat, phi_hat, beta_hat


def lasso_log_counts(data, alpha=0.1, seed=42):
    Y = data.Y
    N, P = Y.shape
    Z = data.Z_aug
    log_Y = np.log(Y + 1.0)
    beta_hat = np.zeros((Z.shape[1], P))
    for j in range(P):
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=5000,
                      random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(Z, log_Y[:, j])
        beta_hat[:, j] = model.coef_
    log_mu_hat = Z @ beta_hat
    mu_hat = np.exp(log_mu_hat) - 1.0
    mu_hat = np.maximum(mu_hat, 1e-6)
    phi_hat = _mom_phi(Y, mu_hat)
    gamma_hat = (np.abs(beta_hat[1:, :]) > 1e-8).astype(float)
    return mu_hat, phi_hat, beta_hat, gamma_hat


def _add_selection_metrics(res, beta_hat, truth):
    """Add precision/recall/F1/AUC if truth is available."""
    if truth is None or "gamma" not in truth or beta_hat is None:
        return
    gamma_hat = (np.abs(beta_hat[1:, :]) > 0.01).astype(float)
    gamma_true = truth["gamma"]
    L_min = min(gamma_hat.shape[0], gamma_true.shape[0])
    tp = ((gamma_hat[:L_min] == 1) & (gamma_true[:L_min] == 1)).sum()
    fp = ((gamma_hat[:L_min] == 1) & (gamma_true[:L_min] == 0)).sum()
    fn_count = ((gamma_hat[:L_min] == 0) & (gamma_true[:L_min] == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn_count, 1)
    res["precision"] = precision
    res["recall"] = recall
    res["f1"] = 2 * precision * recall / max(precision + recall, 1e-8)


def run_all_baselines(data, truth=None, verbose=True, seed=42):
    Y = data.Y
    results = []

    methods = [
        ("KNN Spatial Avg", lambda: knn_average(data)),
        ("Ridge log(y+1)", lambda: ridge_log_counts(data, alpha=1.0, seed=seed)),
    ]

    for name, fn in methods:
        t0 = time.perf_counter()
        out = fn()
        elapsed = time.perf_counter() - t0
        mu_hat, phi_hat = out[0], out[1]
        beta_hat = out[2] if len(out) > 2 else None
        res = evaluate(Y, mu_hat, phi_hat, name)
        res["time_s"] = elapsed
        _add_selection_metrics(res, beta_hat, truth)
        results.append(res)
        if verbose:
            print(f"  {name:>20s}:  ll/obs={res['nb_loglik_per_obs']:+.4f}  "
                  f"mse={res['mse']:.2f}±{res['mse_std']:.2f}  "
                  f"mae={res['mae']:.2f}±{res['mae_std']:.2f}  "
                  f"({elapsed:.2f}s)")

    # lasso
    t0 = time.perf_counter()
    mu_hat, phi_hat, beta_hat, gamma_hat = lasso_log_counts(data, alpha=0.05,
                                                             seed=seed)
    elapsed = time.perf_counter() - t0
    res = evaluate(Y, mu_hat, phi_hat, "Lasso log(y+1)")
    res["time_s"] = elapsed
    if truth is not None and "gamma" in truth:
        gamma_true = truth["gamma"]
        L_min = min(gamma_hat.shape[0], gamma_true.shape[0])
        tp = ((gamma_hat[:L_min] == 1) & (gamma_true[:L_min] == 1)).sum()
        fp = ((gamma_hat[:L_min] == 1) & (gamma_true[:L_min] == 0)).sum()
        fn_count = ((gamma_hat[:L_min] == 0) & (gamma_true[:L_min] == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn_count, 1)
        res["precision"] = precision
        res["recall"] = recall
        res["f1"] = 2 * precision * recall / max(precision + recall, 1e-8)
        from sklearn.metrics import roc_auc_score
        try:
            scores = np.abs(beta_hat[1:L_min+1, :]).flatten()
            res["pip_auc"] = roc_auc_score(gamma_true[:L_min].flatten(), scores)
        except ValueError:
            res["pip_auc"] = float("nan")
    results.append(res)
    if verbose:
        print(f"  {'Lasso log(y+1)':>20s}:  ll/obs={res['nb_loglik_per_obs']:+.4f}  "
              f"mse={res['mse']:.2f}±{res['mse_std']:.2f}  "
              f"mae={res['mae']:.2f}±{res['mae_std']:.2f}  "
              f"({elapsed:.2f}s)")
    return results


def add_mcmc_result(results, sampler, data, truth=None, label="Bayesian Hierarch"):
    Y = data.Y
    N, P = Y.shape
    beta_mean = sampler.trace_beta.mean(axis=0)
    gamma_mean = sampler.trace_gamma.mean(axis=0)
    active = beta_mean * gamma_mean
    log_theta = data.Z_aug @ active
    log_theta = np.clip(log_theta, -30, 30)
    theta = np.exp(log_theta)
    G = data.G.toarray()
    G_rowsum = G.sum(axis=1, keepdims=True)
    lam = (G @ theta) / G_rowsum
    mu_hat = data.s[:, None] * lam
    phi_hat = sampler.posterior_mean_phi()

    res = evaluate(Y, mu_hat, phi_hat, label)
    if truth is not None and "gamma" in truth:
        pip = sampler.pip()
        gamma_true = truth["gamma"]
        L_min = min(pip.shape[0], gamma_true.shape[0])
        from sklearn.metrics import roc_auc_score
        try:
            res["pip_auc"] = roc_auc_score(
                gamma_true[:L_min].flatten(), pip[:L_min].flatten())
        except ValueError:
            res["pip_auc"] = float("nan")
        gamma_hat = (pip[:L_min] > 0.5).astype(float)
        tp = ((gamma_hat == 1) & (gamma_true[:L_min] == 1)).sum()
        fp = ((gamma_hat == 1) & (gamma_true[:L_min] == 0)).sum()
        fn_count = ((gamma_hat == 0) & (gamma_true[:L_min] == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn_count, 1)
        res["precision"] = precision
        res["recall"] = recall
        res["f1"] = 2 * precision * recall / max(precision + recall, 1e-8)
    results.append(res)
    return results


def print_comparison_table(results):
    print(f"{'Method':<22s} {'LL/obs':>10s} {'logMSE/spot':>14s} {'logMAE/spot':>14s} "
          f"{'F1':>7s} {'AUC':>7s}")
    print("-" * 82)
    for r in results:
        f1 = f"{r.get('f1', float('nan')):.3f}" if 'f1' in r else "  —"
        auc = f"{r.get('pip_auc', float('nan')):.3f}" if 'pip_auc' in r else "  —"
        mse_str = f"{r['mse']:.2f}±{r['mse_std']:.2f}"
        mae_str = f"{r['mae']:.2f}±{r['mae_std']:.2f}"
        print(f"{r['method']:<22s} {r['nb_loglik_per_obs']:>+10.4f} "
              f"{mse_str:>14s} {mae_str:>14s} "
              f"{f1:>7s} {auc:>7s}")