import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy.special import gammaln

class PosteriorAnalysis:
    def __init__(self, sampler, data, truth=None, gene_names=None, figsize_scale=1.0):
        self.s = sampler
        self.data = data
        self.truth = truth
        self.sc = figsize_scale
        self.N, self.P = data.Y.shape
        self.L = data.L

        if gene_names is not None:
            self.gene_names = gene_names
        else:
            self.gene_names = [f"Gene {j}" for j in range(self.P)]

        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    def plot_spatial_prediction(self, gene_ids=None, n_genes=3):
        T = self.data.T
        Y = self.data.Y
 
        if gene_ids is None:
            gene_ids = np.linspace(0, self.P - 1, n_genes, dtype=int)
        n = len(gene_ids)
 
        # Reconstruct mu from each posterior sample
        n_samples = self.s.trace_beta.shape[0]
        G = self.data.G.toarray()
        G_rowsum = G.sum(axis=1, keepdims=True)
        s = self.data.s
        n_sub = min(n_samples, 200)
        idx = np.linspace(0, n_samples-1, n_sub, dtype=int)
        mu_samples = np.zeros((n_sub, self.N, len(gene_ids)))
 
        for t_idx, t in enumerate(idx):
            active = self.s.trace_beta[t] * self.s.trace_gamma[t]
            log_theta = self.data.Z_aug @ active
            np.clip(log_theta, -30, 30, out=log_theta)
            theta = np.exp(log_theta)
            lam = (G @ theta) / G_rowsum
            mu = s[:, None] * lam
            for k, j in enumerate(gene_ids):
                mu_samples[t_idx, :, k] = mu[:, j]
 
        mu_mean = mu_samples.mean(axis=0)
        mu_lo = np.quantile(mu_samples, 0.025, axis=0)
        mu_hi = np.quantile(mu_samples, 0.975, axis=0)
        ci_width = mu_hi - mu_lo
 
        fig, axes = plt.subplots(n, 3, figsize=(14*self.sc, 3.5*n*self.sc),
                                 constrained_layout=True)
        if n == 1:
            axes = axes[np.newaxis, :]
 
        for row, (j, k) in enumerate(zip(gene_ids, range(len(gene_ids)))):
            # Observed
            ax = axes[row, 0]
            sc = ax.scatter(T[:, 0], T[:, 1], c=np.log1p(Y[:, j]),
                            s=8, cmap="viridis", edgecolors="none")
            plt.colorbar(sc, ax=ax, shrink=0.7)
            if row == 0:
                ax.set_title("Observed log(y+1)")
            ax.set_ylabel(self.gene_names[j])
            ax.set_aspect("equal")
 
            # Predicted
            ax = axes[row, 1]
            sc = ax.scatter(T[:, 0], T[:, 1], c=np.log1p(mu_mean[:, k]),
                            s=8, cmap="viridis", edgecolors="none")
            plt.colorbar(sc, ax=ax, shrink=0.7)
            if row == 0:
                ax.set_title("Posterior mean log(μ+1)")
            ax.set_aspect("equal")
 
            # CI width on log scale
            ax = axes[row, 2]
            log_ci = np.log1p(mu_hi[:, k]) - np.log1p(mu_lo[:, k])
            sc = ax.scatter(T[:, 0], T[:, 1], c=log_ci,
                            s=8, cmap="Reds", edgecolors="none")
            plt.colorbar(sc, ax=ax, shrink=0.7)
            if row == 0:
                ax.set_title("95% CI width (log scale)")
            ax.set_aspect("equal")
 
        fig.suptitle("Spatial prediction with posterior uncertainty",
                     fontsize=13, y=1.01)
        return fig

    def plot_phi_credible(self, gene_names=None):
        phi_samples = self.s.trace_phi
        phi_mean = phi_samples.mean(axis=0)
        phi_lo = np.quantile(phi_samples, 0.025, axis=0)
        phi_hi = np.quantile(phi_samples, 0.975, axis=0)
 
        order = np.argsort(phi_mean)
        P = len(phi_mean)
 
        if gene_names is None:
            gene_names = self.gene_names
 
        fig, ax = plt.subplots(figsize=(8*self.sc, max(4, P*0.2)*self.sc),
                               constrained_layout=True)
 
        y_pos = np.arange(P)
        ax.barh(y_pos, phi_mean[order], height=0.6, color="#2563eb",
                alpha=0.6, edgecolor="none")
        ax.errorbar(phi_mean[order], y_pos,
                    xerr=[phi_mean[order] - phi_lo[order],
                          phi_hi[order] - phi_mean[order]],
                    fmt="none", ecolor="#64748b", capsize=2, lw=0.8)
 
        if self.truth is not None:
            ax.scatter(self.truth["phi"][order], y_pos, marker="|", s=80,
                       color="#dc2626", zorder=5, label="True φ")
            ax.legend(fontsize=8)
 
        ax.set_yticks(y_pos)
        ax.set_yticklabels([gene_names[i] for i in order], fontsize=7)
        ax.set_xlabel("Dispersion (phi)")
        ax.set_title("Posterior credible intervals for (phi)")
        ax.axvline(1.0, color="#94a3b8", ls=":", lw=0.8, label="phi=1")
 
        return fig

    def save_all(self, output_dir="downstream/", fmt="png", dpi=150):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, fn in [
            ("spatial_prediction", self.plot_spatial_prediction),
            ("phi_credible", self.plot_phi_credible),
        ]:
            fig = fn()
            fig.savefig(out / f"{name}.{fmt}", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        print("Downstream plots finished")