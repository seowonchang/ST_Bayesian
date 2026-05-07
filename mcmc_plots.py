"""
MCMC diagnostic and posterior summary plots
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path


class MCMCPlotter:
    def __init__(self, sampler, truth=None, figsize_scale=1.0):
        self.s = sampler
        self.truth = truth
        self.sc = figsize_scale
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    # log posterior trace
    def plot_trace_lp(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8*self.sc, 3*self.sc))
        iters = np.arange(len(self.s.trace_lp))
        n_obs = self.s.N * self.s.P
        ax.plot(iters, self.s.trace_lp / n_obs, linewidth=0.5, color="#2563eb")
        ax.set_xlabel("Post-burnin sample")
        ax.set_ylabel("Log-posterior / obs")
        ax.set_title("Log-posterior trace")
        return ax

    # individ param trace
    def plot_traces(self, gene_ids=None, n_genes=5):
        if gene_ids is None:
            gene_ids = np.linspace(0, self.s.P - 1, n_genes, dtype=int)
        n = len(gene_ids)
        fig, axes = plt.subplots(n, 3, figsize=(14*self.sc, 2.5*n*self.sc), constrained_layout=True)
        if n == 1:
            axes = axes[np.newaxis, :]
        for row, j in enumerate(gene_ids):
            # beta0
            axes[row, 0].plot(self.s.trace_beta[:, 0, j], lw=0.4, color="#2563eb")
            if self.truth:
                axes[row, 0].axhline(self.truth["beta0"][j], color="#dc2626", ls="--", lw=1, label="true")
            axes[row, 0].set_ylabel(f"Gene {j}")
            if row == 0:
                axes[row, 0].set_title("β₀ (row 0 of β)")
            # phi
            axes[row, 1].plot(self.s.trace_phi[:, j], lw=0.4, color="#16a34a")
            if self.truth:
                axes[row, 1].axhline(self.truth["phi"][j], color="#dc2626", ls="--", lw=1)
            if row == 0:
                axes[row, 1].set_title("φ")
            # gamma mean (rows 1: only — skip intercept)
            gamma_j = self.s.trace_gamma[:, 1:, j].mean(axis=1)
            axes[row, 2].plot(gamma_j, lw=0.4, color="#9333ea")
            if self.truth:
                axes[row, 2].axhline(self.truth["gamma"][:, j].mean(), color="#dc2626", ls="--", lw=1)
            if row == 0:
                axes[row, 2].set_title("γ̄ (mean inclusion, L1:L)")

        fig.suptitle("Parameter traces (selected genes)", fontsize=13, y=1.01)
        return fig

    # post vs truth
    def plot_recovery(self):
        if self.truth is None:
            raise ValueError("Need truth dict for recovery plots")

        fig, axes = plt.subplots(1, 3, figsize=(14*self.sc, 4.5*self.sc), constrained_layout=True)
        from scipy.stats import pearsonr
        # intercept
        est = self.s.posterior_mean_beta0()
        tru = self.truth["beta0"]
        ax = axes[0]
        ax.scatter(tru, est, s=12, alpha=0.6, c="#2563eb", edgecolors="none")
        lims = [min(tru.min(), est.min()) - 0.3, max(tru.max(), est.max()) + 0.3]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
        r, _ = pearsonr(tru, est)
        ax.set_xlabel("True β₀"); ax.set_ylabel("Estimated β₀")
        ax.set_title(f"β₀ recovery  (r = {r:.3f})")
        # phi
        est = self.s.posterior_mean_phi()
        tru = self.truth["phi"]
        ax = axes[1]
        ax.scatter(tru, est, s=12, alpha=0.6, c="#16a34a", edgecolors="none")
        lims = [0, max(tru.max(), est.max()) * 1.1]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
        r, _ = pearsonr(tru, est)
        ax.set_xlabel("True φ"); ax.set_ylabel("Estimated φ")
        ax.set_title(f"φ recovery  (r = {r:.3f})")
        # pip 
        pip = self.s.pip().flatten()
        gam = self.truth["gamma"].flatten()
        ax = axes[2]
        jitter = np.random.default_rng(0).normal(0, 0.02, size=gam.shape)
        colors = np.where(gam == 1, "#dc2626", "#94a3b8")
        ax.scatter(gam + jitter, pip, s=6, alpha=0.4, c=colors, edgecolors="none")
        ax.set_xlabel("True γ (jittered)"); ax.set_ylabel("PIP")
        ax.set_title(f"PIP vs truth  (AUC = {self._pip_auc():.3f})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Inactive", "Active"])
        fig.suptitle("Parameter recovery", fontsize=13, y=1.02)
        return fig

    # pip heatmap
    def plot_pip_heatmap(self, gene_ids=None, ax=None):
        pip = self.s.pip()
        L, P = pip.shape

        if gene_ids is not None:
            pip = pip[:, gene_ids]
            labels = [str(g) for g in gene_ids]
        else:
            labels = [str(j) for j in range(P)]

        if ax is None:
            w = max(8, len(labels) * 0.18)
            fig, ax = plt.subplots(figsize=(w*self.sc, (1.5 + L*0.35)*self.sc), constrained_layout=True)

        im = ax.imshow(pip, aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.5, vmin=0, vmax=1),
                       interpolation="nearest")
        ax.set_yticks(range(L))
        ax.set_yticklabels([f"L{l+1}" for l in range(L)])
        ax.set_xlabel("Gene")
        ax.set_title("Posterior Inclusion Probability (PIP)")
        if len(labels) > 30:
            tick_pos = list(range(0, len(labels), 5))
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([labels[i] for i in tick_pos], fontsize=7)
        else:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7, rotation=90)
        plt.colorbar(im, ax=ax, shrink=0.8, label="PIP")

        if self.truth is not None:
            gamma = self.truth["gamma"]
            if gene_ids is not None:
                gamma = gamma[:, gene_ids]
            for l in range(min(L, gamma.shape[0])):
                for j in range(gamma.shape[1]):
                    if gamma[l, j] == 1:
                        rect = plt.Rectangle((j-0.5, l-0.5), 1, 1,
                                             linewidth=1.2, edgecolor="#dc2626",
                                             facecolor="none")
                        ax.add_patch(rect)
        return ax

    # phi
    def plot_phi_diagnostic(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10*self.sc, 4*self.sc), constrained_layout=True)
        phi_est = self.s.posterior_mean_phi()
        P = len(phi_est)
        colors = np.where(phi_est < 0.01, "#dc2626", "#2563eb")
        ax.bar(range(P), np.log10(phi_est + 1e-10), color=colors, width=0.8)
        ax.axhline(np.log10(0.01), color="#dc2626", ls="--", lw=0.8, label="φ < 0.01 (collapsed)")
        ax.set_xlabel("Gene"); ax.set_ylabel("log₁₀(φ)")
        ax.set_title(f"Dispersion estimates — {(phi_est < 0.01).sum()} collapsed genes (red)")
        ax.legend(fontsize=8)
        return ax

    # acceptance rates
    def plot_acceptance(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6*self.sc, 3.5*self.sc), constrained_layout=True)
        keys = list(self.s.accept.keys())
        rates = []
        for k in keys:
            tot = self.s.total[k]
            rates.append(self.s.accept[k] / tot if tot > 0 else 0)

        bars = ax.barh(keys, rates, color="#2563eb", height=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{rate:.1%}", va="center", fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_xlabel("Acceptance rate")
        ax.set_title("MH acceptance rates")
        ax.axvspan(0.20, 0.50, alpha=0.08, color="green", label="Ideal (20–50%)")
        ax.legend(fontsize=8, loc="lower right")
        return ax

    # post histograms
    def plot_posteriors(self, gene_ids=None, n_genes=4):
        if gene_ids is None:
            gene_ids = np.linspace(0, self.s.P - 1, n_genes, dtype=int)
        n = len(gene_ids)
        fig, axes = plt.subplots(n, 3, figsize=(12*self.sc, 2.5*n*self.sc), constrained_layout=True)
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, j in enumerate(gene_ids):
            ax = axes[row, 0]
            ax.hist(self.s.trace_beta[:, 0, j], bins=40, density=True, color="#2563eb", alpha=0.7, edgecolor="none")
            if self.truth:
                ax.axvline(self.truth["beta0"][j], color="#dc2626", ls="--", lw=1.2)
            ax.set_ylabel(f"Gene {j}")
            if row == 0:
                ax.set_title("β₀ posterior")

            # phi posterior
            ax = axes[row, 1]
            ax.hist(self.s.trace_phi[:, j], bins=40, density=True, color="#16a34a", alpha=0.7, edgecolor="none")
            if self.truth:
                ax.axvline(self.truth["phi"][j], color="#dc2626", ls="--", lw=1.2)
            if row == 0:
                ax.set_title("φ posterior")

            # PIP bar for this gene
            ax = axes[row, 2]
            pip_j = self.s.pip()[:, j]  # (L,)
            colors_pip = ["#dc2626" if p > 0.5 else "#94a3b8" for p in pip_j]
            ax.bar(range(len(pip_j)), pip_j, color=colors_pip)
            ax.axhline(0.5, color="k", ls=":", lw=0.6)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Component")
            if self.truth:
                active = np.where(self.truth["gamma"][:, j] == 1)[0]
                for a in active:
                    if a < len(pip_j):
                        ax.annotate("★", (a, pip_j[a] + 0.03), ha="center", fontsize=8, color="#dc2626")
            if row == 0:
                ax.set_title("PIP by component")

        fig.suptitle("Posterior distributions (selected genes)", fontsize=13, y=1.01)
        return fig

    # component activity
    def plot_component_activity(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8*self.sc, 3.5*self.sc), constrained_layout=True)
        pip = self.s.pip()
        L, P = pip.shape

        n_selected = (pip > 0.5).sum(axis=1)
        mean_pip = pip.mean(axis=1)

        x = np.arange(L)
        ax.bar(x - 0.15, n_selected, width=0.3, color="#2563eb", label=f"Genes with PIP > 0.5 (of {P})")
        ax2 = ax.twinx()
        ax2.bar(x + 0.15, mean_pip, width=0.3, color="#f59e0b", alpha=0.7, label="Mean PIP")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Mean PIP", color="#f59e0b")

        ax.set_xlabel("PCA component")
        ax.set_ylabel("# genes selected", color="#2563eb")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l+1}" for l in range(L)])
        ax.set_title("Component activity across genes")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        return ax

    # autocorr
    @staticmethod
    def _acf(x, max_lag):
        n = len(x)
        max_lag = min(max_lag, n - 1)
        x = x - x.mean()
        var = np.dot(x, x)
        if var < 1e-15:
            return np.zeros(max_lag + 1)
        acf = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf[k] = np.dot(x[:n-k], x[k:]) / var
        return acf

    # ess
    @staticmethod
    def _ess(x):
        n = len(x)
        max_lag = min(n - 1, 1000)
    
        x_c = x - x.mean()
        var = np.dot(x_c, x_c) / (n - 1)
        if var <= 0:
            return float(n)
    
        acf = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf[k] = np.dot(x_c[:n-k], x_c[k:]) / ((n - k) * var)
    
        tau = 1.0
        for k in range(1, max_lag, 2):
            pair = acf[k] + acf[k + 1]
            if pair < 0:
                break
            tau += 2.0 * pair
    
        return max(1.0, n / tau)

    def compute_ess(self):
        P = self.s.P
        return {
            "beta0": np.array([self._ess(self.s.trace_beta[:, 0, j]) for j in range(P)]),
            "phi": np.array([self._ess(self.s.trace_phi[:, j]) for j in range(P)]),
            "gamma_bar": np.array([
                self._ess(self.s.trace_gamma[:, 1:, j].mean(axis=1)) for j in range(P)
            ]),
        }

    def plot_autocorrelation(self, gene_ids=None, n_genes=4, max_lag=100):
        if gene_ids is None:
            gene_ids = np.linspace(0, self.s.P - 1, n_genes, dtype=int)
        n = len(gene_ids)
        fig, axes = plt.subplots(n, 3, figsize=(14*self.sc, 2.5*n*self.sc), constrained_layout=True)
        if n == 1:
            axes = axes[np.newaxis, :]

        params = [
            ("β₀", lambda j: self.s.trace_beta[:, 0, j], "#2563eb"),
            ("φ", lambda j: self.s.trace_phi[:, j], "#16a34a"),
            ("γ̄", lambda j: self.s.trace_gamma[:, 1:, j].mean(axis=1), "#9333ea"),
        ]

        for row, j in enumerate(gene_ids):
            for col, (title, get_trace, color) in enumerate(params):
                ax = axes[row, col]
                trace = get_trace(j)
                lags = np.arange(max_lag + 1)
                acf = self._acf(trace, max_lag)
                ess = self._ess(trace)

                ax.bar(lags, acf, width=1.0, color=color, alpha=0.7, edgecolor="none")
                ax.axhline(0, color="k", lw=0.5)
                ci = 1.96 / np.sqrt(len(trace))
                ax.axhline(ci, color="#dc2626", ls="--", lw=0.7)
                ax.axhline(-ci, color="#dc2626", ls="--", lw=0.7)
                ax.set_ylim(-0.2, 1.05)
                ax.text(0.95, 0.9, f"ESS={ess:.0f}", transform=ax.transAxes,
                        ha="right", fontsize=8, color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
                if row == 0:
                    ax.set_title(f"{title} autocorrelation")
                if row == n - 1:
                    ax.set_xlabel("Lag")
                if col == 0:
                    ax.set_ylabel(f"Gene {j}")

        fig.suptitle("Autocorrelation (selected genes)", fontsize=13, y=1.01)
        return fig

    def plot_ess_summary(self, ax=None):
        """ESS across all genes for each parameter type."""
        ess = self.compute_ess()
        P = self.s.P
        n_samples = len(self.s.trace_lp)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10*self.sc, 4*self.sc), constrained_layout=True)

        x = np.arange(P)
        w = 0.25
        ax.bar(x - w, ess["beta0"], width=w, color="#2563eb", alpha=0.7, label="β₀")
        ax.bar(x, ess["phi"], width=w, color="#16a34a", alpha=0.7, label="φ")
        ax.bar(x + w, ess["gamma_bar"], width=w, color="#9333ea", alpha=0.7, label="γ̄")
        ax.axhline(n_samples, color="k", ls=":", lw=0.6, label=f"n_samples={n_samples}")
        ax.set_xlabel("Gene")
        ax.set_ylabel("Effective sample size")
        ax.set_title(f"ESS by gene — median: β₀={np.median(ess['beta0']):.0f}, "
                     f"φ={np.median(ess['phi']):.0f}, "
                     f"γ̄={np.median(ess['gamma_bar']):.0f}")
        ax.legend(fontsize=8, ncol=4)
        return ax

    # plot all figures
    def plot_dashboard(self, gene_ids=None):
        fig = plt.figure(figsize=(18*self.sc, 16*self.sc), constrained_layout=True)
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.3])

        self.plot_trace_lp(fig.add_subplot(gs[0, 0]))
        self.plot_acceptance(fig.add_subplot(gs[0, 1]))
        self.plot_component_activity(fig.add_subplot(gs[0, 2]))

        if self.truth:
            from scipy.stats import pearsonr

            ax_r = fig.add_subplot(gs[1, 0])
            est = self.s.posterior_mean_beta0()
            tru = self.truth["beta0"]
            ax_r.scatter(tru, est, s=10, alpha=0.6, c="#2563eb", edgecolors="none")
            lims = [min(tru.min(), est.min())-0.3, max(tru.max(), est.max())+0.3]
            ax_r.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
            r, _ = pearsonr(tru, est)
            ax_r.set_xlabel("True β₀"); ax_r.set_ylabel("Est β₀")
            ax_r.set_title(f"β₀ (r={r:.3f})")

            ax_r2 = fig.add_subplot(gs[1, 1])
            est = self.s.posterior_mean_phi()
            tru = self.truth["phi"]
            ax_r2.scatter(tru, est, s=10, alpha=0.6, c="#16a34a", edgecolors="none")
            r, _ = pearsonr(tru, est)
            lims = [0, max(tru.max(), est.max())*1.1]
            ax_r2.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
            ax_r2.set_xlabel("True φ"); ax_r2.set_ylabel("Est φ")
            ax_r2.set_title(f"φ (r={r:.3f})")

        self.plot_phi_diagnostic(fig.add_subplot(gs[1, 2]))
        self.plot_pip_heatmap(gene_ids=gene_ids, ax=fig.add_subplot(gs[2, :]))

        fig.suptitle("MCMC Diagnostic Dashboard", fontsize=15, y=1.01)
        return fig

    #save
    def save_all(self, output_dir="plots/", fmt="png", dpi=150):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # all plots
        fig = self.plot_dashboard()
        fig.savefig(out / f"dashboard.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        # traceplots
        fig = self.plot_traces()
        fig.savefig(out / f"traces.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        # posterior plots
        fig = self.plot_posteriors()
        fig.savefig(out / f"posteriors.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        # autocorrelation 
        fig = self.plot_autocorrelation()
        fig.savefig(out / f"autocorrelation.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        # ess
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        self.plot_ess_summary(ax)
        fig.savefig(out / f"ess.{fmt}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        # recovery plots for simulated data
        if self.truth:
            fig = self.plot_recovery()
            fig.savefig(out / f"recovery.{fmt}", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
        print(f"Saved plots to {out.resolve()}/")

    # helpers
    def _pip_auc(self):
        if self.truth is None:
            return float("nan")
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(self.truth["gamma"].flatten(),
                             self.s.pip().flatten())

