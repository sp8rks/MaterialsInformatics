from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent.parent


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def notebook_metadata() -> dict:
    return {
        "kernelspec": {
            "display_name": "Python 3.11 (uv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }


def build_bayesian_optimization_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(metadata=notebook_metadata())
    nb.cells = [
        md(
            """
            # Bayesian Optimization Deep Dive for Materials Informatics

            This notebook is designed for a class sequence where we have already covered naive Bayes and Gaussian processes.

            **Goals**
            - Connect the Gaussian process posterior from the previous class to the Bayesian optimization loop.
            - Build intuition for surrogate models, acquisition functions, and the exploration vs. exploitation tradeoff.
            - Work through materials-focused examples, including a real silver nanoparticle synthesis dataset.
            - See how single-objective, multi-objective, constrained, and batch settings show up in materials science.
            """
        ),
        code(
            """
            from pathlib import Path

            import ipywidgets as widgets
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            from IPython.display import Markdown, display
            from scipy.stats import norm
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
            from sklearn.preprocessing import MinMaxScaler

            sns.set_theme(style="whitegrid", context="talk")
            plt.rcParams["figure.figsize"] = (10, 6)
            plt.rcParams["axes.spines.top"] = False
            plt.rcParams["axes.spines.right"] = False

            DATA_PATH = Path("AgNP_dataset.csv")
            # Fixed RNG keeps the lecture visuals reproducible from run to run.
            rng = np.random.default_rng(7)
            """
        ),
        md(
            r"""
            ## 1. From Gaussian Processes to Bayesian Optimization

            A Gaussian process gives us a predictive **mean** and **uncertainty** over a function we have not fully observed yet.
            Bayesian optimization adds a decision rule on top of that posterior:

            $$
            \text{next experiment} = \arg\max_x \ \alpha(x \mid \mathcal{D})
            $$

            where $\alpha$ is an **acquisition function** such as:
            - Probability of Improvement (PI)
            - Upper Confidence Bound (UCB)
            - Expected Improvement (EI)

            In materials science, this matters because each experiment or simulation can be expensive:
            wet-lab synthesis, DFT calculations, high-temperature processing runs, or long cycling experiments.
            """
        ),
        md(
            """
            ### Symbol and Variable Guide

            Before jumping into code, here is the meaning of the main symbols and variables that appear throughout the notebook.
            """
        ),
        md(
            r"""
            | Symbol / variable | Meaning |
            | --- | --- |
            | $x$ | A candidate material, recipe, or processing condition. |
            | $y$ | The measured property we care about, such as yield, loss, stability, or strength. |
            | $\mu(x)$ | The GP posterior mean: our current best prediction at $x$. |
            | $\sigma(x)$ | The GP posterior standard deviation: our uncertainty at $x$. |
            | $\alpha(x)$ | The acquisition score that tells us how attractive $x$ is as the next experiment. |
            | `xi` | Improvement bonus in PI/EI. Larger values encourage more exploration. |
            | `kappa` | Exploration weight in UCB. Larger values place more weight on uncertainty. |
            | `X_obs`, `y_obs` | Experiments we have already performed and their measured outcomes. |
            | `X_grid` | Candidate points where we evaluate the surrogate and acquisition function. |
            """
        ),
        code(
            """
            def synthetic_objective(x):
                # Synthetic 1D landscape with several local peaks.
                x = np.asarray(x)
                return (
                    np.sin(2.7 * x)
                    + 0.35 * np.cos(8.0 * x)
                    - 0.08 * (x - 2.8) ** 2
                )


            def build_gp():
                # A smooth-but-flexible kernel that matches the type of GP
                # behavior we usually want for Bayesian optimization demos.
                kernel = (
                    ConstantKernel(1.0, constant_value_bounds="fixed")
                    * Matern(length_scale=0.55, length_scale_bounds="fixed", nu=2.5)
                    + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
                )
                return GaussianProcessRegressor(
                    kernel=kernel,
                    optimizer=None,
                    normalize_y=True,
                )


            def probability_of_improvement(mu, sigma, best, xi=0.01):
                # PI asks: what is the probability of beating the current best?
                sigma = np.maximum(sigma, 1e-9)
                z = (mu - best - xi) / sigma
                return norm.cdf(z)


            def upper_confidence_bound(mu, sigma, kappa=1.5):
                # UCB directly trades off predicted value and uncertainty.
                return mu + kappa * sigma


            def expected_improvement(mu, sigma, best, xi=0.01):
                # EI rewards points that are both promising and uncertain.
                sigma = np.maximum(sigma, 1e-9)
                improvement = mu - best - xi
                z = improvement / sigma
                return improvement * norm.cdf(z) + sigma * norm.pdf(z)


            # Dense grid of candidate points used for plotting and BO selection.
            X_grid = np.linspace(0.0, 4.0, 400).reshape(-1, 1)
            y_grid = synthetic_objective(X_grid).ravel()

            # A small initial design, standing in for our first few experiments.
            X_obs = np.array([0.15, 0.75, 1.35, 2.1, 3.35]).reshape(-1, 1)
            y_obs = synthetic_objective(X_obs).ravel()

            gp0 = build_gp()
            gp0.fit(X_obs, y_obs)
            mu0, sigma0 = gp0.predict(X_grid, return_std=True)

            best_so_far = y_obs.max()
            pi0 = probability_of_improvement(mu0, sigma0, best_so_far)
            ucb0 = upper_confidence_bound(mu0, sigma0)
            ei0 = expected_improvement(mu0, sigma0, best_so_far)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(12, 9),
                sharex=True,
                gridspec_kw={"height_ratios": [2.4, 1.4]},
            )

            axes[0].plot(X_grid, y_grid, color="black", linewidth=2, label="True objective")
            axes[0].plot(X_grid, mu0, color="#1f77b4", linewidth=2, label="GP mean")
            axes[0].fill_between(
                X_grid.ravel(),
                mu0 - 1.96 * sigma0,
                mu0 + 1.96 * sigma0,
                color="#9ecae1",
                alpha=0.45,
                label="95% credible interval",
            )
            axes[0].scatter(X_obs, y_obs, color="#d62728", s=90, zorder=5, label="Observed experiments")
            axes[0].set_ylabel("Objective value")
            axes[0].set_title("GP posterior after a few initial experiments")
            axes[0].legend(loc="best")

            axes[1].plot(X_grid, pi0, label="PI", linewidth=2)
            axes[1].plot(X_grid, ucb0, label="UCB", linewidth=2)
            axes[1].plot(X_grid, ei0, label="EI", linewidth=2)
            axes[1].set_xlabel("Design variable")
            axes[1].set_ylabel("Acquisition score")
            axes[1].set_title("Acquisition functions turn a posterior into an experimental decision")
            axes[1].legend(loc="best")

            plt.tight_layout()
            """
        ),
        md(
            """
            **Interpretation**

            - The GP mean is our current best guess of the response surface.
            - The uncertainty band is widest where we have little or no data.
            - PI/UCB/EI all reward promising regions, but they balance promise and uncertainty differently.
            - EI is often a nice teaching default because it directly measures the expected amount of improvement.
            """
        ),
        md(
            """
            ### Interactive Acquisition Playground

            Use the controls below to ask a classroom question like:
            **"If we reward uncertainty more strongly, where does the algorithm want to sample next?"**
            """
        ),
        code(
            """
            def plot_acquisition_playground(acquisition="EI", xi=0.01, kappa=1.5):
                # Refit the surrogate each time the widget settings change.
                gp = build_gp()
                gp.fit(X_obs, y_obs)
                mu, sigma = gp.predict(X_grid, return_std=True)

                acquisition = acquisition.upper()
                if acquisition == "PI":
                    scores = probability_of_improvement(mu, sigma, y_obs.max(), xi=xi)
                    subtitle = f"PI with xi={xi:.2f}"
                elif acquisition == "UCB":
                    scores = upper_confidence_bound(mu, sigma, kappa=kappa)
                    subtitle = f"UCB with kappa={kappa:.2f}"
                else:
                    scores = expected_improvement(mu, sigma, y_obs.max(), xi=xi)
                    subtitle = f"EI with xi={xi:.2f}"

                # The BO recommendation is the point with the largest acquisition score.
                next_idx = int(np.argmax(scores))
                x_next = float(X_grid[next_idx, 0])
                y_next = float(y_grid[next_idx])

                fig, axes = plt.subplots(
                    2,
                    1,
                    figsize=(12, 8),
                    sharex=True,
                    gridspec_kw={"height_ratios": [2.2, 1.2]},
                )

                axes[0].plot(X_grid, y_grid, color="black", linewidth=2, label="True objective")
                axes[0].plot(X_grid, mu, color="#1f77b4", linewidth=2, label="GP mean")
                axes[0].fill_between(
                    X_grid.ravel(),
                    mu - 1.96 * sigma,
                    mu + 1.96 * sigma,
                    color="#9ecae1",
                    alpha=0.45,
                    label="95% credible interval",
                )
                axes[0].scatter(X_obs, y_obs, color="#d62728", s=90, zorder=5, label="Observed experiments")
                axes[0].axvline(x_next, color="#2ca02c", linestyle="--", linewidth=2, label="Suggested next experiment")
                axes[0].scatter([x_next], [y_next], color="#2ca02c", s=110, zorder=6)
                axes[0].set_ylabel("Objective value")
                axes[0].set_title("Posterior and suggested next experiment")
                axes[0].legend(loc="best")

                axes[1].plot(X_grid, scores, color="#9467bd", linewidth=2.5)
                axes[1].fill_between(X_grid.ravel(), 0, scores, color="#c5b0d5", alpha=0.45)
                axes[1].axvline(x_next, color="#2ca02c", linestyle="--", linewidth=2)
                axes[1].set_xlabel("Design variable")
                axes[1].set_ylabel("Acquisition score")
                axes[1].set_title(subtitle)
                plt.tight_layout()
                plt.show()

                display(
                    Markdown(
                        f"**Next suggested experiment:** x = `{x_next:.3f}` with true objective value `{y_next:.3f}`."
                    )
                )


            widgets.interact(
                plot_acquisition_playground,
                acquisition=widgets.Dropdown(options=["EI", "PI", "UCB"], value="EI", description="Rule"),
                xi=widgets.FloatSlider(value=0.01, min=0.0, max=0.5, step=0.01, description="xi"),
                kappa=widgets.FloatSlider(value=1.5, min=0.1, max=4.0, step=0.1, description="kappa"),
            )
            """
        ),
        code(
            """
            initial_design_pool = np.array([0.15, 0.55, 0.95, 1.35, 2.10, 2.65, 3.35, 3.75]).reshape(-1, 1)


            def run_synthetic_bo(n_steps=12, acquisition="EI", xi=0.01, kappa=1.5, n_initial=5):
                # Start from the chosen number of initial experiments.
                X_current = initial_design_pool[:n_initial].copy()
                y_current = synthetic_objective(X_current).ravel()
                snapshots = []

                for step in range(n_steps):
                    # Fit the surrogate to all experiments completed so far.
                    gp = build_gp()
                    gp.fit(X_current, y_current)
                    mu, sigma = gp.predict(X_grid, return_std=True)
                    acquisition = acquisition.upper()
                    if acquisition == "PI":
                        scores = probability_of_improvement(mu, sigma, y_current.max(), xi=xi)
                    elif acquisition == "UCB":
                        scores = upper_confidence_bound(mu, sigma, kappa=kappa)
                    else:
                        scores = expected_improvement(mu, sigma, y_current.max(), xi=xi)
                    # Pick the next experiment by maximizing the acquisition function.
                    next_index = int(np.argmax(scores))
                    x_next = X_grid[next_index].reshape(1, 1)
                    y_next = synthetic_objective(x_next).reshape(-1)

                    snapshots.append(
                        {
                            "step": step + 1,
                            "x_next": float(x_next[0, 0]),
                            "y_next": float(y_next[0]),
                            "best_before": float(y_current.max()),
                            "best_after": float(max(y_current.max(), y_next[0])),
                        }
                    )

                    X_current = np.vstack([X_current, x_next])
                    y_current = np.concatenate([y_current, y_next])

                return X_current, y_current, pd.DataFrame(snapshots)


            X_final, y_final, synthetic_log = run_synthetic_bo()
            display(synthetic_log.head(8))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            n_initial_default = 5

            axes[0].plot(X_grid, y_grid, color="black", linewidth=2)
            axes[0].scatter(X_final[:n_initial_default], y_final[:n_initial_default], color="#7f7f7f", s=80, label="Initial points")
            axes[0].scatter(X_final[n_initial_default:], y_final[n_initial_default:], color="#2ca02c", s=85, label="BO-selected points")
            axes[0].set_xlabel("Design variable")
            axes[0].set_ylabel("Objective value")
            axes[0].set_title("Expected improvement quickly focuses on high-value regions")
            axes[0].legend(loc="best")

            best_trace = [y_final[: i].max() for i in range(n_initial_default, len(X_final) + 1)]
            axes[1].plot(range(len(best_trace)), best_trace, marker="o", linewidth=2, color="#1f77b4")
            axes[1].axhline(y_grid.max(), linestyle="--", color="black", label="Global optimum")
            axes[1].set_xlabel("Bayesian optimization iteration")
            axes[1].set_ylabel("Best observed objective")
            axes[1].set_title("Best-so-far improves with each chosen experiment")
            axes[1].legend(loc="best")

            plt.tight_layout()
            """
        ),
        md(
            """
            ### Interactive Optimization Loop

            This widget lets students change the acquisition rule, how exploratory it is, the number of initial experiments, and the BO budget.
            It is useful for in-class questions like:

            - How sensitive is the trajectory to the initial design?
            - What changes when we use UCB instead of EI?
            - Does more exploration help early or late in the campaign?
            """
        ),
        code(
            """
            def explore_bo_trajectory(acquisition="EI", xi=0.01, kappa=1.5, n_initial=5, n_steps=10):
                X_run, y_run, run_log = run_synthetic_bo(
                    n_steps=n_steps,
                    acquisition=acquisition,
                    xi=xi,
                    kappa=kappa,
                    n_initial=n_initial,
                )

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].plot(X_grid, y_grid, color="black", linewidth=2)
                axes[0].scatter(X_run[:n_initial], y_run[:n_initial], color="#7f7f7f", s=80, label="Initial design")
                axes[0].scatter(X_run[n_initial:], y_run[n_initial:], color="#2ca02c", s=85, label="BO suggestions")
                axes[0].set_xlabel("Design variable")
                axes[0].set_ylabel("Objective value")
                axes[0].set_title(f"{acquisition} trajectory with {n_initial} initial points")
                axes[0].legend(loc="best")

                best_trace = [y_run[:i].max() for i in range(n_initial, len(X_run) + 1)]
                axes[1].plot(range(len(best_trace)), best_trace, marker="o", linewidth=2.2, color="#1f77b4")
                axes[1].axhline(y_grid.max(), linestyle="--", color="black", label="Global optimum")
                axes[1].set_xlabel("BO iteration")
                axes[1].set_ylabel("Best observed objective")
                axes[1].set_title("Best-so-far trajectory")
                axes[1].legend(loc="best")
                plt.tight_layout()
                plt.show()

                display(run_log.head(min(8, len(run_log))))


            widgets.interact(
                explore_bo_trajectory,
                acquisition=widgets.Dropdown(options=["EI", "PI", "UCB"], value="EI", description="Rule"),
                xi=widgets.FloatSlider(value=0.01, min=0.0, max=0.5, step=0.01, description="xi"),
                kappa=widgets.FloatSlider(value=1.5, min=0.1, max=4.0, step=0.1, description="kappa"),
                n_initial=widgets.IntSlider(value=5, min=3, max=8, step=1, description="initial"),
                n_steps=widgets.IntSlider(value=10, min=4, max=18, step=1, description="steps"),
            )
            """
        ),
        md(
            """
            ## 2. Why Materials Science Is a Natural Home for Bayesian Optimization

            Bayesian optimization is a strong fit whenever each function evaluation is expensive, noisy, and information-rich.
            That describes many materials workflows.
            """
        ),
        code(
            """
            example_table = pd.DataFrame(
                [
                    {
                        "Materials problem": "Silver nanoparticle synthesis",
                        "Design variables": "Flow rates, precursor fractions, seed fraction",
                        "Objective(s)": "Minimize optical loss or size dispersion",
                        "Why BO helps": "Each experiment consumes reagents and instrument time",
                    },
                    {
                        "Materials problem": "Battery cathode formulation",
                        "Design variables": "Composition, calcination temperature, dwell time",
                        "Objective(s)": "Maximize capacity and cycle life",
                        "Why BO helps": "Multi-objective tradeoffs and limited synthesis budget",
                    },
                    {
                        "Materials problem": "Perovskite processing",
                        "Design variables": "Solvent ratio, spin speed, anneal profile",
                        "Objective(s)": "Maximize PCE while maintaining stability",
                        "Why BO helps": "Nonlinear response surfaces and unstable experiments",
                    },
                    {
                        "Materials problem": "Electrocatalyst discovery",
                        "Design variables": "Alloy composition, support, pH, loading",
                        "Objective(s)": "Maximize activity and durability",
                        "Why BO helps": "Expensive characterization and many coupled variables",
                    },
                    {
                        "Materials problem": "Additive manufacturing of alloys",
                        "Design variables": "Laser power, scan speed, hatch spacing",
                        "Objective(s)": "Minimize porosity, maximize strength",
                        "Why BO helps": "Parallel constraints and small high-value datasets",
                    },
                ]
            )
            display(example_table)
            """
        ),
        md(
            """
            ## 3. Real Example: Silver Nanoparticle Optimization

            Here we treat the existing dataset as a catalog of experiments that a student team could choose from sequentially.
            Because there are repeated rows, we first average the repeated measurements so each unique condition becomes a single candidate experiment.
            """
        ),
        code(
            """
            df = pd.read_csv(DATA_PATH)
            feature_cols = ["QAgNO3(%)", "Qpva(%)", "Qtsc(%)", "Qseed(%)", "Qtot(uL/min)"]
            # Average repeated measurements so each unique recipe becomes one candidate.
            agnp = (
                df.groupby(feature_cols, as_index=False)["loss"]
                .mean()
                .sort_values("loss", ascending=True)
                .reset_index(drop=True)
            )

            print(f"Raw rows: {len(df):,}")
            print(f"Unique experimental conditions: {len(agnp):,}")
            print(f"Best averaged loss in the catalog: {agnp['loss'].min():.4f}")

            display(agnp.head(10))
            """
        ),
        md(
            """
            ### What Do These AgNP Variables Mean?

            This is a good place to pause and ask students which variables are likely to be most influential before showing the correlation plot.
            """
        ),
        code(
            """
            agnp_glossary = pd.DataFrame(
                [
                    {"Column": "QAgNO3(%)", "Interpretation": "Silver nitrate fraction. Controls the precursor available for nanoparticle formation."},
                    {"Column": "Qpva(%)", "Interpretation": "Polyvinyl alcohol fraction. Acts like a stabilizer or process modifier."},
                    {"Column": "Qtsc(%)", "Interpretation": "Reducing-agent-related fraction. Influences nucleation and growth kinetics."},
                    {"Column": "Qseed(%)", "Interpretation": "Seed fraction. Can strongly affect particle growth pathways and size distribution."},
                    {"Column": "Qtot(uL/min)", "Interpretation": "Total flow rate. Changes residence time and mixing conditions."},
                    {"Column": "loss", "Interpretation": "Response variable to minimize in this dataset."},
                ]
            )
            display(agnp_glossary)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            sns.histplot(agnp["loss"], bins=24, kde=True, ax=axes[0], color="#1f77b4")
            axes[0].set_title("Distribution of averaged AgNP loss values")

            corr = agnp[feature_cols + ["loss"]].corr(numeric_only=True)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1])
            axes[1].set_title("Feature-response correlations")

            plt.tight_layout()
            """
        ),
        md(
            """
            ### Discrete Bayesian Optimization on a Fixed Experimental Catalog

            In practice, a lab often chooses the next experiment from a **finite list of allowed recipes** rather than from a mathematically continuous search space.
            We can emulate that by:

            1. fitting a GP on the experiments we have already tried,
            2. scoring the **untested** recipes with an acquisition function,
            3. selecting the highest-scoring candidate for the next run.

            Since the dataset uses **loss**, we will **minimize** rather than maximize.
            """
        ),
        code(
            """
            X_catalog = agnp[feature_cols].to_numpy()
            y_catalog = agnp["loss"].to_numpy()

            # Scale the design variables so GP length scales behave more sensibly.
            scaler = MinMaxScaler().fit(X_catalog)
            X_scaled = scaler.transform(X_catalog)

            discrete_kernel = (
                ConstantKernel(1.0, constant_value_bounds="fixed")
                * Matern(length_scale=np.full(X_scaled.shape[1], 0.25), length_scale_bounds="fixed", nu=2.5)
                + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
            )


            def expected_improvement_min(mu, sigma, best, xi=0.01):
                # Minimization version of EI for the AgNP loss objective.
                sigma = np.maximum(sigma, 1e-9)
                improvement = best - mu - xi
                z = improvement / sigma
                return improvement * norm.cdf(z) + sigma * norm.pdf(z)


            def fit_catalog_gp(X_train, y_train):
                # Train the surrogate only on the recipes we have already tested.
                gp = GaussianProcessRegressor(
                    kernel=discrete_kernel,
                    optimizer=None,
                    normalize_y=True,
                )
                gp.fit(X_train, y_train)
                return gp


            def run_catalog_bo(strategy="ei", n_init=6, n_steps=15, seed=0):
                local_rng = np.random.default_rng(seed)
                # Random initial design before the BO policy takes over.
                chosen = list(local_rng.choice(len(X_scaled), size=n_init, replace=False))
                records = []

                for step in range(n_steps):
                    # Only score recipes that have not been tried yet.
                    remaining = np.setdiff1d(np.arange(len(X_scaled)), np.array(chosen), assume_unique=False)

                    if strategy == "random":
                        next_idx = int(local_rng.choice(remaining))
                    else:
                        gp = fit_catalog_gp(X_scaled[chosen], y_catalog[chosen])
                        mu, sigma = gp.predict(X_scaled[remaining], return_std=True)
                        scores = expected_improvement_min(mu, sigma, y_catalog[chosen].min(), xi=0.005)
                        # Select the untested recipe with the largest EI score.
                        next_idx = int(remaining[np.argmax(scores)])

                    chosen.append(next_idx)
                    best_loss = float(y_catalog[chosen].min())
                    records.append(
                        {
                            "step": step + 1,
                            "selected_index": next_idx,
                            "selected_loss": float(y_catalog[next_idx]),
                            "best_loss_so_far": best_loss,
                        }
                    )

                return chosen, pd.DataFrame(records)


            def summarize_strategy(strategy, runs=40, n_init=6, n_steps=15):
                # Repeating the campaign shows average behavior, not just one lucky run.
                traces = []
                final_best = []
                for seed in range(runs):
                    chosen, _ = run_catalog_bo(strategy=strategy, n_init=n_init, n_steps=n_steps, seed=seed)
                    best_curve = [y_catalog[chosen[:i]].min() for i in range(n_init, len(chosen) + 1)]
                    traces.append(best_curve)
                    final_best.append(best_curve[-1])
                return np.array(traces), np.array(final_best)
            """
        ),
        code(
            """
            chosen_ei, ei_log = run_catalog_bo(strategy="ei", seed=11)
            display(ei_log.head(8))

            chosen_rows = agnp.iloc[chosen_ei].copy()
            chosen_rows["selection_order"] = range(len(chosen_rows))
            display(chosen_rows.nsmallest(8, "loss"))
            """
        ),
        md(
            """
            ### Interactive AgNP Campaign Explorer

            The widget below lets you explore a realistic question for the lab:
            **If we have only a small number of experimental rounds, how much does Bayesian optimization help compared with random search?**
            """
        ),
        code(
            """
            def explore_catalog_campaign(strategy="ei", n_init=6, n_steps=12, seed=11):
                chosen, log_df = run_catalog_bo(strategy=strategy, n_init=n_init, n_steps=n_steps, seed=seed)
                chosen_table = agnp.iloc[chosen].copy()
                chosen_table["selection_order"] = range(len(chosen_table))

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                sns.scatterplot(
                    data=agnp,
                    x="QAgNO3(%)",
                    y="loss",
                    color="#c7c7c7",
                    s=70,
                    ax=axes[0],
                    label="Catalog conditions",
                )
                sns.scatterplot(
                    data=chosen_table,
                    x="QAgNO3(%)",
                    y="loss",
                    hue="selection_order",
                    palette="viridis",
                    s=110,
                    ax=axes[0],
                    legend=False,
                )
                axes[0].set_title(f"Selected AgNP conditions with strategy = {strategy}")

                best_trace = [y_catalog[chosen[:i]].min() for i in range(n_init, len(chosen) + 1)]
                axes[1].plot(range(len(best_trace)), best_trace, marker="o", linewidth=2.2, color="#1f77b4")
                axes[1].axhline(y_catalog.min(), linestyle="--", color="black", label="Best in catalog")
                axes[1].set_xlabel("Campaign step")
                axes[1].set_ylabel("Best observed loss")
                axes[1].set_title("Best-so-far loss over the campaign")
                axes[1].legend(loc="best")
                plt.tight_layout()
                plt.show()

                display(log_df.head(min(10, len(log_df))))


            widgets.interact(
                explore_catalog_campaign,
                strategy=widgets.Dropdown(options=["ei", "random"], value="ei", description="strategy"),
                n_init=widgets.IntSlider(value=6, min=4, max=12, step=1, description="initial"),
                n_steps=widgets.IntSlider(value=12, min=4, max=24, step=1, description="steps"),
                seed=widgets.IntSlider(value=11, min=0, max=40, step=1, description="seed"),
            )
            """
        ),
        code(
            """
            ei_traces, ei_final = summarize_strategy("ei", runs=40)
            random_traces, random_final = summarize_strategy("random", runs=40)

            x_axis = np.arange(ei_traces.shape[1])
            ei_mean = ei_traces.mean(axis=0)
            ei_std = ei_traces.std(axis=0)
            random_mean = random_traces.mean(axis=0)
            random_std = random_traces.std(axis=0)

            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(x_axis, ei_mean, label="Expected improvement", linewidth=2.5, color="#1f77b4")
            ax.fill_between(x_axis, ei_mean - ei_std, ei_mean + ei_std, alpha=0.2, color="#1f77b4")
            ax.plot(x_axis, random_mean, label="Random search", linewidth=2.5, color="#ff7f0e")
            ax.fill_between(x_axis, random_mean - random_std, random_mean + random_std, alpha=0.2, color="#ff7f0e")
            ax.axhline(y_catalog.min(), color="black", linestyle="--", label="Best recipe in catalog")
            ax.set_xlabel("Number of evaluated conditions beyond the initial design")
            ax.set_ylabel("Best observed loss so far")
            ax.set_title("Bayesian optimization usually finds better AgNP recipes faster than random search")
            ax.legend(loc="best")
            plt.tight_layout()

            print(f"Mean final best loss with EI:     {ei_final.mean():.4f}")
            print(f"Mean final best loss with random: {random_final.mean():.4f}")
            print(f"Global best loss in catalog:      {y_catalog.min():.4f}")
            """
        ),
        md(
            """
            **Teaching point:** this is exactly the experimental-budget argument for Bayesian optimization.
            If we only get a handful of synthesis rounds, we want each next experiment to be as informative as possible.
            """
        ),
        md(
            """
            ## 4. Multi-Objective Thinking: Cathode Design Tradeoffs

            Many materials problems are not truly single-objective.
            A battery cathode recipe might increase capacity but hurt cycle life or cost.
            Below is a synthetic example to visualize why **Pareto fronts** matter.
            """
        ),
        code(
            """
            nickel_fraction = np.linspace(0.45, 0.90, 80)
            calcination_temp = np.linspace(650, 850, 80)
            n_mesh, t_mesh = np.meshgrid(nickel_fraction, calcination_temp)

            # Synthetic response surfaces chosen to create a realistic tradeoff.
            capacity = (
                175
                + 28 * np.exp(-((n_mesh - 0.76) / 0.12) ** 2 - ((t_mesh - 780) / 65) ** 2)
                + 4 * np.sin(8 * n_mesh)
            )
            stability = (
                82
                + 16 * np.exp(-((n_mesh - 0.60) / 0.10) ** 2 - ((t_mesh - 710) / 80) ** 2)
                - 7 * np.maximum(n_mesh - 0.72, 0)
            )
            cost = 40 + 70 * n_mesh + 0.04 * (t_mesh - 650)

            design_df = pd.DataFrame(
                {
                    "nickel_fraction": n_mesh.ravel(),
                    "calcination_temp": t_mesh.ravel(),
                    "capacity": capacity.ravel(),
                    "stability": stability.ravel(),
                    "cost_index": cost.ravel(),
                }
            )


            def pareto_mask(values):
                # Keep only points that are not dominated on both objectives.
                n_points = values.shape[0]
                mask = np.ones(n_points, dtype=bool)
                for i in range(n_points):
                    if not mask[i]:
                        continue
                    dominated = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
                    mask[dominated] = False
                    mask[i] = True
                return mask


            pareto_points = design_df.loc[pareto_mask(design_df[["capacity", "stability"]].to_numpy())].copy()
            pareto_points = pareto_points.sort_values("capacity")
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(9, 7))
            scatter = ax.scatter(
                design_df["capacity"],
                design_df["stability"],
                c=design_df["cost_index"],
                cmap="viridis",
                alpha=0.35,
                s=22,
                label="Candidate recipes",
            )
            ax.plot(
                pareto_points["capacity"],
                pareto_points["stability"],
                color="#d62728",
                linewidth=2.5,
                label="Pareto front",
            )
            ax.set_xlabel("Discharge capacity")
            ax.set_ylabel("Stability metric")
            ax.set_title("Synthetic cathode design tradeoff surface")
            ax.legend(loc="lower left")
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Cost index")
            plt.tight_layout()
            """
        ),
        md(
            """
            ## 5. Key Takeaways for Class

            - A Gaussian process is the **surrogate**; Bayesian optimization is the **decision loop** built on top of it.
            - Acquisition functions tell us where to sample next by mixing promise and uncertainty.
            - Materials science is full of small, expensive, noisy datasets, so BO is often more practical than brute-force search.
            - The AgNP example shows BO on a real dataset.
            - The synthetic cathode example shows why many materials problems naturally become **multi-objective**.
            - In the companion notebook, we will use **Honegumi** and the **Honegumi RAG assistant** to generate Ax-based Bayesian optimization code from structured options or natural-language problem descriptions.
            """
        ),
    ]
    return nb


def build_honegumi_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(metadata=notebook_metadata())
    nb.cells = [
        md(
            """
            # Honegumi for Materials Bayesian Optimization

            This notebook showcases two related tools:

            1. **Honegumi** for deterministic template generation of Ax-based Bayesian optimization code.
            2. **Honegumi RAG Assistant** for turning a natural-language problem description into Honegumi parameters and then into a fuller Bayesian optimization script.

            The goal is to make it easy to go from a materials problem statement to a runnable starting point.
            """
        ),
        code(
            """
            from pathlib import Path
            from pprint import pprint

            import ipywidgets as widgets
            import pandas as pd
            from IPython.display import Code, Markdown, display

            import honegumi
            from honegumi.ax._ax import option_rows
            from honegumi.ax.utils import constants as cst
            from honegumi.core._honegumi import Honegumi
            from honegumi_rag_assistant.app_config import settings
            from honegumi_rag_assistant.extractors import ParameterExtractor
            from honegumi_rag_assistant.orchestrator import run_from_text

            script_template_dir = honegumi.ax.__path__[0]
            core_template_dir = honegumi.core.__path__[0]

            hg = Honegumi(
                cst,
                option_rows,
                script_template_dir=script_template_dir,
                core_template_dir=core_template_dir,
                script_template_name="main.py.jinja",
                core_template_name="honegumi.html.jinja",
            )

            generated_dir = Path("generated")
            generated_dir.mkdir(exist_ok=True)
            settings.reload_from_env()
            """
        ),
        md(
            """
            ## 1. The Honegumi Option Space

            Honegumi exposes a finite set of high-level switches that describe the shape of the Bayesian optimization problem.
            That makes it especially useful for teaching because students can think in terms of:

            - single vs. multi-objective,
            - existing data vs. greenfield optimization,
            - batch vs. sequential evaluation,
            - categorical variables and constraints.
            """
        ),
        code(
            """
            options_df = pd.DataFrame(
                [
                    {
                        "name": row["name"],
                        "display_name": row["display_name"],
                        "options": row["options"],
                        "hidden": row["hidden"],
                        "disabled": row["disable"],
                    }
                    for row in option_rows
                ]
            )
            display(options_df)
            """
        ),
        md(
            """
            ### Quick Interpretation Guide

            Honegumi is helpful in class because each switch corresponds to a design decision that students can reason about before writing code.
            """
        ),
        code(
            """
            decision_guide = pd.DataFrame(
                [
                    {"Decision": "objective", "Teaching question": "Are we optimizing one materials property or balancing several at once?"},
                    {"Decision": "existing_data", "Teaching question": "Do we already have historical experiments to warm-start the model?"},
                    {"Decision": "categorical", "Teaching question": "Is one of the variables a discrete choice like solvent or substrate?"},
                    {"Decision": "composition_constraint", "Teaching question": "Do the fractions need to add up to a total composition?"},
                    {"Decision": "synchrony", "Teaching question": "Can the lab evaluate one candidate at a time or a batch in parallel?"},
                ]
            )
            display(decision_guide)
            """
        ),
        md(
            """
            ## 2. Deterministic Template Generation with Honegumi

            We will start with a materials-flavored use case:

            - optimize a cathode formulation,
            - use existing screening data,
            - enforce composition-aware reasoning,
            - run experiments in small parallel batches.
            """
        ),
        code(
            """
            manual_problem = {
                "objective": "Single",
                "model": "Default",
                "task": "Single",
                "categorical": False,
                "sum_constraint": False,
                "order_constraint": False,
                "linear_constraint": False,
                "composition_constraint": True,
                "custom_threshold": False,
                "existing_data": True,
                "synchrony": "Batch",
                "visualize": True,
            }

            options_model = hg.OptionsModel(**manual_problem)
            template_code, resolved_problem = hg.generate(options_model, return_selections=True)

            manual_template_path = generated_dir / "honegumi_materials_template.py"
            manual_template_path.write_text(template_code, encoding="utf-8")

            print("Resolved Honegumi selections:")
            pprint(resolved_problem)
            print(f"\\nTemplate written to: {manual_template_path}")
            display(Code("\\n".join(template_code.splitlines()[:80]), language="python"))
            """
        ),
        code(
            """
            scenario_library = pd.DataFrame(
                [
                    {
                        "Scenario": "Ag nanoparticle recipe tuning",
                        "objective": "Single",
                        "task": "Single",
                        "existing_data": True,
                        "composition_constraint": False,
                        "categorical": False,
                        "synchrony": "Single",
                    },
                    {
                        "Scenario": "Battery cathode composition screening",
                        "objective": "Single",
                        "task": "Single",
                        "existing_data": True,
                        "composition_constraint": True,
                        "categorical": False,
                        "synchrony": "Batch",
                    },
                    {
                        "Scenario": "Perovskite processing with solvent choice",
                        "objective": "Single",
                        "task": "Single",
                        "existing_data": False,
                        "composition_constraint": False,
                        "categorical": True,
                        "synchrony": "Single",
                    },
                    {
                        "Scenario": "Cathode capacity vs. stability tradeoff",
                        "objective": "Multi",
                        "task": "Single",
                        "existing_data": True,
                        "composition_constraint": True,
                        "categorical": False,
                        "synchrony": "Batch",
                    },
                ]
            )
            display(scenario_library)
            """
        ),
        md(
            """
            ### Interactive Scenario Picker

            This widget is useful for discussion because students can switch between materials scenarios and immediately see how the Honegumi settings and template change.
            """
        ),
        code(
            """
            scenario_map = {
                "Ag nanoparticle recipe tuning": {
                    "objective": "Single",
                    "model": "Default",
                    "task": "Single",
                    "categorical": False,
                    "sum_constraint": False,
                    "order_constraint": False,
                    "linear_constraint": False,
                    "composition_constraint": False,
                    "custom_threshold": False,
                    "existing_data": True,
                    "synchrony": "Single",
                    "visualize": True,
                },
                "Battery cathode composition screening": {
                    "objective": "Single",
                    "model": "Default",
                    "task": "Single",
                    "categorical": False,
                    "sum_constraint": False,
                    "order_constraint": False,
                    "linear_constraint": False,
                    "composition_constraint": True,
                    "custom_threshold": False,
                    "existing_data": True,
                    "synchrony": "Batch",
                    "visualize": True,
                },
                "Perovskite processing with solvent choice": {
                    "objective": "Single",
                    "model": "Default",
                    "task": "Single",
                    "categorical": True,
                    "sum_constraint": False,
                    "order_constraint": False,
                    "linear_constraint": False,
                    "composition_constraint": False,
                    "custom_threshold": False,
                    "existing_data": False,
                    "synchrony": "Single",
                    "visualize": True,
                },
                "Cathode capacity vs. stability tradeoff": {
                    "objective": "Multi",
                    "model": "Default",
                    "task": "Single",
                    "categorical": False,
                    "sum_constraint": False,
                    "order_constraint": False,
                    "linear_constraint": False,
                    "composition_constraint": True,
                    "custom_threshold": True,
                    "existing_data": True,
                    "synchrony": "Batch",
                    "visualize": True,
                },
            }


            def show_honegumi_scenario(scenario_name):
                scenario = scenario_map[scenario_name]
                scenario_model = hg.OptionsModel(**scenario)
                scenario_code, scenario_resolved = hg.generate(scenario_model, return_selections=True)
                display(pd.DataFrame([scenario_resolved]))
                display(Markdown("**Template preview**"))
                display(Code("\\n".join(scenario_code.splitlines()[:60]), language="python"))


            widgets.interact(
                show_honegumi_scenario,
                scenario_name=widgets.Dropdown(
                    options=list(scenario_map.keys()),
                    value="Battery cathode composition screening",
                    description="Scenario",
                    layout=widgets.Layout(width="70%"),
                ),
            )
            """
        ),
        md(
            """
            ## 3. From Natural Language to Honegumi Parameters

            The Honegumi RAG Assistant adds an LLM-based parameter selector on top of Honegumi.
            The first step is to interpret a free-text problem description and infer the Honegumi settings.

            This live step requires an `OPENAI_API_KEY`.
            """
        ),
        code(
            """
            problem_description = (
                "Optimize a nickel-rich cathode synthesis for maximum discharge capacity while keeping "
                "cycling degradation low. We already have historical data from prior formulations, the "
                "composition fractions should add up to the total precursor blend, and we can test four "
                "candidate recipes per experimental round."
            )

            print(problem_description)
            """
        ),
        code(
            """
            if settings.openai_api_key:
                extracted = ParameterExtractor.invoke(problem_description)
                pprint(extracted)
            else:
                extracted = {"bo_params": None}
                print("OPENAI_API_KEY is not set, so live parameter extraction is skipped.")
                print("Add the key to a local .env file and rerun this cell to exercise the RAG assistant.")
            """
        ),
        code(
            """
            if extracted.get("bo_params"):
                rag_options = hg.OptionsModel(**extracted["bo_params"])
                rag_template_code = hg.generate(rag_options)
                rag_template_path = generated_dir / "honegumi_rag_selected_template.py"
                rag_template_path.write_text(rag_template_code, encoding="utf-8")
                print(f"RAG-selected template written to: {rag_template_path}")
                display(Code("\\n".join(rag_template_code.splitlines()[:80]), language="python"))
            else:
                print("No extracted parameters available yet.")
            """
        ),
        md(
            """
            ## 4. Full Honegumi RAG Workflow

            The full pipeline can:

            1. infer Honegumi parameters from the problem description,
            2. generate a deterministic skeleton,
            3. retrieve Ax documentation context from a vector store,
            4. write a fuller Bayesian optimization script.

            The vector store is optional, but recommended for better retrieval quality.
            """
        ),
        code(
            """
            vectorstore_path = Path(settings.retrieval_vectorstore_path) if settings.retrieval_vectorstore_path else Path("data/processed/ax_docs_vectorstore")
            print(f"Configured vector store path: {vectorstore_path}")
            print(f"Exists: {vectorstore_path.exists()}")
            print("\\nRecommended one-time build command:")
            print("uv run python -m honegumi_rag_assistant.build_vector_store")
            """
        ),
        code(
            """
            if settings.openai_api_key:
                try:
                    rag_code = run_from_text(
                        problem_description,
                        output_dir=str(generated_dir),
                        debug=False,
                        enable_review=False,
                    )
                    print("Full RAG pipeline completed.")
                    display(Code("\\n".join(rag_code.splitlines()[:100]), language="python"))
                except Exception as exc:
                    print(f"Full RAG pipeline did not complete: {exc}")
                    print("If retrieval is the blocker, build the vector store with:")
                    print("uv run python -m honegumi_rag_assistant.build_vector_store")
            else:
                print("Skipping the live RAG pipeline because OPENAI_API_KEY is not set.")
            """
        ),
        md(
            """
            ## 5. Useful Commands Outside the Notebook

            ```powershell
            uv sync
            uv run jupyter lab
            uv run honegumi-rag --help
            uv run python -m honegumi_rag_assistant.build_vector_store
            ```

            Once an API key is present in `.env`, you can also run the assistant directly from the terminal and describe your materials optimization problem interactively.
            """
        ),
    ]
    return nb


def main() -> None:
    outputs = {
        ROOT / "bayesian_optimization_deep_dive.ipynb": build_bayesian_optimization_notebook(),
        ROOT / "honegumi_materials_workflow.ipynb": build_honegumi_notebook(),
    }

    for path, notebook in outputs.items():
        nbf.write(notebook, path)
        print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
