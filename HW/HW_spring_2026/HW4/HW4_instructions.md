# HW4: Bayesian Inference, Gaussian Processes, and Bayesian Optimization (Spring 2026)

## Overview
**A starter Jupyter Notebook** `HW4_spring2026.ipynb` is provided in the assignment folder. You are encouraged to use it to develop and test your code, but the preferred final submission is a Python script.

The goal of this assignment is to build a complete conceptual and practical arc through Bayesian methods for materials science, from the mathematical foundations of inference through Gaussian Process regression to active learning via Bayesian Optimization.

**Assignment Structure:**
- **Part 1:** Bayesian Inference and Information Theory Foundations (~1 hours)
- **Part 2:** Gaussian Process Fundamentals (~1 hours)
- **Part 3:** Bayesian Optimization by Hand (~1~2 hours)
- **Part 4:** Bayesian Hyperparameter Tuning with Optuna (~1–2 hours)

**Total Estimated Time:** 4-6 hours | **Total Points:** 100 (+ 6 bonus)

**Submission:** Submit a Python script named `hw4_yourname.py` OR a completed `HW4_spring2026.ipynb`. All plots must be saved to files — do not use `plt.show()`. All written answers and discussion must go in a separate markdown file `hw4_yourname_answers.md` OR in dedicated markdown cells in the Jupyter notebook. Do not put discussion answers inside code comments or print statements.

**Data Requirements:**
- `data/crossed_barrel_dataset.csv` — geometric parameters (n, θ, r, t) → toughness (Parts 3–4)
- `data/cp_data_cleaned.csv` — formula, temperature T, heat capacity Cp (Part 4)

**Important:** Use `random_state=88` everywhere randomness is involved.

---

## Part 1: Bayesian Inference and Information Theory Foundations

### Background

You have just joined a research group screening yttria-stabilized zirconia (YSZ) ceramic coatings for thermal barrier applications in gas turbine blades. Each coating is deposited by air plasma spray onto a superalloy substrate and then subjected to a thermal cycling adhesion test at 1100 °C: the coating either survives 100 cycles intact (pass) or spalls off the substrate (fail). Your group is trying to establish whether this particular deposition process is reliable enough for production — but lab time is limited, and each test run ties up the furnace for several days.

Before you run a single experiment, you already know things: similar YSZ processes in the literature show pass rates scattered between 40% and 80%, and your process parameters are close to those of a moderately successful group. That qualitative knowledge is your **prior**. As you run tests and collect results, each outcome gives you new evidence — the **likelihood** — and combining the two produces an updated **posterior** distribution over the true pass rate θ.

This is the core equation of Bayesian inference:
```
P(θ | data) ∝ P(data | θ) · P(θ)
   posterior     likelihood    prior
```

In GP regression and Bayesian Optimization, the same logic applies — except that θ is not a scalar pass rate but an entire unknown *function* mapping compositions or process parameters to material properties. Part 1 builds the conceptual and mathematical foundation you need before those methods make sense.

---

### Task 1.1: Bayesian Updating with a Conjugate Prior

Your first job in the lab is to run adhesion tests and update your belief about the true pass rate θ as each result arrives. You want to track not just your best estimate of θ but your *full distribution* over it — so that you can report a credible interval to your PI rather than a single number.

The update rule for this problem has a clean closed form: if your current belief is `Beta(α, β)` and you observe a pass, the posterior is `Beta(α+1, β)`. A fail gives `Beta(α, β+1)`. This is because the Beta and Bernoulli distributions are *conjugate* — a property that makes sequential updating tractable by hand.

**Setup:**
- Model the pass rate as `P(θ) = Beta(α₀, β₀)`
- Each test result is a Bernoulli draw with likelihood `P(data | θ) = θˢ · (1−θ)ᶠ`
- The posterior after s passes and f fails is `Beta(α₀ + s, β₀ + f)`

**Requirements:**
- Your prior knowledge from the literature — that similar YSZ processes pass somewhere in the middle range, with no strong asymmetry — translates to `Beta(α₀=3, β₀=3)`. The parameters α₀ and β₀ can be interpreted as *virtual prior observations*: α₀=3 is like having seen 2 previous passes (α−1), β₀=3 like having seen 2 previous fails (β−1), giving an effective prior sample size of α₀+β₀−2 = 4 pseudo-observations. Plot the prior and label its mean and mode.
- Your technician runs the first ten test coupons in order; the results are: `[1, 1, 0, 1, 0, 0, 1, 1, 1, 0]` (1 = survived 100 thermal cycles, 0 = spalled). After each observation, update α and β. Plot the posterior at stages 1, 3, 5, and 10 overlaid on a single axes; label each curve with its (s, f) count.
- For the final posterior, compute and print: the posterior mean, posterior mode, 95% credible interval, and the MLE estimate `s/(s+f)`.
- Answer the following questions:
  - Why is the Beta distribution a natural choice for modeling a pass rate θ? Your answer should address the support of the distribution and what it means for a prior family to be *conjugate* with the Bernoulli likelihood.
  - The posterior mean is pulled toward the prior mean relative to the MLE. Is this a bug or a feature in a small-data setting? What happens to this pull as n → ∞?
  - A 95% credible interval and a 95% frequentist confidence interval are computed differently and interpreted differently. State the correct interpretation of each.

**Deliverables:**
- `bayes_posterior_update.png` — posterior curves at 4 stages
- Printed posterior statistics (mean, mode, credible interval, MLE)
- Written answers to the three questions

---

### Task 1.2: Prior Sensitivity Analysis

Now imagine three different researchers in your group, each about to analyze the same 10 test results but starting from different priors.

The first researcher is brand-new to the project — no literature background, no institutional knowledge. They treat every pass rate as equally plausible. The second is you from Task 1.1: you have some familiarity with similar processes and expect the pass rate to be somewhere in the middle range, but you are not strongly committed. The third is a senior postdoc who spent last year working with a collaborating group that reported ~85% pass rates on a similar YSZ process. They have adopted that result as their prior. What the postdoc does not yet know is that the collaborating group used a different substrate grit-blasting protocol — a subtle difference that turns out to matter for adhesion.

All three researchers see the same 10 results: `[1, 1, 0, 1, 0, 0, 1, 1, 1, 0]`.

**Requirements:**
- The three priors map directly to the three researchers' situations:
  - New student, no background: `Beta(1, 1)` — the uniform distribution, equivalent to zero prior observations. Every θ ∈ [0,1] is equally plausible.
  - You (Task 1.1): `Beta(3, 3)` — weakly informative, encoding the equivalent of ~4 virtual observations split evenly between pass and fail, centered at θ = 0.5.
  - Senior postdoc: `Beta(10, 2)` — strongly informative, encoding the equivalent of ~10 virtual observations biased heavily toward passing (α−1=9 virtual passes, β−1=1 virtual fail), reflecting the collaborator's ~85% success rate. The total effective prior strength is α+β−2 = 10 pseudo-observations — equal in weight to the 10 real experiments you are about to share.

  Compute the final posterior for each and plot all three on the same axes. Save as `bayes_prior_sensitivity.png`.
- Answer the following questions:
  1. How different are the three posteriors after 10 observations? Characterize the difference quantitatively using posterior means and 95% credible intervals.
  2. The postdoc's `Beta(10,2)` prior has effective strength equal to 10 virtual observations. Your actual dataset also has 10 observations. Given this rough balance, reason qualitatively about how many *additional* real experiments the postdoc would need before their posterior converges close to the data-driven estimate.
  3. Prior choice is most dangerous when experiments are expensive and knowledge transfers imperfectly from related systems. Describe a concrete scenario in your own materials domain where you would (a) confidently use an informative prior, and (b) deliberately start with a weakly informative or uninformative prior.

**Deliverables:**
- `bayes_prior_sensitivity.png`
- Written answers to the three questions

---

### Task 1.3: From Parameters to Functions — The Predictive Distribution

The pass/fail tests in Tasks 1.1–1.2 tell you whether your YSZ process is reliable, but not *why* some coatings fail. Your group suspects that coating porosity is a key driver: higher porosity lowers thermal conductivity, which is good for insulation, but too much porosity weakens adhesion. You are now characterizing thermal conductivity (κ) as a function of porosity (φ) by measuring five coupons with different spray standoff distances — each distance produces a different mean porosity, and each κ measurement carries some uncertainty from the laser flash diffusivity technique.

You expect κ to decrease roughly linearly with φ over the range you can achieve (5–20% porosity). A single least-squares fit gives you one line through the data. But before your first measurement, *every* plausible slope and intercept is consistent with your prior knowledge. After five measurements, only lines near the data remain consistent. A Bayesian linear model tracks this entire distribution — and this is the direct conceptual ancestor of a Gaussian Process. A GP is Bayesian linear regression taken to the limit of infinitely many basis functions, so the same intuition applies exactly.

**Setup:** Consider `y = w₀ + w₁x + ε`, where x is porosity (normalized so your measurement range spans roughly [−3, 3]) and y is thermal conductivity. `ε ~ N(0, σ²)` captures laser flash measurement noise. Place a prior `w ~ N(0, I)` on the weights — before any measurements, slopes and intercepts of order ~1 in normalized units are typical, with no strong directional preference. The posterior over `w` given observations has a closed-form solution. Derive or look up this update and implement it yourself — do not use a library that does it automatically.

> **Derivation hint:** Build a design matrix **Φ** of shape *(n × 2)* whose rows are `[1, xᵢ]`.
> With prior **w** ~ N(**0**, **I**) and likelihood **y** | **Φ**, **w** ~ N(**Φw**, σ²**I**):
> ```
> Posterior covariance:  Σ_w = (ΦᵀΦ / σ² + I)⁻¹
> Posterior mean:        μ_w = Σ_w @ Φᵀ @ y / σ²
>
> Predictive mean at x*:      μ*(x*) = φ(x*)ᵀ μ_w
> Predictive variance at x*:  σ²*(x*) = φ(x*)ᵀ Σ_w φ(x*) + σ²
> ```
> where `φ(x*) = [1, x*]`. Sample lines from the posterior by drawing
> **w** ~ N(**μ_w**, **Σ_w**) via `rng.multivariate_normal(mu_w, Sigma_w)`.

**Requirements:**
1. Sample 20 weight vectors from the prior `w ~ N(0, I)` and plot the corresponding κ–φ lines over `x ∈ [−3, 3]`. This is the **prior predictive distribution** — every line that is plausible before any measurement. Save as `bayes_prior_predictive.png`.

2. Generate 5 noisy measurements from a decreasing linear κ–φ trend of your choice with `σ=0.3` (corresponding to typical laser flash precision of ~0.15 W/m·K in the original units). Compute the posterior over weights and sample 20 lines from it. Plot them alongside the data. Save as `bayes_posterior_predictive.png`.

3. Answer the following questions:
   - Describe what changed between the prior and posterior samples. What specifically constrained the lines?
   - The true κ–φ relationship in YSZ is approximately linear but has a slight upturn at very low porosity (the coating becomes denser and more crystalline). If your 5 measurements were all in the middle porosity range, how would the Bayesian linear model behave when extrapolating to very low porosity? What would the predictive uncertainty look like there?
   - A GP replaces the two-dimensional weight vector with an infinite-dimensional function. Given the intuition from this exercise, what do you expect the GP posterior to look like at compositions (or porosity values) far from your training data versus near them?

**Deliverables:**
- `bayes_prior_predictive.png` and `bayes_posterior_predictive.png`
- Written answers to the three questions

---

### Task 1.4: Entropy, Conditional Entropy, and Joint Entropy

At the end of your YSZ campaign, your PI asks: "After 10 tests, how much do we still not know about our process?" You could show them the posterior plot from Task 1.1 — but that requires them to interpret a curve. What you want is a single number that captures *how spread out* your belief still is. Shannon entropy is that number.

For a discrete distribution, `H(X) = -∑ p(x) ln p(x)`. For a continuous distribution like the Beta, there is an analogous quantity called **differential entropy**, with a closed form involving the digamma function ψ(·) and the beta function B(α, β). A flat distribution has high entropy; a sharply concentrated one has low entropy. Crucially, the chain rule of entropy connects joint and conditional uncertainty:
```
H(θ, data) = H(data) + H(θ | data) = H(θ) + H(data | θ)
```
This identity will be essential when you compute mutual information and expected information gain in Task 1.7.

**Requirements:**

1. Look up or derive the differential entropy of a Beta(α, β) distribution. Compute and plot it at each of the 10 observation stages from Task 1.1. Save as `entropy_vs_observations.png`. Does entropy decrease monotonically? Why or why not — and what does a non-monotonic answer tell you about how individual experiments can be more or less informative?

2. Discretize θ into 200 bins over [0, 1] — fine enough that the approximation error is negligible relative to the distributional differences you are measuring. Use this discrete approximation to compute the entropy of each of the three priors from Task 1.2 (the new student, you, and the postdoc) and their final posteriors after all 10 results. Print a table: prior name | prior entropy | posterior entropy | entropy reduction. Which researcher started with the least uncertainty? Which ended with the least?

3. Compute `H(θ | data)` — the conditional entropy of θ given the observed sequence — after 5 observations and after 10 observations. Numerically verify the chain rule identity `H(θ, data) = H(data) + H(θ | data)` for the 10-observation case. Describe your approach.

4. Answer the following questions:
   - After five experiments, the entropy of your Beta posterior is still relatively high. What are two distinct reasons this could happen — one related to your prior and one related to the data you happened to observe?
   - Entropy is a property of a single distribution and is symmetric in the sense that `H(X) = H(X)`. KL divergence (Task 1.5) is asymmetric. When you are comparing two distributions — say, a posterior to a prior — why does direction matter in a way that entropy alone cannot capture?
   - Differential entropy can be negative. Construct a simple example (any continuous distribution) where this occurs, and explain physically what "negative uncertainty" means in that context.

**Deliverables:**
- `entropy_vs_observations.png`
- Printed entropy table (prior vs. posterior for all three priors)
- Written answers to the three questions

---

### Task 1.5: KL Divergence and f-Divergences

Entropy tells you how uncertain a single distribution is. After 10 coating tests, you can quantify your uncertainty with a number. But there is a different question your PI might ask: "How much did we actually *learn* from those 10 experiments?" Entropy alone cannot answer this — you need a way to compare two distributions. **KL divergence** (relative entropy) is that tool:

```
KL(P ‖ Q) = ∫ p(θ) ln [p(θ) / q(θ)] dθ
```

Tracking `KL(posterior_t ‖ prior)` over your campaign tells you how far your updated belief has moved from your starting point — a quantitative measure of cumulative information gained. But KL divergence has a subtlety: it is **asymmetric**. `KL(P‖Q)` and `KL(Q‖P)` answer different questions, and choosing the wrong direction has real consequences — particularly in variational inference (Task 1.8), where minimizing one versus the other leads to qualitatively different approximations.

KL divergence belongs to a broader family called **f-divergences**, defined by:
```
D_f(P ‖ Q) = ∫ q(θ) f(p(θ)/q(θ)) dθ
```
where `f` is any convex function with `f(1) = 0`. Different choices of `f` give KL, reverse KL, χ², Hellinger distance, and total variation — each with different sensitivity to distributional differences and different mathematical properties.

**Requirements:**

1. For each of the 10 observation stages in Task 1.1, compute `KL(posterior_t ‖ prior)` where the prior is your Beta(3,3) — the weakly-informed starting belief from the literature review. Evaluate both distributions on a fine grid over [0,1]. Plot KL divergence vs. number of observations. Save as `kl_divergence_curve.png`. Compare this curve to your entropy curve from Task 1.4 — how do they relate?

2. Compute both `KL(posterior_10 ‖ prior)` and `KL(prior ‖ posterior_10)`. Print both values. Explain the asymmetry in plain language: what physical question does each direction answer in the context of your coating experiment?

3. The f-divergence family generalizes KL. For the choices `f(t) = (t−1)²`, `f(t) = (√t − 1)²`, and `f(t) = ½|t−1|`, compute the corresponding χ² divergence, squared Hellinger distance, and total variation distance between your final `posterior_10` and the Beta(3,3) prior. Print all values alongside the KL. Which is most sensitive to tail differences?

4. The **Jensen-Shannon divergence** symmetrizes KL via a mixture distribution `M = ½(P + Q)`:
   ```
   JSD(P, Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M)
   ```
   Compute JSD between each pair of the three priors from Task 1.2 (three pairs). Which two priors are closest? Does this match your visual impression from `bayes_prior_sensitivity.png`?

5. Answer the following questions:
   - In variational inference you minimize `KL(q ‖ p)` — the *reverse* KL. What distributional behavior does minimizing the reverse KL tend to encourage, and how does this differ from minimizing the forward KL? Why does this matter when the true posterior has multiple modes?
   - Hellinger distance is bounded: `0 ≤ H(P,Q) ≤ 1`. KL is unbounded. Give a concrete scenario in materials science where the unboundedness of KL could cause numerical or practical problems in a workflow, and explain why a bounded divergence would be preferable there.
   - After running 10 coating tests you compute `KL(posterior ‖ prior) ≈ 0.02`. What would you tell your PI about whether those 10 experiments were informative?

**Deliverables:**
- `kl_divergence_curve.png`
- Printed KL (both directions), three f-divergences, three JSD values
- Written answers to the three questions

---

### Task 1.6: Cross-Entropy, Negative Log-Likelihood, and Proper Scoring Rules

Your three researchers from Task 1.2 each made a sequence of predictions before each test result arrived. Researcher 1 (uniform prior) predicted roughly 50% chance of pass throughout. Researcher 3 (strong/wrong prior) confidently predicted ~80% chance of pass from the start. After 10 results, you can ask: whose probabilistic predictions were better?

This requires a **scoring rule** — a function that takes a predicted probability and an observed outcome and returns a number measuring prediction quality. A rule is **proper** if the only way to maximize your expected score is to report your true beliefs. You cannot game a proper scoring rule by hedging or overstating confidence.

The two most common proper scoring rules are the **log score** (negative log-likelihood) and the **Brier score**:
```
Log score (NLL):  S_log = −log q(xᵢ)      for each observation xᵢ
Brier score:      BS = (1/N) ∑ᵢ (pᵢ − oᵢ)²
```
There is also a fundamental identity connecting cross-entropy, entropy, and KL divergence:
```
H(P, Q) = −∑ p(x) log q(x) = H(P) + KL(P ‖ Q)
```
This decomposition means cross-entropy loss in model training decomposes into irreducible noise (the entropy of the data) plus the error of your model (KL divergence). Minimizing cross-entropy is equivalent to minimizing KL divergence between your model Q and the true distribution P — which is the foundation of maximum likelihood estimation.

**Requirements:**

1. For each of the 10 coating observations, use the predictive distribution *just before* that observation arrives (i.e., the posterior from all previous observations) to predict the probability of pass. Compute the per-observation NLL and the overall Brier score for all three researchers. Plot the per-observation NLL as a grouped bar chart, one group per observation, three bars per group. Save as `scoring_rules.png`.

2. Print the cumulative NLL and Brier score for each of the three priors. Which researcher made the best probabilistic predictions? Does the winner match your intuition from Task 1.2?

3. Answer the following questions:
   - The log score penalizes a confident wrong prediction far more than an uncertain one. Evaluate whether this asymmetry is desirable for a materials screening campaign. Give a specific scenario where you would want this property, and one where it could be dangerous.
   - The Brier score is bounded in [0, 1]; the NLL is not. What practical consequence does this have when you compare models across datasets of different sizes or different base rates?
   - Cross-entropy decomposes as `H(P, Q) = H(P) + KL(P ‖ Q)`. During neural network training, P is fixed (the empirical data distribution). What does minimizing the cross-entropy loss actually minimize — and what part of the loss is irreducible regardless of how good your model is?
   - A **calibrated** model should, among all predictions of 70%, be correct 70% of the time. Researcher 3 consistently predicted ~80% but was only right 60% of the time. What does this tell you about their model? Describe how you would construct a calibration (reliability) diagram with more data, and explain whether it is possible to be well-calibrated but have a high NLL.

**Deliverables:**
- `scoring_rules.png`
- Printed cumulative NLL and Brier score for each of the three priors
- Written answers to the four questions

---

### Task 1.7: Mutual Information, Conditional MI, and Expected Information Gain

After running 9 coating tests you have a fairly well-informed posterior. You can afford one more experiment. Two candidate runs are on the table. How do you decide which one to run?

Entropy told you how uncertain you are overall. KL told you how much previous experiments moved your belief. What you need now is a forward-looking quantity: *how much will a specific future measurement reduce my uncertainty?* This is **mutual information** (MI):

```
MI(θ ; Y) = H(θ) − H(θ | Y) = H(Y) − H(Y | θ)
           = KL( P(θ, Y) ‖ P(θ) · P(Y) )
```

MI is symmetric and non-negative. It equals zero if and only if θ and Y are independent — that is, the measurement Y carries no information about θ at all. In a sequential experimental campaign, you want to run the measurement whose outcome Y is most *dependent* on the unknown parameter θ — the one that will move your posterior the most, in expectation.

For a new measurement at location x with unknown outcome y(x), the **Expected Information Gain (EIG)** formalizes this:
```
EIG(x) = E_{y(x)} [ KL( P(θ | y(x), D) ‖ P(θ | D) ) ] = MI(θ ; y(x) | D)
```
EIG equals the expected reduction in entropy that measurement x would produce, averaged over all outcomes you might observe. Choosing the next experiment to maximize EIG is called **entropy search** — a theoretically principled BO acquisition function. In Part 3, you will use UCB and EI; understanding EIG helps you appreciate what those heuristic acquisition functions are approximating.

**Requirements:**

1. Before any coating tests, starting from your Beta(3,3) literature prior, estimate `MI(θ ; single_observation)` — how many nats of information does one binary test result carry about θ? Compute this by numerical integration over θ ∈ [0,1] and both possible outcomes (the coupon passes, or it spalls).

2. Track how `MI(θ ; next_observation | current_data)` evolves as your campaign progresses. Compute it at each of the 10 stages from Task 1.1 and plot it alongside your entropy curve. Save as `mutual_information_curve.png`. What happens to the informativeness of a new test as the posterior concentrates?

3. Using the κ–φ model from Task 1.3, compute EIG for 50 candidate porosity values uniformly spaced across your measurement range. These represent possible next spray conditions you could run. Plot EIG vs. porosity alongside σ(x) from the posterior. Save as `eig_vs_sigma.png`.

4. Answer the following questions:
   - MI is symmetric: `MI(X;Y) = MI(Y;X)`. Does this mean observing Y is "equally informative" about X as observing X is about Y? Think carefully about what symmetry means here versus the physical act of measuring.
   - Looking at your `eig_vs_sigma.png` plot: what is the relationship between EIG and σ(x)? Is EIG just another way to write σ(x), or is there a meaningful difference?
   - You have two candidate coating experiments: one in a process-parameter region your model has high uncertainty about (high σ), and one near a composition that your collaborator believes is theoretically interesting but where you have already sampled nearby points (low σ). How does EIG inform the choice? What does EIG fail to account for that a domain expert might override it on?
   - Give a concrete materials example where `MI(property ; feature_A | feature_B) ≈ 0` — that is, feature A carries no additional information about the property once you already know feature B.

**Deliverables:**
- `mutual_information_curve.png`, `eig_vs_sigma.png`
- Written answers to the four questions

---

### Task 1.8: Differential Entropy of Gaussians and the ELBO

You have been working with the Beta-Bernoulli model because it is one of the rare cases where the posterior is tractable — it stays in the same distributional family as the prior. Gaussian Process regression (Part 2) is another such case. But most Bayesian models encountered in real materials science are not conjugate: hierarchical models of batch-to-batch synthesis variability, neural network interatomic potentials trained with uncertainty, models with latent microstructural state. In all of these, the true posterior `p(θ | data)` has no closed form. You must approximate it.

The dominant approach is **variational inference**: choose a tractable approximate family `q(θ)` and find the member of that family closest to the true posterior. But "closest" requires a distance — and we use KL divergence. This leads directly to the **Evidence Lower Bound (ELBO)**:
```
ELBO(q) = E_q[log p(data | θ)] − KL(q(θ) ‖ p(θ))
```
The first term rewards `q` for explaining the data. The second penalizes `q` for straying too far from the prior. The ELBO is always ≤ log p(data), with equality only when `q` is exactly the true posterior. Maximizing the ELBO is equivalent to minimizing `KL(q ‖ posterior)`.

Before applying this to a complex model, you need to understand the entropy of Gaussian distributions — because the entropy of `q` appears directly in the ELBO when `q` is Gaussian, and because the GP posterior is a Gaussian whose entropy you can compute analytically from the covariance matrix alone:
```
h[N(μ, Σ)] = ½ log det(2πe Σ)
```

**Requirements:**

1. For your Bayesian linear regression from Task 1.3 (the YSZ thermal conductivity vs. porosity model), compute the differential entropy of the posterior predictive distribution at 50 evenly-spaced porosity values across your prediction range. Plot `h(x)` vs. x alongside σ(x). Save as `gp_posterior_entropy.png`. Where is entropy highest — in the center of your measured porosity range or at the edges? Why?

2. Pick 5 evenly-spaced porosity values across your prediction range. Their joint posterior forms a multivariate Gaussian `N(μ*, Σ*)` — not 5 independent Gaussians, because nearby porosity values have correlated predictions. Compute the joint differential entropy and compare it to the sum of the five individual marginal entropies. Which is larger? What does the difference represent, and how does it connect to your result from Task 1.4 on joint vs. conditional entropy?

3. Consider approximating the Beta posterior `p(θ | data)` from Task 1.1 with a Gaussian `q(θ) = N(μ, σ²)`. This is a poor approximation — the Beta has bounded support, the Gaussian does not — but it is instructive. Write out both terms of the ELBO for this case. You do not need to optimize μ and σ²; just identify: what is the reconstruction term, what is the KL regularization term, and what does each push `q` to do? Why is the Gaussian a poor choice here, and what approximate family would be better?

4. Answer the following questions:
   - The ELBO has a data-fit term and a `KL(q ‖ prior)` penalty. If you make `q` arbitrarily flexible — say, a normalizing flow that can represent any distribution — what happens to the KL term as `q → posterior`? What is the practical consequence of removing the KL term entirely?
   - The GP's log marginal likelihood (LML, Task 2.4) is the *exact* log evidence `log p(y | X)`. How does the ELBO relate to the LML? Under what conditions would you use the ELBO instead of the LML for a GP model?
   - Entropy search selects the next experiment by maximizing `EIG(x) = ½ log det(Σ_posterior_with_x) / det(Σ_posterior_without_x)`. Derive this expression from the definition of EIG and the formula for the differential entropy of a multivariate Gaussian. What assumption about the noise model makes this tractable?

**Deliverables:**
- `gp_posterior_entropy.png`
- Printed joint entropy vs. sum of marginal entropies
- Written ELBO term identification and answers to the three questions

---

### [Bonus] Task 1.9: Fisher Information and the Cramér-Rao Bound

Your campaign has given you 10 coating test results and a posterior over θ. But here is a fundamental question you have not yet asked: *how much could any estimator possibly narrow down θ from a single binary test result, regardless of how clever the analysis?* This is what **Fisher information** quantifies.

Fisher information `I(θ)` measures the expected curvature of the log-likelihood at a given value of θ:
```
I(θ) = E[ (∂/∂θ log p(x | θ))² ]
```
Sharp curvature means the likelihood changes quickly with θ — each observation strongly constrains where θ can be. Flat curvature means the data is weakly informative. The **Cramér-Rao bound** turns this into a hard floor on estimator variance:
```
Var(θ̂) ≥ 1 / I(θ)
```
This is not a computational limitation — it is a fundamental information-theoretic bound. No matter how sophisticated your analysis, you cannot estimate θ with variance smaller than `1/I(θ)` from n i.i.d. observations (the bound becomes `1/(n·I(θ))` for n samples).

Fisher information is also the curvature of the KL divergence near the identity: `KL(P_θ ‖ P_{θ+dθ}) ≈ ½ I(θ) dθ²`. This makes it the natural metric on the space of probability distributions — the basis of information geometry and natural gradient optimization.

**Requirements:**

1. For the Bernoulli likelihood, analytically derive `I(θ)`. Show each step. At what value of θ is Fisher information maximized, and what does this mean physically for the coating screening problem?

2. At each of the 10 observation stages, treat the current posterior mean as your estimate `θ̂` and compute the Cramér-Rao bound `1/I(θ̂)`. Plot this alongside the actual posterior variance. Does the bound get tighter as more data arrives?

3. Plot Fisher information and posterior entropy on the same axes vs. number of observations (two y-axes if needed). Save as `fisher_vs_entropy.png`. Describe the relationship between the two quantities — do they move together, or do they diverge?

4. Answer the following questions:
   - Fisher information is additive: `I_n(θ) = n · I₁(θ)` for i.i.d. samples. What does this imply about the rate at which your posterior variance should shrink with sample size? Is this consistent with your entropy curve from Task 1.4?
   - Natural gradient descent scales the gradient update by `I(θ)⁻¹`. Why is this preferable to standard gradient descent for models where parameter space has non-Euclidean geometry? Give a concrete example of where this matters.
   - Fisher information measures how quickly the likelihood changes with θ. How does this connect to the log marginal likelihood curvature you will compute in Task 2.4? Are they measuring the same thing?

**Deliverables:**
- Analytical Fisher information derivation (**markdown cell**, not code comments or `print` statements)
- `fisher_vs_entropy.png`
- Written answers to the three questions (3 bonus pts)

> **Note:** `matplotlib.use('Agg')` is set in the import cell — correct for saving `.png` files,
> but remove it if running interactively in Jupyter (otherwise inline plots will not render).

---

## Part 2: Gaussian Process Fundamentals

### Background

You are the lead experimentalist on a project mapping the elastic modulus of Al-Cu-Mg ternary alloys for aerospace structural applications. The phase diagram is rich — single-phase FCC solid solutions, θ-phase precipitates, S-phase regions — and modulus varies substantially across the composition triangle. There are thousands of candidate compositions. Each requires arc melting, homogenization at 450 °C for 48 hours, rolling to a standard gauge, and nanoindentation across 50 sites: roughly $400 and four days per composition. You have a semester and can realistically measure 30–50 compositions.

After your first 15 measurements you face a question that a random forest or neural network cannot fully answer: *"At a composition I have never tested, what is my best estimate of the modulus — and how uncertain am I in that estimate, given what I know about how smoothly modulus varies in this alloy system?"* A Gaussian Process answers both parts. It gives a predicted mean and a calibrated posterior standard deviation that shrinks near measured compositions and expands in unexplored regions. This uncertainty is not an afterthought — it is what drives the experimental design strategy in Part 3.

A **Gaussian Process** is a distribution over functions. Every finite collection of function values `[f(x₁), ..., f(xₙ)]` follows a multivariate Gaussian, with covariance determined by a **kernel function** `k(x, x')`. The kernel is your physical intuition made mathematical: when you say "I expect modulus to vary smoothly across composition space," you are asserting that nearby compositions should have similar moduli. A kernel encodes that assertion precisely — `k(x, x')` is large when `x` and `x'` are compositionally similar and small when they are far apart. The four tasks in this part progressively build your ability to choose, engineer, evaluate, and interpret GP models on the Al-Cu-Mg system.

GPs are the standard surrogate model for Bayesian Optimization because they are the only common model that delivers both a predicted mean and a *calibrated* uncertainty estimate derived from Bayesian inference.

---

### Task 2.1: GP as a Prior Over Functions

Before you take a single measurement, you must make a modeling choice that encodes everything you believe about how elastic modulus varies across the Al-Cu-Mg composition triangle. This choice is the **kernel**. A GP prior with an RBF kernel says "I expect modulus to vary smoothly — two compositions 2 at.% apart should have nearly identical moduli." A Matérn kernel says "I expect some roughness — phase boundaries can cause abrupt jumps." A periodic kernel would only make sense if you believed modulus repeats as a function of composition, which is physically unreasonable here. Choosing wrong wastes experiments; choosing well means your model learns quickly.

**Physical intuition for each kernel:**

| Kernel | What it assumes | Materials example |
|---|---|---|
| **RBF** | Infinitely smooth, continuous variation | Bandgap in a semiconductor alloy (A₁₋ₓBₓ) as a function of composition x — varies gradually and continuously |
| **Matérn ν=0.5** | Highly irregular, rough, almost nowhere differentiable | Fatigue lifetime vs. applied stress — sensitive to microstructural defects, shows abrupt local variation |
| **Matérn ν=2.5** | Once-differentiable, realistic roughness — the default for most materials properties | Elastic modulus, yield strength, thermal conductivity — smooth enough to interpolate but not perfectly regular |
| **Periodic** | Repeating structure with a fixed period | Magnetization as a function of applied field cycling; Cp variation in a material with a periodic phase transition; properties sampled along a crystallographic direction |
| **Rational Quadratic** | Multi-scale variation (mixture of length scales) | A property showing both long-range compositional trends and shorter-range local ordering effects |

The **length scale** is also physically interpretable: it tells you how far apart two compositions (or temperatures, or process conditions) must be before the GP treats them as essentially uncorrelated. A length scale of 0.1 at.% Cu means the model expects the property to change significantly over tiny composition steps. A length scale of 10 at.% means large composition changes are needed before the model expects anything to be meaningfully different.

**Requirements:**
1. Using `sklearn.gaussian_process.GaussianProcessRegressor` with **no training data**, draw 5 function samples from a GP prior for each of the following kernels over `x ∈ [0, 10]`:
   - **RBF:** `RBF(length_scale=1.0)`
   - **Matérn ν=0.5:** `Matern(length_scale=1.0, nu=0.5)` (Ornstein-Uhlenbeck)
   - **Matérn ν=2.5:** `Matern(length_scale=1.0, nu=2.5)`
   - **Periodic:** `ExpSineSquared(length_scale=1.0, periodicity=3.0)`
   - **Rational Quadratic:** `RationalQuadratic(length_scale=1.0, alpha=0.5)`

   Arrange samples in a 5-panel figure. Save as `gp_prior_samples.png`.

   **Hint:** Call `gp.sample_y(X_grid, n_samples=5)` on an *unfitted* GPR instance.

2. For each kernel, compute the kernel matrix `K` over 50 evenly spaced points in `[0, 10]` and plot it as a heatmap. Save as `gp_kernel_heatmaps.png`.

3. Answer the following questions:
   - What does the length scale control in the RBF kernel? What happens to function samples as length_scale → 0 or → ∞?
   - The Matérn family is parameterized by ν. What does ν control? Why is ν=2.5 the standard choice for BO in materials science?
   - Which kernel would you choose for a property expected to vary periodically with composition? With temperature?
   - What does it mean when two points in the heatmap have high covariance? Low covariance?

**Deliverables:**
- `gp_prior_samples.png`, `gp_kernel_heatmaps.png`
- Written answers to the four questions

---

### Task 2.2: Kernel Engineering — Composite and Anisotropic Kernels

After 15 measurements spread across the Al-Cu-Mg triangle, you notice that the modulus variation has two distinct scales: a broad, slowly-varying trend driven by the overall Al/Cu/Mg ratio (the Al-rich corner is consistently stiffer than the Mg-rich corner), plus finer local fluctuations near the two-phase boundaries where intermetallic precipitates form. No single base kernel captures both. You also notice that modulus is much more sensitive to changes in Cu content than to equivalent changes in Mg content — a 5 at.% shift in Cu moves the modulus meaningfully, while a 5 at.% shift in Mg barely changes it. A single shared length scale misses this.

Kernel engineering lets you encode both observations directly into the model:

- **Sum `k₁ + k₂`:** The property has two additive components. Here: a long-range RBF capturing the broad Al/Cu/Mg trend, plus a short-range RBF or Matérn capturing the local variation near phase boundaries. Total variation = slow background trend + fast local fluctuation.
- **Product `k₁ × k₂`:** The property has an interaction structure — a function whose amplitude or period is itself modulated by another variable. Less relevant here but important in general (e.g., a property that is periodic in one dimension but whose amplitude decays in another).
- **ARD (Automatic Relevance Determination):** Assigns a separate length scale to each input dimension — one for Cu content, one for Mg content, one for Al content (constrained since they sum to 1). The GP learns that Cu matters at a scale of ~3 at.% while Mg matters at ~8 at.%. Without ARD, the model assumes both variables are equally important, which is almost never physically correct in a multi-component system.

**Requirements:**
1. Generate a 1D synthetic dataset: 15 noisy observations from `f(x) = sin(x) + 0.3·sin(5x)` over `[0, 2π]` with `σ_noise = 0.15`.

2. Fit four GPs using `n_restarts_optimizer=5`:
   - **K1:** `RBF(length_scale=1.0)`
   - **K2:** `Matern(nu=2.5)`
   - **K3:** `RBF() + WhiteKernel()`
   - **K4:** `RBF(length_scale=2.0) + RBF(length_scale=0.3)` — long + short range

   For each, plot the posterior mean ± 2σ with the data and the true function. Arrange as a 2×2 panel. Save as `gp_kernel_comparison.png`.

3. Print the optimized kernel hyperparameters and log marginal likelihood for each. Which kernel fits best?

4. On the **crossed-barrel dataset**, fit a GP to predict toughness from `r` and `t` only. Fit two versions:
   - **Isotropic:** single shared length scale
   - **ARD:** `length_scale=[1.0, 1.0]` (one per feature)

   Compare the optimized length scales. Which feature is weighted more? Does this make physical sense?

5. Answer the following questions:
   - When would you add kernels vs. multiply them?
   - What physical knowledge could guide kernel choice for bandgap, elastic modulus, or thermal conductivity?
   - Why does ARD matter for high-dimensional spaces like CBFV (100+ features)?

**Deliverables:**
- `gp_kernel_comparison.png`
- Printed kernel hyperparameters and log marginal likelihoods
- Written answers to the three questions

---

### Task 2.3: The Three Sources of Uncertainty

Your GP model is now trained on 15 Al-Cu-Mg compositions and reports a posterior σ(x) at every candidate point. Your PI looks at the uncertainty map and says: "Great — measure the top-5 highest-σ compositions next." Before you order the raw materials, you pause. The GP's σ(x) is not a single thing — it conflates two fundamentally different types of uncertainty, and measuring them at high-σ points will only help if that uncertainty is reducible. If the uncertainty is irreducible — from nanoindentation scatter across grains, from surface preparation variability, from instrument calibration — then running more experiments at those same conditions will not narrow your GP's predictions at all.

**Background:** The GP's reported σ(x) conflates three types of uncertainty with different implications for experimental design:

| Type | Name | Source | Materials example | Reducible with more data? |
|---|---|---|---|---|
| **Aleatoric** | Observation noise | Measurement error, process variability | Hardness scatter from grain-to-grain variation in a polycrystal; batch-to-batch synthesis variability; XRD peak fitting error | No — it reflects irreducible physical noise in the system |
| **Epistemic** | Model uncertainty | Sparse observations; GP posterior σ(x) | High uncertainty at an unexplored corner of composition space; uncertainty in a temperature range you haven't sampled yet | Yes — more measurements directly reduce this |
| **Hyperparameter** | Parameter uncertainty | Uncertainty in kernel hyperparameters θ | Uncertainty about the true length scale in composition space; not knowing whether properties are smooth or rough | Partially — more data constrains hyperparameters, but sklearn's standard approach ignores this |

**Why does this matter for experimental design?** If σ(x) at a candidate point is high because it is far from any training data (epistemic), measuring it will directly reduce your model's uncertainty and is likely a good investment. If σ(x) is high because the synthesis process is intrinsically noisy (aleatoric), measuring it repeatedly will not improve your model — you are just observing the same scatter. Confusing these two leads to wasted experiments.

**Requirements:**
1. **Aleatoric vs. epistemic:** To isolate these two uncertainty types cleanly, use a 1D stand-in for the modulus landscape: generate 10 measurements of a property that varies non-monotonically with a single composition parameter, using `f(x) = x·sin(x)` over `[0, 10]` with `σ_n = 0.5`. Treat x as a normalized composition axis and σ_n = 0.5 as the nanoindentation scatter (aleatoric: it will not decrease no matter how many times you re-test the same composition). Fit a GP with a `Matern(nu=2.5) + WhiteKernel()` kernel and `normalize_y=True`. Extract the optimized WhiteKernel noise level (aleatoric) and the posterior σ(x) (epistemic). Plot both on the same axes: a flat ± 2σ_aleatoric band and the varying ± 2σ_epistemic band. Mark the 10 measured compositions. Where is epistemic uncertainty highest, and why? Save as `gp_uncertainty_decomposition.png`.

2. **Hyperparameter uncertainty:** When sklearn fits a GP it maximizes the LML — but the LML landscape can have multiple local optima (as you will see in Task 2.4). Using the same 10-point dataset, fit the GP 10 times with `n_restarts_optimizer=0` and different random seeds. Plot all 10 posterior means on the same axes. Compare to a single well-optimized fit using `n_restarts_optimizer=10`. What happens to your modulus predictions across the composition range if you happen to land in the wrong LML optimum? Save as `gp_hyperparameter_sensitivity.png`.

3. Answer the following questions:
   - You observe high σ(x) at a candidate point. Should you evaluate it? Does your answer depend on whether the uncertainty is aleatoric or epistemic?
   - Your GP's noise parameter converges to near zero. Is this a good sign or a warning sign?
   - In a materials synthesis campaign, give concrete examples of aleatoric and epistemic uncertainty.
   - Does the GP's reported σ(x) account for hyperparameter uncertainty? What are the practical consequences in a BO campaign?

**Deliverables:**
- `gp_uncertainty_decomposition.png`, `gp_hyperparameter_sensitivity.png`
- Written answers to the four questions

---

### Task 2.4: Log Marginal Likelihood and Kernel Selection

You have now fitted several GP models to your Al-Cu-Mg modulus data — RBF, Matérn, composite kernels — and each reports a different posterior mean and uncertainty. All of them fit the 15 training points reasonably well. How does sklearn decide on a length scale of 3.2 at.% rather than 0.5 or 15? And how should you choose between two kernels that have similar training error? The answer is the **log marginal likelihood (LML)** — the probability of observing your 15 modulus measurements under the GP model, integrated over all possible function values. It is the Bayesian model evidence, and sklearn maximizes it automatically when you call `.fit()`.

**Background:** sklearn selects kernel hyperparameters by maximizing the **log marginal likelihood (LML)**:
```
log p(y | X, θ) = -½ yᵀ K⁻¹ y  -  ½ log|K|  -  n/2 log(2π)
                    data fit term   complexity term
```
The LML automatically balances fit quality against model complexity — penalizing overly flexible models even when they fit training data well. This is Occam's razor as a consequence of Bayesian inference.

**Materials intuition for the two terms:**
- *Data-fit term* (`-½ yᵀ K⁻¹ y`): How well does the kernel explain the observed property values? A very short length scale (wiggly kernel) can pass through every data point and makes this term large. A very long length scale (flat kernel) ignores local variation and makes this term small.
- *Complexity penalty* (`-½ log|K|`): How "large" is the model's effective hypothesis space? A short length scale allows the GP to explain almost any dataset — it is a very flexible model, and this term penalizes that flexibility. Think of it as the Bayesian equivalent of regularization, but automatically derived from probability theory rather than tuned by hand.

The LML peak is the sweet spot: a kernel flexible enough to describe the data, but not so flexible that it fits noise. This is exactly the bias-variance tradeoff, but handled automatically through Bayesian inference rather than cross-validation.

**Requirements:**
1. Using the 1D synthetic dataset from Task 2.2 (which mimics the kind of multi-scale modulus variation you expect near phase boundaries), manually sweep the RBF length scale from 0.01 to 100 over a log-scale grid of 30 values. At each fixed length scale, fit the GP and record the LML. Plot LML vs. log(length_scale) and mark the optimum. Save as `gp_lml_curve.png`. Interpret the two flanks of the curve: what happens to fit quality and model complexity on each side of the peak?

2. Create a 2D LML surface by sweeping both length scale and noise level (via a WhiteKernel) over a 20×20 log-scale grid. Plot as a heatmap. Save as `gp_lml_surface.png`. Where would an optimizer initialized at a very small length scale end up? What does this tell you about the importance of `n_restarts_optimizer` when fitting a GP to your Al-Cu-Mg data?

3. Answer the following questions:
   - The LML has a data-fit term and a complexity penalty. What happens to each as length_scale → 0? Relate your answer to what you would physically observe if you used that length scale to predict modulus at unmeasured compositions.
   - After fitting five different kernels to your Al-Cu-Mg training data, all five have training RMSE within 0.5 GPa of each other. How would you use LML to make a principled choice between them?
   - What is a practical failure mode of LML-based kernel selection when your 15 Al-Cu-Mg measurements are clustered in one corner of the composition triangle?

**Deliverables:**
- `gp_lml_curve.png`, `gp_lml_surface.png`
- Written answers to the three questions

---

## Part 3: Bayesian Optimization by Hand

### Background

A materials startup called StructureLab is developing 3D-printed polymer lattice crash absorbers for electric vehicle battery pack protection. Their design of choice is the **crossed-barrel lattice** — a tubular structure with a crossed-helix geometry controlled by four parameters: the number of helix repeats (n), the helix angle (θ), the outer radius (r), and the wall thickness (t). The key performance metric is **specific energy absorption (toughness)** — the energy absorbed per unit mass before the structure fails catastrophically. Higher toughness means better crash protection.

StructureLab's simulation team spent three months running ~1800 finite element analyses on a university HPC cluster, sweeping through combinations of the four geometric parameters. Each simulation took 8 hours. They now have a complete table of designs and their simulated toughness values — but simulations are not the same as physical tests. Before they can submit to a supplier, they need physical impact test data from a drop-weight machine. Each physical test requires 3D printing the specimen (6 hours), post-processing, and running the impact test. The total cost is $150 and 3 days per candidate. Their testing budget covers **150 candidates** total.

Their goal is straightforward: from the ~1800 simulated designs, find at least 15 from the highest-performing 5% — the designs most likely to become the final product — while spending as few physical tests as possible. Random screening would require ~30 tests to expect 1.5 top-5% candidates by chance. BO should do much better.

You are the data scientist brought in to run this campaign. The simulation data is `data/crossed_barrel_dataset.csv`. **The rule is: you cannot look at the toughness values in the dataset directly. You can only evaluate a candidate's toughness by querying it from the table — simulating the cost of a physical test.** Your goal is to reach 15 top-5% candidates within 150 queries.

A BO pipeline has three core components, each solving a **distinct** optimization problem:
1. **Surrogate model** — a GP fitted to your observed tests; its hyperparameters are chosen by maximizing the log marginal likelihood
2. **Acquisition function** — a cheap function of (μ, σ) that ranks candidates; the next test is `argmax` of this function over the candidate pool
3. **Data collection policy** — the overall campaign design: how many initial tests, when to stop, whether to query in batches

These solve different problems. The surrogate answers "what is my best model of the data I have?" The acquisition function answers "given that model, which unqueried design looks most promising?" The data collection policy answers "how should I structure the campaign as a whole?" Conflating them is the most common source of bugs in BO implementations.

**Code quality:** all functions must have inline comments on non-obvious logic. Inline comments are part of each task's points (see rubric).

---

### Task 3.1: Understand the Design Space

Before running any optimization, you need to understand what you are working with. Load `data/crossed_barrel_dataset.csv` and characterize the dataset. StructureLab's engineers want to know how many designs are in the top 5% — that is the target population you are trying to find efficiently.

**Requirements:**
- Print the dataset shape, column names, and summary statistics for all four geometric parameters and toughness
- Compute and print the 95th-percentile toughness threshold and the number of designs that exceed it
- Answer: given random guessing, how many tests would you expect to need before finding 15 top-5% candidates?

**Deliverables:**
- Printed statistics, threshold, and top-5% count
- Written answer to the random-baseline question

---

### Task 3.2: Initialize the Campaign

StructureLab's lab manager says you can run 5 physical tests before the BO campaign begins — these will seed the surrogate model. You pick 5 designs at random (use `random_state=88` so your results are reproducible) and look up their toughness values from the simulation table. These become your first observed set; every other design becomes a candidate.

**Requirements:**
- Split the dataset into an observed set (5 initial random samples) and a candidate pool (everything else)
- Print the 5 initial designs and their toughness values
- Note how many of the 5 are already in the top 5% — this sets your baseline before BO begins

**Deliverables:**
- Observed set and candidate pool created; initial 5 samples printed

---

### Task 3.3: Build the Gaussian Process Surrogate

With 5 observations in hand, you fit a GP to build a toughness model over the four-dimensional design space. This model will be queried at every candidate design to produce a predicted mean μ(x) and uncertainty σ(x). Both are needed for the acquisition functions in Tasks 3.4–3.6. Because n, θ, r, and t are on very different physical scales (integers vs. angles in degrees vs. millimeters), you must normalize the features before fitting — otherwise the GP's single length scale will be dominated by whichever parameter happens to have the largest numerical range.

**Requirements:**
- Implement two functions: one that fits a GP with a `Matern(nu=2.5)` + `ConstantKernel` to a set of (X, y) training observations, normalizing X internally with a `StandardScaler` and returning both the fitted model and the scaler; and one that applies a fitted (gp, scaler) pair to a raw candidate array and returns predicted (μ, σ) as 1D numpy arrays. Both functions must have inline comments on non-obvious steps.
- Test your functions on the 5 initial samples. Print μ and σ for 10 randomly selected candidates and confirm that σ > 0 for at least some of them — if σ is zero everywhere, the GP has collapsed and something is wrong.

**Deliverables:**
- `fit_gp` and `predict_gp` functions with inline comments
- Test output confirming σ > 0 for at least some candidates

---

### Task 3.4: Implement Upper Confidence Bound (UCB)

Your first acquisition strategy: score every candidate by its predicted mean plus a multiple of its uncertainty.

```
UCB(x) = μ(x) + κ · σ(x)
```

The parameter κ controls how aggressively you explore uncertain regions. At κ = 0, UCB reduces to pure exploitation — you always test the design with the highest predicted toughness, ignoring whether your model is uncertain there. At large κ, UCB is dominated by σ(x) and you chase uncertainty regardless of mean prediction, which is pure exploration. Neither extreme is good: pure exploitation gets stuck early if your initial model is wrong; pure exploration ignores the information your model already has.

Implement a `ucb` function (with inline comments) that takes arrays of predicted means and standard deviations plus a κ parameter and returns a UCB score array.

**κ sensitivity analysis:** Fit the GP on your 5 initial samples. Compute UCB scores across the entire candidate pool for κ ∈ {0.01, 0.5, 2.0, 5.0, 20.0}. For each κ, print the toughness of the top-ranked candidate and the fraction of top-5% designs in the top-10 UCB-ranked candidates.

Answer the following questions:
- At κ = 0.01, which designs does UCB select and why? What is the risk of this strategy after only 5 observations?
- At κ = 20.0, which designs does UCB select? Is the top-ranked candidate likely to be in the top 5%?
- Is there a principled way to set κ, or is it always a manual choice? (Hint: Task 3.7 introduces an adaptive formula.)

**Deliverables:**
- `ucb` function with inline comments
- κ sensitivity output and written answers

---

### Task 3.5: Implement Expected Improvement (EI)

UCB scores a candidate by its optimistic upper bound on toughness. Expected Improvement asks a different question: *how much above the current best design do I expect this candidate to be, on average?* EI is zero for a candidate that the model predicts is definitely worse than the best design found so far, and large for candidates that have both a reasonable predicted mean and enough uncertainty that they might surprise you. Formally:

```
Z      = (μ(x) − f_best − ξ) / σ(x)
EI(x)  = (μ(x) − f_best − ξ) · Φ(Z)  +  σ(x) · φ(Z)
EI(x)  = 0    if σ(x) = 0
```

where Φ is the standard normal CDF, φ is the standard normal PDF, `f_best` is the best toughness observed so far among your physical tests, and ξ is a small exploration bonus that prevents EI from collapsing once your model becomes very confident.

Implement an `expected_improvement` function (with inline comments) taking (μ, σ, f_best, ξ=0.01). Be careful about the σ = 0 case — it will occur for designs already in your observed set.

**Deliverables:**
- `expected_improvement` function with inline comments

---

### Task 3.6: Implement Probability of Improvement (PI) and Compare All Three

The third acquisition strategy, Probability of Improvement, asks an even simpler question than EI: *what is the probability that this candidate beats the current best, regardless of by how much?*

```
PI(x) = Φ( (μ(x) − f_best − ξ) / σ(x) )
```

PI is easy to compute and interpret — it is literally a probability — but it can be overly conservative. A candidate with 99% probability of a tiny improvement will beat one with 60% probability of a very large improvement, even though the second is clearly the better test from StructureLab's perspective.

Implement a `probability_of_improvement` function (with inline comments) taking (μ, σ, f_best, ξ=0.01).

**Three-way comparison:** Fit the GP on your 5 initial samples. Compute UCB (κ=2.0), EI (ξ=0.01), and PI (ξ=0.01) scores for all candidates. Plot a three-panel scatter where each candidate is a point colored by its acquisition score; mark the top-10 candidates in each panel. Save as `acquisition_comparison.png`.

Answer the following questions:
- Looking at your `acquisition_comparison.png`: do the three functions agree on which designs to test next, or do they identify different regions of the design space? What does disagreement tell you about the tradeoffs between them?
- EI accounts for the magnitude of improvement; PI only asks if improvement occurs. Describe a concrete StructureLab scenario where PI's conservatism would lead to a poor choice that EI would avoid.
- All three acquisition functions reduce to pure exploitation as σ → 0 everywhere. Why does this happen mathematically, and is it the right behavior when your model is very confident?
- UCB has κ; EI and PI have ξ. Are they conceptually equivalent tuning parameters? What is the practical difference in how each affects which candidate gets selected?

**Deliverables:**
- `probability_of_improvement` function with inline comments
- `acquisition_comparison.png`
- Written answers to the four questions

---

### Task 3.8: Run the Campaigns

Now run three full campaigns against StructureLab's design pool. **Before each campaign, reset to the identical 5 initial samples** (`random_state=88`) — every campaign must start from the same initial state so comparisons are fair.

Each campaign follows the same loop: fit the GP to your current observed designs, predict (μ, σ) for every remaining candidate, score them with your acquisition function, "test" the top-ranked candidate by looking up its toughness from the simulation table, move it from the candidate pool to your observed set, and record whether it was a top-5% hit. Stop when you have found 15 top-5% designs or exhausted 150 iterations, whichever comes first.

Run:
- **UCB campaign** using κ = 2.0. Record at each iteration: iteration number, cumulative top-5% count, best toughness found so far, and toughness of the design just queried.
- **EI campaign** using ξ = 0.01 and `f_best = max(y_observed)` at each step.
- **Random baseline** — no GP, no acquisition function. Select a random candidate each step and run for all 150 iterations regardless of how many top-5% designs are found. This is your reference point for "how much does BO actually help?"

Print for each method: total iterations run and total top-5% designs found.

**Deliverables:**
- Three working campaigns with all history arrays stored; printed summary for each

---

### Task 3.7: Regret Analysis and Theoretical Bounds

> **Note:** This task uses the campaign history arrays produced in Task 3.8 above
> (`UCB_history`, `EI_history`, `random_history`). Complete Task 3.8 first.

StructureLab's CEO wants a progress report: after each test, how far are you from the absolute best design their simulation database contains? This is **simple regret** — the gap between the toughest design ever found and the single best design in the full dataset:

```
r_T = f* − max(toughness observed through iteration T)
```

A related quantity, **cumulative regret**, tracks total missed value across all tests — relevant when StructureLab is paying $150 per test and wants to know whether those dollars were well spent:

```
R_T = Σₜ (f* − toughness queried at step t)
```

BO theory proves that for GP-UCB with an adaptive κ that grows slowly with iteration number:
```
κ_t = sqrt(2 · log(N · t² · π² / (6 · δ)))
```
the cumulative regret is sublinear in T — meaning BO gets progressively more efficient as the campaign goes on. The constant in the bound depends on the **maximum information gain** γ_T, which is itself kernel-dependent: smoother kernels (RBF) have smaller γ_T, meaning BO converges faster on smooth toughness landscapes than rough ones.

**Requirements:**
1. Find f* — the true global maximum toughness in the full dataset — and print it. This is your campaign ceiling.

2. Plot simple regret vs. iteration for UCB, EI, and the random baseline on the same axes. Save as `bo_simple_regret.png`. On the same figure, include a fourth curve: a GP-UCB campaign using the adaptive κ_t formula above with N = size of the candidate pool and δ = 0.1.

3. Plot cumulative regret vs. iteration for all three methods. Save as `bo_cumulative_regret.png`.

4. Answer the following questions:
   - StructureLab has two engineers debating campaign strategy. Engineer A says "we need the absolute best design — minimize simple regret." Engineer B says "every bad test wastes money — minimize cumulative regret." Under what conditions is each engineer right?
   - Your toughness landscape has varying smoothness — some regions of the (n, θ, r, t) space vary gradually, others sharply near phase-transition-like geometry changes. What does the theoretical bound imply about how the campaign should perform in those two regions?
   - Does your empirical simple regret curve look roughly like O(1/√T)? If your BO outperforms the theoretical bound, what could explain it? If it underperforms, what are the most likely causes?

**Deliverables:**
- `bo_simple_regret.png` (with adaptive κ_t curve), `bo_cumulative_regret.png`
- Printed f* value
- Written answers to the three questions

---

### Task 3.9: When BO Fails

StructureLab's CEO has now seen your BO results and is excited. Before scaling up to their next project — optimizing a completely different lattice topology with sparser simulation data and higher measurement noise — you write a short technical memo on the failure modes they should watch for. BO is not universally reliable, and understanding when it breaks down is as important as knowing how to run it.

**Requirements:**

1. **Kernel misspecification.** Run a fourth campaign identical to UCB but using an **RBF kernel instead of Matérn**. Plot its simple regret curve alongside the Matérn-UCB campaign. Does kernel choice significantly affect performance on this dataset? Explain why the crossed-barrel toughness landscape might or might not be sensitive to this choice, and describe a material property where you would expect the kernel to matter much more.

2. **Non-Gaussian measurement noise.** In StructureLab's next project, they plan to measure impact energy on 3D-printed specimens that occasionally have voids from a printing defect — producing rare catastrophic low-energy outliers (heavy-tailed noise). To demonstrate the problem: generate a 1D synthetic dataset from `f(x) = sin(x)` over `[0, 2π]` with noise drawn from a Student-t distribution (ν=2, scale=0.5). Fit a standard GP assuming Gaussian noise and plot its posterior against the data. Save as `gp_nongaussian_noise.png`. How does the GP posterior behave near the outliers? What would you do in practice to handle this?

3. **Surrogate optimization vs. acquisition optimization.** Answer the following questions:
   - At the surrogate fitting step, what mathematical quantity is being maximized, and what are the free variables?
   - At the acquisition step, what is being maximized, and over what space?
   - In this assignment, acquisition optimization is `argmax` over a discrete candidate pool — computationally trivial. If StructureLab moved to a continuous design space (any real-valued n, θ, r, t), why would acquisition optimization become hard? What are the standard approaches?

4. **Campaign design.** StructureLab's manufacturing team realizes they can run 5 tests simultaneously on their drop-weight machine in a single day. Answer the following questions:
   - How would you modify your BO loop to query 5 candidates at once instead of 1? What is the simplest approximation, and what does it sacrifice?
   - If a senior engineer looks at your top-10 UCB-ranked candidates and says "three of these are geometrically impossible to 3D print at our resolution — replace them with the next best," is this a problem for your campaign? How does this kind of human override interact with the BO framework?
   - StructureLab now asks: "How do we know when to stop?" Describe a stopping criterion that does not require knowing f* in advance.

**Deliverables:**
- `gp_nongaussian_noise.png`
- Written answers to all four items

---

### Task 3.10: The Final Report to StructureLab

Your 150-iteration campaign is complete. You write up the results to hand to StructureLab's engineering team. This section is the written deliverable — figures and answers that a non-expert can read and act on.

**Required plots:**
1. **`bo_discovery_plot.png`** — scatter of all ~1800 simulated designs (x-axis: design index or one geometric parameter, y-axis: toughness); mark the top-5% threshold; highlight the candidates found by UCB and EI. Show UCB and EI side by side in two subplots so the discovery trajectories can be visually compared.
2. **`bo_cumulative_discovery.png`** — cumulative number of top-5% candidates found vs. iteration: UCB, EI, and random baseline on the same axes. Include a horizontal dashed line at 15.
3. **`bo_best_value_curve.png`** — best toughness found so far vs. iteration for all three methods.

**Answer the following questions:**
1. How many iterations did each method need to find 15 top-5% candidates? Did any fail within 150 iterations? How does this compare to the random-baseline expectation you calculated in Task 3.1?
2. Which acquisition function found the most top-5% designs within 150 tests? Was this result surprising given the theoretical properties of UCB vs. EI?
3. Looking at your `bo_discovery_plot.png` and cumulative discovery curve: did one method find good candidates early and then plateau, while the other found them more gradually? What does this tell you about their exploration-exploitation balance?
4. Looking at your regret curves: does BO meaningfully outperform random search on this dataset? If the gap is smaller than you expected, what properties of the crossed-barrel landscape could explain it?
5. Write a two-paragraph recommendation for StructureLab's next campaign (a different lattice topology with the same budget). Address: which acquisition function to use, how to set κ or ξ, how many initial random tests to run before starting BO, and when to stop.

**Deliverables:**
- `bo_discovery_plot.png`, `bo_cumulative_discovery.png`, `bo_best_value_curve.png`
- Written answers to the five questions

---

## Part 4: Bayesian Hyperparameter Tuning with Optuna

### Background

A ceramics research group at a national lab is developing high-entropy oxide (HEO) ceramics for thermal energy storage in concentrated solar power (CSP) plants. The working principle is simple: the storage medium absorbs heat during the day and releases it at night to drive a steam turbine. The key figure of merit is **heat capacity (Cp)** — materials with higher Cp store more energy per kilogram per degree. They want to systematically screen hundreds of HEO candidates across a wide temperature range (300 K to 1200 K), but differential scanning calorimetry (DSC) measurements require single-phase samples — synthesis, phase verification by XRD, and the DSC measurement itself takes roughly 3 weeks and $500 per compound.

Your job is to build a fast predictive model for Cp as a function of chemical composition and temperature, so the group can computationally screen 500+ candidate formulas before committing to synthesis. The training data is in `data/cp_data_cleaned.csv` — a collection of ~300 compounds with experimentally measured Cp values at multiple temperatures, giving roughly 4,500 data point rows in total.

The model architecture is a **Random Forest regressor**, with composition encoded as a 100+ dimensional CBFV feature vector and temperature added as a direct numerical feature. This is a high-dimensional, moderate-sample-size problem — exactly the regime where hyperparameter choices (tree depth, minimum leaf size, feature subsampling) matter most and where grid search is computationally wasteful. You will use **Optuna**, a Bayesian hyperparameter optimization framework built on a Tree-structured Parzen Estimator (TPE), to find good hyperparameters in 50 trials instead of exhaustively searching thousands of combinations.

Part 4 closes the loop on the assignment: the TPE sampler inside Optuna is itself a form of Bayesian optimization — it maintains a probabilistic model of which hyperparameter regions have been good so far and uses an acquisition function to decide where to search next. By the end of this part, you will be able to describe exactly how TPE maps onto the GP-UCB framework you built in Part 3.

---

### Task 4.1: Understand the Dataset

Before building any model, understand what the HEO Cp dataset actually contains. The group needs to know how many compounds are in the training pool and how densely each compound is measured across the temperature range — this affects both model complexity and the risk of data leakage in the train/test split.

**Requirements:**
- Load `data/cp_data_cleaned.csv` and print the shape, column names, and summary statistics for Cp and T
- Print the number of unique chemical formulas, the number of unique temperature points, and the average number of temperature measurements per formula
- Answer: if you used a naive random 80/20 train/test split, could test rows leak information from the same compound's training rows? Why is this a problem for evaluating whether your model will generalize to *new* compounds the solar group hasn't measured?

**Deliverables:**
- Summary statistics printed
- Written answer to the data leakage question

---

### Task 4.2: Build Features and a Leakage-Free Train/Test Split

Each row in the dataset is one (formula, temperature, Cp) triplet. The RF needs numerical features, so you will represent each formula as a 100+ dimensional CBFV (composition-based feature vector) using the Oliynyk elemental property set, then append temperature as an additional numerical feature.

The critical step is the train/test split. A naive random split would put rows from the same formula in both train and test — the model could memorize per-formula patterns and achieve artificially low test error without learning to generalize. Instead, use a **group-based split** that keeps all temperature rows for a given formula together, either entirely in train or entirely in test.

**Requirements:**
- Generate CBFV features using `cbfv.composition.generate_features` with `elem_prop='oliynyk'` and `drop_duplicates=False`. Note: CBFV silently skips formulas it cannot parse — check the `skipped` return value and ensure you realign the temperature column to `X_comp.index` before adding it.
- Append temperature as a feature column.
- Perform a group-based 80/20 train/test split using `sklearn.model_selection.GroupShuffleSplit` with formula name as the grouping key and `random_state=88`. Print train and test sizes.
- Answer: what would the test MAE look like if you used a random split instead of a group split? Would it be higher or lower than the group-split MAE, and why?

**Deliverables:**
- Feature matrix X (CBFV + T) and target y (Cp); train/test sizes printed
- Written answer to the split comparison question

---

### Task 4.3: Define the Optuna Objective Function

Optuna searches the hyperparameter space by calling an **objective function** repeatedly. Each call samples a new hyperparameter configuration, trains a model, evaluates it, and returns the metric to minimize (test MAE). Optuna's TPE sampler uses previous results to decide which configurations to try next — exactly the GP-acquisition-function loop from Part 3, but operating in hyperparameter space rather than material design space.

Implement an `objective(trial)` function with an inline comment on each hyperparameter line explaining why that Optuna sampling method is appropriate for that parameter type. The function should:
- Sample five hyperparameters using Optuna's trial API: `n_estimators` (integer, 50–500), `max_depth` (categorical: None, 5, 10, 20, 50), `min_samples_split` (integer, 2–20), `min_samples_leaf` (integer, 1–10), and `max_features` (categorical: "sqrt", "log2", 0.5, 1.0)
- Train a `RandomForestRegressor` with those parameters and `random_state=88` on the training set
- Return the test-set MAE

**Deliverables:**
- `objective` function with per-parameter inline comments

---

### Task 4.4: Run the Study

With your objective function defined, hand it to Optuna and let the TPE sampler run 50 trials. Suppress verbose logging so only the final result is printed.

**Requirements:**
- Create a minimization study and optimize for 50 trials
- Print the best trial number, best MAE, and the full set of best hyperparameters
- Answer: after 50 trials, how many total hyperparameter combinations have been evaluated? If you had used a 5-value grid search over all 5 parameters instead, how many combinations would that require?

**Deliverables:**
- Study completed; best parameters and answer to the grid-search comparison printed

---

### Task 4.5: Evaluate the Best Model

Retrain a fresh `RandomForestRegressor` from scratch using the best hyperparameters found by Optuna, fitted on the full training set (not just one trial's fold). Evaluate on the held-out test formulas.

**Requirements:**
- Retrain with `study.best_params` and `random_state=88`; compute and print test MAE and R²
- Save a parity plot (predicted vs. actual Cp) with the identity line and MAE/R² annotated as `optuna_rf_parity_plot.png`
- Save a plot of MAE vs. trial number showing how Optuna's search improved over time as `optuna_optimization_history.png`. Look up the relevant Optuna visualization function.
- Answer: does the optimization history show a clear improvement trend, or does it plateau quickly? What does this tell you about the difficulty of this hyperparameter landscape?

**Deliverables:**
- MAE and R² printed; `optuna_rf_parity_plot.png` and `optuna_optimization_history.png` saved
- Written answer to the optimization history question

---

### Task 4.6: Situate the Result

The HEO solar group's final question is: "Should we use this RF model, switch to CrabNet, or something else entirely?" Answer the following questions to give them a principled recommendation:

1. Report your RF test-set MAE and R². Recall your CrabNet Cp results from HW3 Task 3.7. The key difference: HW3 CrabNet was trained only at T = 298 K (~243 samples); this RF uses all temperatures (~4,500 rows). Does that difference make the comparison fair? If not, how would you redesign the comparison?

2. Compare the two approaches across five dimensions: model complexity, interpretability, data requirements, physical priors encoded, and ease of adding new compounds to the training set.

3. The solar group wants to screen 500 new HEO compositions across 20 temperatures each (10,000 prediction queries). They have 2 weeks and no GPU. Which model do you recommend, and why?

4. Optuna's TPE sampler is itself a form of Bayesian optimization operating in hyperparameter space. Drawing directly on the vocabulary from Part 3: what plays the role of the "surrogate model" in TPE? What plays the role of the "acquisition function"? How does TPE differ structurally from the GP-UCB loop you implemented — specifically, what does TPE use instead of a GP, and what does it use as the objective to maximize instead of UCB?

**Deliverables:**
- Written answers to all four questions

---

## Common Mistakes and Tips

### Part 1: Bayesian Inference and Information Theory
- **Mistake:** Confusing the posterior mean with the MLE. The posterior mean is pulled toward the prior mean — this shrinkage is intentional and the whole point.
- **Mistake:** Setting a very tight prior and then treating the result as data-driven. Informative priors are valid but must be explicitly justified.
- **Mistake (Task 1.4):** Using the formula for Shannon entropy (discrete, in bits) on a continuous distribution without accounting for the differences. Differential entropy can be negative; Shannon entropy of a discrete approximation depends on bin width.
- **Mistake (Task 1.5):** Forgetting that KL divergence requires the support of Q to cover the support of P. If `q(θ) = 0` somewhere that `p(θ) > 0`, KL is infinite. Use a small epsilon offset or ensure both distributions share support before computing numerically.
- **Mistake (Task 1.5):** Treating KL(P‖Q) and KL(Q‖P) as interchangeable. They are not — they have different interpretations and different failure modes.
- **Mistake (Task 1.6):** Computing NLL using the posterior *after* observing the data point being scored. This is data leakage — use the predictive distribution *before* each observation.
- **Mistake (Task 1.7):** Confusing MI with correlation. MI captures any statistical dependence (including nonlinear), while correlation only captures linear relationships. They are equal only for Gaussian distributions.
- **Mistake (Task 1.8):** Confusing the ELBO lower bound direction. The ELBO is always ≤ log p(x), not ≥. Maximizing the ELBO tightens this gap.

### Part 2: Gaussian Processes
- **Mistake:** Not understanding how to sample from a GP prior. Call `gp.sample_y(X_grid, n_samples=5)` on an *unfitted* GPR — do not call `.fit()` first.
- **Mistake:** Treating all uncertainty as epistemic. The `WhiteKernel` noise level is aleatoric — it does not shrink with more data.
- **Mistake:** Assuming a high LML always means the best kernel. LML can have multiple local optima; always use `n_restarts_optimizer ≥ 5`.

### Part 3: Bayesian Optimization
- **Mistake:** Not normalizing features before GP fitting. Always use `StandardScaler` on X before training.
- **Mistake:** Fitting the GP on the full dataset instead of only the observed points.
- **Mistake:** Not resetting the observed/candidate sets to their identical initial state before each campaign run.
- **Mistake:** EI returns all zeros. Check that σ > 0 for at least some candidates and that `f_best = max(y_observed)`.
- **Mistake:** Conflating the surrogate fitting step with the acquisition optimization step — these solve different problems.

### Part 4: Optuna
- **Mistake:** Using a naive random split instead of `GroupShuffleSplit` — this introduces data leakage.
- **Mistake:** Not retraining a fresh model with `study.best_params` after the study finishes.
- **Mistake:** CBFV silently skips some formulas (reported in `skipped`). Drop those rows before aligning the temperature column with `X_comp.index`.

---

## Grading Rubric

**Part 1: Bayesian Inference Foundations (30 points)**
- Task 1.1 (Bayesian updating + statistics): 4 pts
- Task 1.2 (Prior sensitivity): 3 pts
- Task 1.3 (Predictive distribution): 3 pts
- Task 1.4 (Entropy, conditional entropy, joint entropy): 4 pts
- Task 1.5 (KL divergence and f-divergences): 5 pts
- Task 1.6 (Cross-entropy, NLL, proper scoring rules, calibration): 4 pts
- Task 1.7 (Mutual information, conditional MI, EIG): 4 pts
- Task 1.8 (Differential entropy, conditional Gaussian entropy, ELBO): 3 pts

**Part 2: Gaussian Process Fundamentals (20 points)**
- Task 2.1 (GP priors and kernel visualization): 5 pts
- Task 2.2 (Kernel engineering and ARD): 5 pts
- Task 2.3 (Uncertainty decomposition): 5 pts
- Task 2.4 (Log marginal likelihood): 5 pts

**Part 3: Bayesian Optimization by Hand (35 points)**
- Tasks 3.1–3.3 (Setup and GP surrogate with inline comments): 4 pts
- Task 3.4 (UCB + kappa analysis, ucb function with inline comments): 5 pts
- Task 3.5 (EI, expected_improvement function with inline comments): 3 pts
- Task 3.6 (PI + acquisition comparison, probability_of_improvement with inline comments): 5 pts
- Task 3.8 (BO campaigns): 4 pts
- Task 3.7 (Regret analysis + theoretical bounds): 5 pts
- Task 3.9 (Failure modes): 5 pts
- Task 3.10 (Plots + commentary): 4 pts

**Part 4: Optuna (15 points)**
- Tasks 4.1–4.2 (Load + group split): 3 pts
- Task 4.3 (Objective function): 4 pts
- Task 4.4 (Run study): 2 pts
- Task 4.5 (Evaluate model): 3 pts
- Task 4.6 (Comparison + discussion): 3 pts

**Total: 100 points**

**Bonus:**
- Adaptive κ_t campaign in Task 3.7: 3 pts
- Task 1.9 (Fisher information and Cramér-Rao bound): 3 pts

---

## Submission Guidelines

**Important Note on File Naming:** Use the exact filenames listed below. Submissions may be graded by scripts that look for these files. Incorrect filenames may result in lost points.

1. **Code:** Submit `hw4_yourname.py` (preferred) — or `HW4_spring2026.ipynb` if using the notebook
2. **Figures:** All `.png` files listed below
3. **Written answers:** Submit `hw4_yourname_answers.md` (if using a `.py` script) — or use dedicated markdown answer cells in the notebook. All discussion question responses must be in markdown, not code comments or print statements.

**Required files:**

| File | Task |
|---|---|
| `hw4_yourname.py` (or `HW4_spring2026.ipynb`) | — |
| `hw4_yourname_answers.md` (or markdown cells in notebook) | All parts |
| `bayes_posterior_update.png` | 1.1 |
| `bayes_prior_sensitivity.png` | 1.2 |
| `bayes_prior_predictive.png` | 1.3 |
| `bayes_posterior_predictive.png` | 1.3 |
| `entropy_vs_observations.png` | 1.4 |
| `kl_divergence_curve.png` | 1.5 |
| `scoring_rules.png` | 1.6 |
| `mutual_information_curve.png` | 1.7 |
| `eig_vs_sigma.png` | 1.7 |
| `gp_posterior_entropy.png` | 1.8 |
| `fisher_vs_entropy.png` | 1.9 (bonus) |
| `gp_prior_samples.png` | 2.1 |
| `gp_kernel_heatmaps.png` | 2.1 |
| `gp_kernel_comparison.png` | 2.2 |
| `gp_uncertainty_decomposition.png` | 2.3 |
| `gp_hyperparameter_sensitivity.png` | 2.3 |
| `gp_lml_curve.png` | 2.4 |
| `gp_lml_surface.png` | 2.4 |
| `acquisition_comparison.png` | 3.6 |
| `bo_simple_regret.png` | 3.7 |
| `bo_cumulative_regret.png` | 3.7 |
| `gp_nongaussian_noise.png` | 3.9 |
| `bo_discovery_plot.png` | 3.10 |
| `bo_cumulative_discovery.png` | 3.10 |
| `bo_best_value_curve.png` | 3.10 |
| `optuna_rf_parity_plot.png` | 4.5 |
| `optuna_optimization_history.png` | 4.5 |

**Important Notes:**
- Start early — Part 3 takes time to debug, and the BO loops can be tricky to get right
- Use `random_state=88` everywhere
- Use `matplotlib.use('Agg')` at the top of your script if running headlessly so `plt.savefig()` works without a display
- All functions must have inline comments on non-obvious logic — these are graded within each task (see rubric)

**Deadline:** April 9, 2026 at 23:59
