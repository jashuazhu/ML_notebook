# This script generates a 19-page “blog book”:
#  - /mnt/data/regression_book/index.html
#  - 18 model pages (one per regression model)
# It copies the provided styles.css and embeds MathJax for LaTeX.
#
# Each model page includes:
#   1) Background/overview with formulas and a short derivation outline
#   2) “Best for” guidance
#   3) An expandable <details> code block with a runnable Python example
#      (simulated data with white noise; creates a plot)
#   4) A figure placeholder with explanatory text (original data, fit, errors)
#
# You can download the resulting zip at the end of this cell’s output.

import os, textwrap, json, zipfile, shutil, re, sys, math, random, pathlib

root = "/mnt/data/regression_book"
assets_dir = os.path.join(root, "assets")
os.makedirs(assets_dir, exist_ok=True)

# Copy styles.css (provided by the user) into the book folder for self-containment
src_css = "/mnt/data/styles.css"
dst_css = os.path.join(root, "styles.css")
if os.path.exists(src_css):
    shutil.copyfile(src_css, dst_css)

# MathJax (v3) header snippet
MATHJAX = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]'], ['$$','$$']],
    processEscapes: true,
    tags: 'ams'
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
"""

# Navigation builder
def chapter_nav(i, n, index_href="index.html", slug_list=None):
    prev_html = (
        f'<a href="{slug_list[i-1][1]}">&larr; Prev</a>'
        if i > 0 else '<span class="disabled" aria-disabled="true">&larr; Prev</span>'
    )
    next_html = (
        f'<a href="{slug_list[i+1][1]}">Next &rarr;</a>'
        if i < n-1 else '<span class="disabled" aria-disabled="true">Next &rarr;</span>'
    )
    return f'''
    <nav class="chapter-nav top">
      {prev_html}
      <a class="index-link" href="{index_href}">Index</a>
      {next_html}
    </nav>
    '''

# Model definitions (title, slug, blurb, math outline, best_for bullets, code template hints)
models = [
    # 1
    dict(
        title="Linear Regression",
        slug="model-01-linear-regression.html",
        blurb=(
            "Linear regression models a continuous response as a linear combination of features. "
            "It assumes additive, linear effects and Gaussian noise. The solution minimizes squared errors "
            "and is closed-form when the design matrix has full rank."
        ),
        math=(
            "Given design matrix $X\\in\\mathbb{R}^{n\\times p}$ and response $y\\in\\mathbb{R}^n$, "
            "estimate $\\hat{\\beta}$ by Ordinary Least Squares (OLS): "
            "$$\\hat{\\beta}=\\arg\\min_{\\beta}\\; \\|y - X\\beta\\|_2^2.$$ "
            "Setting the gradient to zero yields the normal equations "
            "$$X^\\top X\\,\\hat{\\beta}=X^\\top y,$$ "
            "so if $X^\\top X$ is invertible, $\\hat{\\beta}=(X^\\top X)^{-1}X^\\top y$."
        ),
        best=[
            "Continuous target with approximately linear relationships",
            "Need interpretable coefficients and effect sizes",
            "Errors roughly homoskedastic and Gaussian"
        ],
        code_kind="sklearn_linear"
    ),
    # 2
    dict(
        title="Multiple Linear Regression",
        slug="model-02-multiple-linear.html",
        blurb=(
            "Multiple linear regression extends OLS to multiple predictors, allowing control of confounders "
            "and estimation of partial effects while keeping interpretability."
        ),
        math=(
            "Identical OLS criterion as linear regression but with $p>1$ predictors: "
            "$$\\hat{\\beta}=\\arg\\min_{\\beta}\\; (y - X\\beta)^\\top(y - X\\beta).$$ "
            "Inference commonly uses $t$-tests for individual coefficients and $F$-tests for groups."
        ),
        best=[
            "Continuous targets with several covariates",
            "Interest in adjusted (partial) effects",
            "Diagnostics for multicollinearity and residual patterns"
        ],
        code_kind="sklearn_linear_multi"
    ),
    # 3
    dict(
        title="Lasso Regression (L1)",
        slug="model-03-lasso.html",
        blurb=(
            "Lasso adds an $\\ell_1$ penalty to induce sparsity, performing embedded feature selection. "
            "Useful when $p\\gg n$ or many predictors are irrelevant."
        ),
        math=(
            "Solve the penalized problem "
            "$$\\hat{\\beta}=\\arg\\min_{\\beta}\\; \\tfrac12\\|y - X\\beta\\|_2^2 + \\lambda\\|\\beta\\|_1,$$ "
            "whose subgradient optimality (KKT) conditions yield exact zeros for some coefficients."
        ),
        best=[
            "High-dimensional data; need for feature selection",
            "When interpretability via sparsity is desired",
            "Handles multicollinearity better than plain OLS"
        ],
        code_kind="sklearn_lasso"
    ),
    # 4
    dict(
        title="Cox Proportional Hazards (Survival)",
        slug="model-04-cox.html",
        blurb=(
            "The Cox model regresses time-to-event outcomes on covariates without specifying the baseline hazard. "
            "It leverages the partial likelihood and assumes proportional hazards."
        ),
        math=(
            "Hazard: $h(t\\mid x)=h_0(t)\\exp(x^\\top\\beta)$. "
            "Estimate $\\beta$ by maximizing the partial likelihood "
            "$$L(\\beta)=\\prod_{i\\in E} \\frac{\\exp(x_i^\\top\\beta)}{\\sum_{j\\in R_i} \\exp(x_j^\\top\\beta)},$$ "
            "where $E$ are event times and $R_i$ are risk sets. The baseline $h_0(t)$ is left unspecified."
        ),
        best=[
            "Right-censored survival data",
            "Interest in hazard ratios and covariate effects",
            "Proportional hazards assumption approximately holds"
        ],
        code_kind="lifelines_cox"
    ),
    # 5
    dict(
        title="Ridge Regression (L2)",
        slug="model-05-ridge.html",
        blurb=(
            "Ridge adds an $\\ell_2$ penalty to shrink coefficients and stabilize estimates under multicollinearity. "
            "Coefficients rarely become exactly zero."
        ),
        math=(
            "Solve $$\\hat{\\beta}=\\arg\\min_{\\beta}\\; \\|y - X\\beta\\|_2^2 + \\lambda\\|\\beta\\|_2^2,$$ "
            "leading to closed-form $(X^\\top X + \\lambda I)\\hat{\\beta}=X^\\top y$."
        ),
        best=[
            "Many correlated predictors",
            "Prediction focus with reduced variance",
            "Avoid dropping variables entirely"
        ],
        code_kind="sklearn_ridge"
    ),
    # 6
    dict(
        title="Stepwise Regression",
        slug="model-06-stepwise.html",
        blurb=(
            "Stepwise (forward/backward) iteratively adds/removes variables using criteria (AIC/BIC) to select a subset model. "
            "Fast and pragmatic, but can be unstable and overfit without care."
        ),
        math=(
            "At each step, choose the move that best improves an information criterion, e.g. "
            "$\\mathrm{AIC}=2k-2\\log\\hat{L}$ or $\\mathrm{BIC}=k\\log n - 2\\log\\hat{L}$, "
            "where $k$ is the number of parameters and $\\hat{L}$ the maximized likelihood."
        ),
        best=[
            "Quick variable screening from many candidates",
            "Baseline model before regularization methods",
            "When interpretability and parsimony matter"
        ],
        code_kind="statsmodels_stepwise"
    ),
    # 7
    dict(
        title="Logistic Regression (Binary)",
        slug="model-07-logistic.html",
        blurb=(
            "Logistic regression models a binary outcome via the log-odds (logit) link. "
            "Parameters are estimated by maximizing the Bernoulli likelihood."
        ),
        math=(
            "For $y\\in\\{0,1\\}$, $\\Pr(y=1\\mid x)=\\sigma(x^\\top\\beta)$ with $\\sigma(z)=1/(1+e^{-z})$. "
            "Log-likelihood: $$\\ell(\\beta)=\\sum_i y_i x_i^\\top\\beta - \\sum_i \\log\\big(1+e^{x_i^\\top\\beta}\\big).$$ "
            "Gradient ascent/Newton methods yield the MLE."
        ),
        best=[
            "Disease yes/no, response vs non-response",
            "Probability estimation with interpretable odds ratios",
            "Baseline linear classifier"
        ],
        code_kind="sklearn_logistic"
    ),
    # 8
    dict(
        title="Elastic Net (L1 + L2)",
        slug="model-08-elastic-net.html",
        blurb=(
            "Elastic Net blends Lasso and Ridge to select groups of correlated predictors and stabilize selection."
        ),
        math=(
            "Solve $$\\hat{\\beta}=\\arg\\min_{\\beta}\\; \\tfrac12\\|y - X\\beta\\|_2^2 "
            "+ \\lambda\\big(\\alpha\\|\\beta\\|_1 + \\tfrac{1-\\alpha}{2}\\|\\beta\\|_2^2\\big),\\; 0<\\alpha<1.$$"
        ),
        best=[
            "Correlated features where pure Lasso is unstable",
            "Feature selection plus shrinkage",
            "High-dimensional omics/finance tabular data"
        ],
        code_kind="sklearn_elastic"
    ),
    # 9
    dict(
        title="Polynomial Regression",
        slug="model-09-polynomial.html",
        blurb=(
            "Polynomial regression uses basis expansion (powers of x) but remains linear in parameters. "
            "Useful for smooth nonlinearities."
        ),
        math=(
            "Model $y\\approx\\sum_{k=0}^{d} \\beta_k x^k$ and estimate $\\beta$ by OLS on the expanded design matrix. "
            "Degree $d$ controls bias–variance; use cross-validation."
        ),
        best=[
            "Single/few features with smooth curvature",
            "Need simple nonlinear fit with interpretability",
            "Low to moderate degrees (avoid overfitting)"
        ],
        code_kind="sklearn_poly"
    ),
    # 10
    dict(
        title="Quantile Regression",
        slug="model-10-quantile.html",
        blurb=(
            "Quantile regression models conditional quantiles (median, tails) via the pinball loss. "
            "It captures distributional heterogeneity beyond the mean."
        ),
        math=(
            "For quantile $\\tau\\in(0,1)$, minimize the check loss "
            "$$\\hat{\\beta}=\\arg\\min_{\\beta}\\sum_i \\rho_\\tau(y_i - x_i^\\top\\beta)$$ "
            "with $\\rho_\\tau(u)=u(\\tau-\\mathbf{1}\\{u<0\\})$."
        ),
        best=[
            "Heteroskedastic data; interest in tails/median",
            "Robustness to outliers vs least squares",
            "Fairness/distributional effects analysis"
        ],
        code_kind="statsmodels_quantile"
    ),
    # 11
    dict(
        title="Decision Tree Regression",
        slug="model-11-tree.html",
        blurb=(
            "CART trees split the feature space into regions minimizing squared error. "
            "They capture interactions and nonlinearities with simple rules."
        ),
        math=(
            "At each node, choose split $(j,s)$ that maximizes reduction in SSE: "
            "$\\Delta=\\sum_{i\\in R} (y_i-\\bar y)^2 - \\sum_{i\\in R_L}(y_i-\\bar y_L)^2 - \\sum_{i\\in R_R}(y_i-\\bar y_R)^2$."
        ),
        best=[
            "Interpretability of splits",
            "Nonlinear, interaction-rich relationships",
            "Baseline for ensemble methods"
        ],
        code_kind="sklearn_tree"
    ),
    # 12
    dict(
        title="Random Forest Regression",
        slug="model-12-rf.html",
        blurb=(
            "Random Forest averages many bootstrapped trees with feature subsampling, reducing variance and overfitting."
        ),
        math=(
            "Bagging reduces variance by averaging. The expected MSE decreases with more weakly correlated trees; "
            "feature subsampling (mtry) decorrelates trees."
        ),
        best=[
            "Strong baseline on tabular data",
            "Handles missingness/outliers reasonably",
            "Variable importance and partial dependence"
        ],
        code_kind="sklearn_rf"
    ),
    # 13
    dict(
        title="Gradient Boosting Regression",
        slug="model-13-gbm.html",
        blurb=(
            "Boosting builds an additive model by fitting weak learners to current residuals/negative gradients. "
            "Powerful for complex tabular data."
        ),
        math=(
            "Stagewise additive modeling: initialize $F_0$. For $m=1..M$: fit $h_m$ to negative gradients and update "
            "$F_m(x)=F_{m-1}(x)+\\nu\\,h_m(x)$ with learning rate $\\nu$."
        ),
        best=[
            "High accuracy with careful tuning",
            "Captures nonlinearities and interactions",
            "Supports custom loss functions"
        ],
        code_kind="sklearn_gbm"
    ),
    # 14
    dict(
        title="Support Vector Regression (SVR)",
        slug="model-14-svr.html",
        blurb=(
            "SVR fits a function that deviates from observations by at most $\\varepsilon$ using the flattest function, "
            "with kernels for nonlinearity."
        ),
        math=(
            "Primal: minimize $\\tfrac12\\|w\\|^2 + C\\sum_i(\\xi_i+\\xi_i^*)$ subject to "
            "$|y_i - w^\\top\\phi(x_i) - b|\\le \\varepsilon + \\xi_i$ and $\\xi_i,\\xi_i^*\\ge 0$."
        ),
        best=[
            "High-dimensional features",
            "Robust to outliers via $\\varepsilon$-insensitive loss",
            "Kernelized nonlinear regression"
        ],
        code_kind="sklearn_svr"
    ),
    # 15
    dict(
        title="XGBoost Regression",
        slug="model-15-xgb.html",
        blurb=(
            "XGBoost is an optimized gradient boosting library with regularization, shrinkage, and second-order updates."
        ),
        math=(
            "Objective $\\mathcal{L}=\\sum_i \\ell(y_i, \\hat y_i) + \\sum_k \\Omega(f_k)$, "
            "with $\\Omega(f)=\\gamma T + \\tfrac12\\lambda\\|w\\|_2^2$. "
            "Uses Taylor expansion to compute splits efficiently."
        ),
        best=[
            "Large/tabular data with complex structure",
            "State-of-the-art accuracy with proper tuning",
            "Handles missing values natively"
        ],
        code_kind="xgboost"
    ),
    # 16
    dict(
        title="LightGBM Regression",
        slug="model-16-lightgbm.html",
        blurb=(
            "LightGBM uses histogram-based splitting and leaf-wise growth with depth limits for speed and accuracy."
        ),
        math=(
            "Approximates continuous features by bins to speed up split finding; "
            "leaf-wise tree growth reduces loss faster but needs regularization."
        ),
        best=[
            "Very large datasets",
            "High-cardinality features and imbalance",
            "Fast training with good accuracy"
        ],
        code_kind="lightgbm"
    ),
    # 17
    dict(
        title="Neural Network Regression (MLP)",
        slug="model-17-nn.html",
        blurb=(
            "Feed-forward neural networks approximate complex functions via layered nonlinear transformations, "
            "trained by backpropagation."
        ),
        math=(
            "Minimize $\\sum_i \\tfrac12(y_i - f_\\theta(x_i))^2$ where $f_\\theta$ composes affine maps and activations. "
            "Backpropagation applies the chain rule to compute gradients."
        ),
        best=[
            "Highly nonlinear relationships",
            "Large datasets and feature interactions",
            "When feature engineering is hard"
        ],
        code_kind="sklearn_mlp"
    ),
    # 18
    dict(
        title="K-Nearest Neighbors Regression",
        slug="model-18-knn.html",
        blurb=(
            "KNN predicts by averaging the target of the K closest points (optionally distance-weighted). "
            "Nonparametric and simple."
        ),
        math=(
            "Prediction: $\\hat y(x)=\\frac{\\sum_{i\\in N_K(x)} w_i y_i}{\\sum_{i\\in N_K(x)} w_i}$, "
            "with $w_i=1$ (uniform) or $1/(d(x,x_i)+\\epsilon)$ (distance)."
        ),
        best=[
            "Local structure matters (spatial/temporal)",
            "Low-dimensional feature spaces",
            "Baseline nonparametric smoother"
        ],
        code_kind="sklearn_knn"
    )
]

# Helper: escape HTML for code blocks
import html
def codeblock(s):
    return "<pre class=\"ex\"><code>" + html.escape(s) + "</code></pre>"

# Code templates for different model kinds
def py_header_common():
    return textwrap.dedent("""
    # Requirements: numpy, matplotlib, scikit-learn (or statsmodels/lifelines as noted).
    # This example simulates data with white noise, fits the model, and plots:
    #   - original data
    #   - fitted curve/predictions
    #   - residuals (errors)
    import numpy as np
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(42)
    """).strip()

def py_footer_plot(file_stub, is_classifier=False):
    # Produces 2-panel plot for regression, simple plot for classifier
    if not is_classifier:
        return textwrap.dedent(f"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].scatter(x, y, s=12, label='Original data')
        axes[0].plot(x_grid, y_pred_grid, linewidth=2, label='Fitted')
        axes[0].set_title('Fit')
        axes[0].legend()

        axes[1].scatter(x, residuals, s=10)
        axes[1].axhline(0, linestyle='--')
        axes[1].set_title('Residuals')

        plt.tight_layout()
        plt.savefig('assets/{file_stub}.png', dpi=140)
        plt.show()
        """).strip()
    else:
        return textwrap.dedent(f"""
        plt.figure(figsize=(6,4))
        plt.scatter(x, y, s=12, label='Class (0/1)', c=y, cmap='coolwarm')
        plt.plot(x_grid, proba_grid, linewidth=2, label='P(y=1|x)')
        plt.title('Logistic fit')
        plt.legend()
        plt.tight_layout()
        plt.savefig('assets/{file_stub}.png', dpi=140)
        plt.show()
        """).strip()

def make_code(kind, file_stub):
    H = py_header_common()

    if kind in ("sklearn_linear", "sklearn_linear_multi"):
        code = textwrap.dedent(f"""
        from sklearn.linear_model import LinearRegression

        # Simulate
        n = 200
        x = rng.uniform(-3, 3, size=(n, 1))
        if "{kind}" == "sklearn_linear_multi":
            z = rng.normal(size=(n, 2))
            X = np.hstack([x, z])  # 3 predictors
            true_beta = np.array([2.0, -1.0, 0.8, 0.0])  # intercept + 3 betas
            y = true_beta[0] + X @ true_beta[1:] + rng.normal(scale=0.8, size=n)
            model = LinearRegression().fit(X, y)
            x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
            # keep z at mean 0 for plotting along x only
            Xg = np.hstack([x_grid, np.zeros((len(x_grid), 2))])
            y_pred_grid = model.predict(Xg)
        else:
            true_beta0, true_beta1 = 1.5, 2.2
            y = true_beta0 + true_beta1 * x[:,0] + rng.normal(scale=1.0, size=n)
            model = LinearRegression().fit(x, y)
            x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
            y_pred_grid = model.predict(x_grid)

        y_pred = model.predict(X if "{kind}" == "sklearn_linear_multi" else x)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_lasso":
        code = textwrap.dedent("""
        from sklearn.linear_model import Lasso

        n, p = 150, 20
        X = rng.normal(size=(n, p))
        beta_true = np.zeros(p); beta_true[:4] = [2.0, -1.0, 0.0, 1.5]
        y = 1.0 + X @ beta_true + rng.normal(scale=1.0, size=n)

        model = Lasso(alpha=0.15, random_state=0).fit(X, y)
        x = X[:, [0]]  # for plotting against one feature
        x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
        # Partial effect plot: vary feature 0, keep others at mean 0
        Xg = np.zeros((len(x_grid), p)); Xg[:,0] = x_grid[:,0]
        y_pred_grid = model.predict(Xg)

        y_pred = model.predict(X)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "lifelines_cox":
        code = textwrap.dedent("""
        # pip install lifelines
        import pandas as pd
        from lifelines import CoxPHFitter

        n = 400
        x = rng.normal(size=n)
        # Simulate log-hazard ratio beta=0.8
        beta = 0.8
        # Weibull baseline via inverse-CDF sampling
        u = rng.uniform(size=n)
        k, lam = 1.5, 2.0
        T0 = ( -np.log(u) )**(1/k) / lam
        # Apply PH effect: hazard scaling by exp(beta*x) -> time scaling by /exp(beta*x)
        T = T0 / np.exp(beta * x)
        # Censoring
        C = rng.uniform(0.1, 3.0, size=n)
        event = (T <= C).astype(int)
        time = np.minimum(T, C)

        df = pd.DataFrame({'time': time, 'event': event, 'x': x})
        cph = CoxPHFitter().fit(df, duration_col='time', event_col='event')
        # No simple "prediction curve" vs x; we can plot partial effects
        x_grid = np.linspace(x.min(), x.max(), 200)
        # Risk score ~ exp(beta_hat * x)
        beta_hat = cph.params_['x']
        y_pred_grid = np.exp(beta_hat * x_grid)  # relative risk
        y = event  # for visualization only
        residuals = y - (y.mean())  # placeholder residual-like view
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_ridge":
        code = textwrap.dedent("""
        from sklearn.linear_model import Ridge

        n, p = 200, 30
        X = rng.normal(size=(n, p))
        # Correlated columns
        X[:,1] = X[:,0] + rng.normal(scale=0.2, size=n)
        X[:,2] = X[:,0] - X[:,1] + rng.normal(scale=0.2, size=n)

        beta_true = rng.normal(scale=0.5, size=p)
        y = 0.7 + X @ beta_true + rng.normal(scale=1.0, size=n)

        model = Ridge(alpha=3.0, random_state=0).fit(X, y)
        x = X[:, [0]]
        x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
        Xg = np.zeros((len(x_grid), p)); Xg[:,0] = x_grid[:,0]
        y_pred_grid = model.predict(Xg)

        y_pred = model.predict(X)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "statsmodels_stepwise":
        code = textwrap.dedent("""
        # pip install statsmodels
        import numpy as np
        import statsmodels.api as sm

        def forward_stepwise(X, y, feature_names, criterion='aic'):
            selected = []
            remaining = list(range(X.shape[1]))
            current_score = np.inf
            while remaining:
                scores = []
                for j in remaining:
                    cand = selected + [j]
                    Xc = sm.add_constant(X[:, cand])
                    model = sm.OLS(y, Xc).fit()
                    score = model.aic if criterion=='aic' else model.bic
                    scores.append((score, j, model))
                scores.sort(key=lambda t: t[0])
                best_score, best_j, best_model = scores[0]
                if best_score < current_score - 1e-6:
                    current_score = best_score
                    selected.append(best_j)
                    remaining.remove(best_j)
                else:
                    break
            return selected, best_model

        rng = np.random.default_rng(0)
        n, p = 250, 12
        X = rng.normal(size=(n, p))
        beta = np.zeros(p); beta[[1,4,7]] = [2.0, -1.5, 0.9]
        y = 1.0 + X @ beta + rng.normal(scale=1.2, size=n)

        names = [f'x{j}' for j in range(p)]
        sel, model = forward_stepwise(X, y, names, criterion='aic')

        x = X[:, [sel[0]]] if sel else X[:, [0]]
        x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
        # Build grid for plotting varying first selected feature only
        Xg = np.zeros((len(x_grid), X.shape[1]))
        Xg[:, sel[0] if sel else 0] = x_grid[:,0]
        y_pred_grid = model.predict(sm.add_constant(Xg))

        y_pred = model.predict(sm.add_constant(X))
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_logistic":
        code = textwrap.dedent("""
        from sklearn.linear_model import LogisticRegression

        n = 300
        x = rng.uniform(-3, 3, size=(n,1))
        beta0, beta1 = -0.5, 1.2
        z = beta0 + beta1 * x[:,0] + rng.normal(scale=0.8, size=n)
        p = 1/(1+np.exp(-z))
        y = (rng.uniform(size=n) < p).astype(int)

        model = LogisticRegression().fit(x, y)
        x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
        proba_grid = model.predict_proba(x_grid)[:,1]

        # For visualization, residuals are less standard; skip residual plot.
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub, is_classifier=True)

    if kind == "sklearn_elastic":
        code = textwrap.dedent("""
        from sklearn.linear_model import ElasticNet

        n, p = 180, 25
        X = rng.normal(size=(n, p))
        beta_true = np.zeros(p); beta_true[:6] = [1.8, 0.0, -1.4, 0.9, 0.0, 0.7]
        y = -0.3 + X @ beta_true + rng.normal(scale=1.0, size=n)

        model = ElasticNet(alpha=0.2, l1_ratio=0.6, random_state=0).fit(X, y)
        x = X[:, [0]]
        x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
        Xg = np.zeros((len(x_grid), p)); Xg[:,0] = x_grid[:,0]
        y_pred_grid = model.predict(Xg)

        y_pred = model.predict(X)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_poly":
        code = textwrap.dedent("""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        n = 200
        x = rng.uniform(-2.5, 2.5, size=(n,1))
        y_true = 1.0 - 0.8*x[:,0] + 0.7*x[:,0]**2 - 0.2*x[:,0]**3
        y = y_true + rng.normal(scale=0.7, size=n)

        model = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('lin', LinearRegression())
        ]).fit(x, y)

        x_grid = np.linspace(x.min(), x.max(), 200).reshape(-1,1)
        y_pred_grid = model.predict(x_grid)
        y_pred = model.predict(x)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "statsmodels_quantile":
        code = textwrap.dedent("""
        # pip install statsmodels
        import statsmodels.api as sm
        n = 250
        x = rng.uniform(-3, 3, size=n)
        y = 0.5 + 1.2*x + 0.6*x*(x>0) + rng.normal(scale=1.0+0.5*np.abs(x), size=n)

        X = sm.add_constant(x)
        mod = sm.quantreg('y ~ x', dict(y=y, x=x))
        res = mod.fit(q=0.5)  # median

        x_grid = np.linspace(x.min(), x.max(), 200)
        Xg = sm.add_constant(x_grid)
        y_pred_grid = res.predict(Xg)

        # For residuals, compare to conditional median
        y_pred = res.predict(X)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_tree":
        code = textwrap.dedent("""
        from sklearn.tree import DecisionTreeRegressor

        n = 250
        x = rng.uniform(-3, 3, size=(n,1))
        y = np.sin(x[:,0]) + 0.3*(x[:,0]>0) + rng.normal(scale=0.4, size=n)

        model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(x, y)
        x_grid = np.linspace(x.min(), x.max(), 400).reshape(-1,1)
        y_pred_grid = model.predict(x_grid)

        y_pred = model.predict(x)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_rf":
        code = textwrap.dedent("""
        from sklearn.ensemble import RandomForestRegressor

        n = 400
        x1 = rng.uniform(-3, 3, size=(n,1))
        x2 = rng.uniform(-3, 3, size=(n,1))
        X = np.hstack([x1, x2])
        y = np.sin(x1[:,0]) + 0.5*np.cos(1.3*x2[:,0]) + rng.normal(scale=0.5, size=n)

        model = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
        # Partial dependence across x1 with x2 fixed at mean
        x = x1
        x_grid = np.linspace(x.min(), x.max(), 300).reshape(-1,1)
        Xg = np.hstack([x_grid, np.zeros((len(x_grid),1))])
        y_pred_grid = model.predict(Xg)

        y_pred = model.predict(X)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_gbm":
        code = textwrap.dedent("""
        from sklearn.ensemble import GradientBoostingRegressor

        n = 350
        x = rng.uniform(-3, 3, size=(n,1))
        y = np.sin(1.5*x[:,0]) + 0.3*x[:,0] + rng.normal(scale=0.5, size=n)

        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=3, random_state=0).fit(x, y)
        x_grid = np.linspace(x.min(), x.max(), 400).reshape(-1,1)
        y_pred_grid = model.predict(x_grid)

        y_pred = model.predict(x)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_svr":
        code = textwrap.dedent("""
        from sklearn.svm import SVR

        n = 250
        x = rng.uniform(-3, 3, size=(n,1))
        y = np.sin(x[:,0]) + 0.1*x[:,0] + rng.normal(scale=0.4, size=n)

        model = SVR(kernel='rbf', C=5.0, epsilon=0.2, gamma='scale').fit(x, y)
        x_grid = np.linspace(x.min(), x.max(), 400).reshape(-1,1)
        y_pred_grid = model.predict(x_grid)

        y_pred = model.predict(x)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "xgboost":
        code = textwrap.dedent("""
        # pip install xgboost
        import xgboost as xgb

        n = 400
        x = rng.uniform(-3, 3, size=(n,1))
        y = np.sin(1.2*x[:,0]) + 0.5*(x[:,0]>0) + rng.normal(scale=0.5, size=n)

        dtrain = xgb.DMatrix(x, label=y)
        params = dict(objective='reg:squarederror', eta=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)
        model = xgb.train(params, dtrain, num_boost_round=400)

        x_grid = np.linspace(x.min(), x.max(), 400).reshape(-1,1)
        dgrid = xgb.DMatrix(x_grid)
        y_pred_grid = model.predict(dgrid)

        y_pred = model.predict(xgb.DMatrix(x))
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "lightgbm":
        code = textwrap.dedent("""
        # pip install lightgbm
        import lightgbm as lgb

        n = 400
        x = rng.uniform(-3, 3, size=(n,1))
        y = np.cos(1.0*x[:,0]) + 0.3*x[:,0] + rng.normal(scale=0.4, size=n)

        train = lgb.Dataset(x, label=y)
        params = dict(objective='regression', learning_rate=0.05, num_leaves=31, feature_fraction=0.9,
                      bagging_fraction=0.8, bagging_freq=1, verbose=-1)
        model = lgb.train(params, train_set=train, num_boost_round=500)

        x_grid = np.linspace(x.min(), x.max(), 400).reshape(-1,1)
        y_pred_grid = model.predict(x_grid)

        y_pred = model.predict(x)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_mlp":
        code = textwrap.dedent("""
        from sklearn.neural_network import MLPRegressor

        n = 500
        x = rng.uniform(-2.5, 2.5, size=(n,1))
        y = np.sin(2.0*x[:,0]) + 0.2*x[:,0]**2 + rng.normal(scale=0.3, size=n)

        model = MLPRegressor(hidden_layer_sizes=(64,64), activation='relu',
                             learning_rate_init=0.01, alpha=1e-4,
                             max_iter=3000, random_state=0).fit(x, y)
        x_grid = np.linspace(x.min(), x.max(), 400).reshape(-1,1)
        y_pred_grid = model.predict(x_grid)

        y_pred = model.predict(x)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    if kind == "sklearn_knn":
        code = textwrap.dedent("""
        from sklearn.neighbors import KNeighborsRegressor

        n = 220
        x = rng.uniform(-3, 3, size=(n,1))
        y = np.sin(1.2*x[:,0]) + 0.4*np.cos(0.8*x[:,0]) + rng.normal(scale=0.5, size=n)

        model = KNeighborsRegressor(n_neighbors=9, weights='distance').fit(x, y)
        x_grid = np.linspace(x.min(), x.max(), 400).reshape(-1,1)
        y_pred_grid = model.predict(x_grid)

        y_pred = model.predict(x)
        residuals = y - y_pred
        """).strip()
        return H + "\n\n" + code + "\n\n" + py_footer_plot(file_stub)

    # Fallback (shouldn't happen)
    return H + "\n\n# TODO: add model-specific code\n"

# Build per-page HTML
def page_html(i, n, model, slug_list):
    title = model["title"]
    blurb = model["blurb"]
    math = model["math"]
    best = model["best"]
    code = make_code(model["code_kind"], file_stub=f"{i+1:02d}-{slug_list[i][0]}")
    # Create a short derivation sentence tailored if possible
    deriv_extra = ""
    if "Least Squares" in math or "OLS" in math or "normal equations" in math:
        deriv_extra = "Differentiating the OLS objective and setting $\\nabla=0$ gives the normal equations."
    elif "partial likelihood" in blurb or "hazard" in math:
        deriv_extra = "Maximization of the partial likelihood yields $\\hat\\beta$ without estimating $h_0(t)$."
    elif "pinball" in blurb or "check loss" in math:
        deriv_extra = "The piecewise-linear pinball loss leads to linear programming formulations."
    elif "SVR" in title:
        deriv_extra = "Dualization introduces kernel functions $K(x_i,x_j)=\\phi(x_i)^\\top\\phi(x_j)$."
    elif "Elastic Net" in title:
        deriv_extra = "KKT conditions blend soft-thresholding (L1) and shrinkage (L2)."

    best_li = "\n".join([f"<li>{html.escape(b)}</li>" for b in best])
    nav_top = chapter_nav(i, n, slug_list=slug_list)
    nav_bot = nav_top.replace('class="chapter-nav top"', 'class="chapter-nav bottom"')

    figure_name = f"assets/{i+1:02d}-{slug_list[i][0]}.png"
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{i+1}. {html.escape(title)} — Regression Models Handbook</title>
<link rel="stylesheet" href="styles.css" />
{MATHJAX}
</head>
<body>
<main class="book-page">
  <header class="book-header">
    <p class="lead">Regression Models Handbook</p>
    <h1>{i+1}. {html.escape(title)}</h1>
  </header>

  {nav_top}

  <section class="card">
    <h2>Overview & Background</h2>
    <p>{html.escape(blurb)}</p>
  </section>

  <section class="card">
    <h2>Mathematics & Derivation</h2>
    <p class="formula">{math}</p>
    <p class="muted">{html.escape(deriv_extra)}</p>
  </section>

  <section class="card">
    <h2>When to Use (Best For)</h2>
    <ul>
      {best_li}
    </ul>
  </section>

  <section class="card">
    <h2>Example & Code</h2>
    <p>Below we simulate data with white noise, fit the model, and create a plot showing original data, fitted values, and residuals. Run this locally (see comments for required packages).</p>
    <details>
      <summary>Show Python example</summary>
      {codeblock(code)}
    </details>
  </section>

  <section class="card">
    <h2>Figure & Interpretation</h2>
    <figure>
      <img src="{figure_name}" alt="Demo plot for {html.escape(title)} (original data, fitted curve, residuals)" style="width:100%;max-width:100%;" />
      <figcaption>Generated by the example code: Left — original data with fitted curve; Right — residual diagnostics (or probability curve for classifiers).</figcaption>
    </figure>
    <p>The <strong>original data</strong> points are scattered due to added white noise. The <strong>fitted curve</strong> represents model predictions on a dense grid. The <strong>residuals</strong> ($y - \\hat y$) highlight model mismatch; patterns may suggest missing features, nonlinearity, or heteroskedasticity.</p>
  </section>

  {nav_bot}

  <footer class="site-footer">
    <p>&copy; Regression Models Handbook</p>
  </footer>
</main>
</body>
</html>
"""
    return page

# Slug list used for nav and filenames
slug_list = [(re.sub(r'[^a-z0-9]+', '-', m["title"].lower()).strip('-'), m["slug"]) for m in models]

# Write model pages
for i, m in enumerate(models):
    html_txt = page_html(i, len(models), m, slug_list)
    with open(os.path.join(root, m["slug"]), "w", encoding="utf-8") as f:
        f.write(html_txt)

# Build index page
toc_items = []
for i, m in enumerate(models):
    number = i+1
    toc_items.append(f"""
    <li>
      <span class="chapter-number"></span>
      <div class="chapter-entry">
        <a href="{m['slug']}">{number}. {html.escape(m['title'])}</a>
        <p>{html.escape(m['blurb'])}</p>
      </div>
    </li>
    """)

index_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Regression Models Handbook — Index</title>
<link rel="stylesheet" href="styles.css" />
{MATHJAX}
</head>
<body>
<main class="book-page">
  <header class="book-header">
    <p class="lead">Regression Models Handbook</p>
    <h1>Index</h1>
    <p class="muted">A concise, uniform “blog book” of 18 regression models with background, math, code, and figures.</p>
  </header>

  <section class="table-of-contents">
    <ol>
      {''.join(toc_items)}
    </ol>
  </section>

  <footer class="site-footer">
    <p>Style unified via <code>styles.css</code>. Formulas rendered with MathJax.</p>
  </footer>
</main>
</body>
</html>
"""

with open(os.path.join(root, "index.html"), "w", encoding="utf-8") as f:
    f.write(index_html)

# Zip the whole folder for download
zip_path = "/mnt/data/regression_book.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder, _, files in os.walk(root):
        for name in files:
            fp = os.path.join(folder, name)
            arc = os.path.relpath(fp, os.path.dirname(root))
            zf.write(fp, arcname=arc)

print("Generated book at:", root)
print("Main entry:", os.path.join(root, "index.html"))
print("Zip archive:", zip_path)
