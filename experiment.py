import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="AM-GM Inequality")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import scipy.optimize as so
    import altair as alt
    from wigglystuff import TangleSlider

    return TangleSlider, alt, mo, np, so


@app.cell(hide_code=True)
def _():
    thesis_url = (
        "https://github.com/mateuszel/amgm-inequality-experiment/"
        "releases/download/v1.0/szablon_dyplomowy_2025.pdf"
    )

    thesis_link = (
        f'<a href="{thesis_url}" target="_blank" '
        'rel="noopener noreferrer">Download the full thesis</a>'
    )
    return (thesis_link,)


@app.cell(hide_code=True)
def _(mo, thesis_link):
    mo.md(fr"""
    # The AM-GM inequality

    This notebook provides a short introduction to one of the main results of my Bachelor's thesis. The thesis studies the gap between the arithmetic and geometric means of positive random variables.

    For any positive numbers $X_1,\dots,X_n$, the AM-GM inequality gives

    $$
    \frac{{1}}{{n}}\sum_{{i=1}}^n X_i
    \ge
    \left(\prod_{{i=1}}^nX_i\right)^{{1/n}}.
    $$

    Therefore,

    $$
    \mathbb E\left[\frac{{1}}{{n}}\sum_{{i=1}}^nX_i\right]
    \ge
    \mathbb E\left[\left(\prod_{{i=1}}^nX_i\right)^{{1/n}}\right].
    $$

    We are interested in how large this gap must be if some information about the underlying distribution is known. Throughout the notebook, we assume that $X_1,\dots,X_n$ are independent and identically distributed copies of a positive random variable $X$.

    If $X$ is constant almost surely, equality holds, so no strictly positive lower bound can be expected without some form of non-degeneracy. As basic information about the distribution, suppose that its mean

    $$
    \mu \coloneqq \mathbb E[X]
    $$

    is known.

    The geometric mean is naturally connected to the logarithmic moment

    $$
    \mu_\ell \coloneqq \mathbb E[\log X],
    $$

    since

    $$
    \log\left(\left(\prod_{{i=1}}^nX_i\right)^{{1/n}}\right)
    =\frac1n\sum_{{i=1}}^n\log X_i.
    $$

    It is therefore natural to ask whether fixing both $\mu$ and $\mu_\ell$ forces a non-trivial gap between the expected arithmetic and geometric means.

    Perhaps surprisingly, it does not. One can construct a sequence of non-degenerate two-point distributions with the same prescribed values of $\mu$ and $\mu_\ell$ for which the expected AM-GM gap becomes arbitrarily small.

    The mechanism behind this construction is the appearance of a very small probability mass located extremely close to zero. As this mass tends to zero, the lower support point must approach zero very rapidly in order to preserve the prescribed logarithmic moment. As a consequence, negative moments of the form

    $$
    \mathbb E[X^{{-q}}]
    $$

    become unbounded.

    This motivates imposing the additional constraint

    $$
    \mathbb E[X^{{-q}}] \le M,
    $$

    for some $q>0$ and finite $M$. This assumption prevents the above degeneration and makes it possible to obtain a strictly positive lower bound on the expected AM-GM gap. This is one of the original results of the thesis.

    Moreover, the extremal distribution can be taken to have only two support points. Writing

    $$
    P(X=a)=p,
    \qquad
    P(X=b)=1-p,
    $$

    the mean constraint gives

    $$
    p=\frac{{b-\mu}}{{b-a}}.
    $$

    The remaining constraints reduce the problem to solving the system

    $$
    \frac{{b-\mu}}{{b-a}}\log a
    +\frac{{\mu-a}}{{b-a}}\log b
    =\mu_\ell,
    $$

    $$
    \frac{{b-\mu}}{{b-a}}a^{{-q}}
    +\frac{{\mu-a}}{{b-a}}b^{{-q}}
    =M.
    $$

    Thus, an infinite-dimensional optimization problem over probability distributions is reduced to solving two nonlinear equations for the support points $a$ and $b$.

    <div style="padding: 1.2rem 1.4rem; margin: 1.5rem 0; border: 1px solid #d9dee8; border-radius: 12px; background: #f8fafc;">
    <strong>Want to dive deeper?</strong><br><br>
    {thesis_link}
    </div>

    The purpose of this notebook is to solve this system numerically and visualize how the resulting lower bound on the expected AM-GM gap changes with the negative-moment constraint $M$.

    Without the loss of generality, we will additionally assume that $\mathbb{{E}}\left[X\right]=1$.
    """)
    return


@app.cell(hide_code=True)
def _(TangleSlider, mo):
    mu_ell_tangle = mo.ui.anywidget(
        TangleSlider(
            amount=-1,
            min_value=-5,
            max_value=0,
            step=0.01,
            digits=2,
        )
    )

    n = mo.ui.anywidget(
        TangleSlider(
            amount=10,
            min_value=2,
            max_value=30,
            step=1,
            digits=0,
        )
    )

    q = mo.ui.anywidget(
        TangleSlider(
            amount=1,
            min_value=0.01,
            max_value=5,
            step=0.01,
            digits=2,
        )
    )
    return mu_ell_tangle, n, q


@app.cell(hide_code=True)
def _(mo, mu_ell_tangle, n):
    mo.md(fr"""
    First, let's show why the additional assumption about the negative moment matters.

    Choose the logarithmic moment $\mu_\ell$ = {mu_ell_tangle} and the number of variables $n$ = {n}.

    For the selected value $\mu_\ell = {mu_ell_tangle.amount:.2f}$, the two-point construction is defined for each $p \in (0,1)$ by

    $$
    b_p = \frac{{1 - p a_p}}{{1-p}},
    $$

    where $a_p \in (0,1)$ is the solution of

    $$
    p\log a_p + (1-p)\log\left(\frac{{1-p a_p}}{{1-p}}\right)
    = {mu_ell_tangle.amount:.2f}.
    $$
    Notice that this construction yields $X_p$ that satisfies both $\mathbb{{E}}[\log X]={mu_ell_tangle.amount:.2f}$ and $\mathbb{{E}}[X] = 1$. 
    Below we can see the behavior of the sequence as $p$ approaches 0.
    """)
    return


@app.cell(hide_code=True)
def _(mu_ell_tangle, n, np, so):
    def construct_two_point_sequence(mu_ell_value, p_values):
        _rows = []
        for _p in p_values:
            def _equation(_log_a):
                _a = np.exp(_log_a)
                _b = (1 - _p * _a) / (1 - _p)
                return _p * _log_a + (1 - _p) * np.log(_b) - mu_ell_value
            _lower = min(-1000.0, 2.0 * mu_ell_value / _p)
            _log_a = so.brentq(_equation, _lower, -1e-12)
            _a = np.exp(_log_a)
            _b = (1 - _p * _a) / (1 - _p)
            _rows.append({"p": _p, "a": max(_a, np.finfo(float).tiny), "b": _b, "log_a": _log_a})
        return _rows

    def enrich_sequence(_rows, _mu_ell_value, _n_value):
        for _row in _rows:
            _row["mean"] = _row["p"] * _row["a"] + (1 - _row["p"]) * _row["b"]
            _row["log_moment"] = _row["p"] * _row["log_a"] + (1 - _row["p"]) * np.log(_row["b"])
            _row["am_gm_gap"] = 1 - (
                _row["p"] * _row["a"] ** (1 / _n_value)
                + (1 - _row["p"]) * _row["b"] ** (1 / _n_value)
            ) ** _n_value
        return _rows

    p_min = 0.0001
    support_p_grid = np.geomspace(0.01, 0.5, 100)
    gap_p_grid = np.geomspace(p_min, 0.5, 120)
    mu_ell_value = float(mu_ell_tangle.amount)
    n_value = int(n.amount)
    support_sequence_rows = enrich_sequence(construct_two_point_sequence(mu_ell_value, support_p_grid), mu_ell_value, n_value)
    gap_sequence_rows = enrich_sequence(construct_two_point_sequence(mu_ell_value, gap_p_grid), mu_ell_value, n_value)
    sequence_rows = gap_sequence_rows
    support_rows = [
        {"p": _row["p"], "support_point": _point, "value": _row[_point]}
        for _row in support_sequence_rows
        for _point in ["a", "b"]
    ]
    gap_rows = [{"p": _row["p"], "gap": _row["am_gm_gap"]} for _row in gap_sequence_rows]
    return gap_rows, gap_sequence_rows, n_value, p_min, support_rows


@app.cell(hide_code=True)
def _(alt, gap_rows, mo, n_value, p_min, support_rows):
    a_chart = (
        alt.Chart({"values": support_rows})
        .transform_filter((alt.datum.support_point == "a") & (alt.datum.p >= 0.01))
        .mark_line(color="#2563eb")
        .encode(
            x=alt.X("p:Q", title="Probability p", scale=alt.Scale(type="log", domain=[0.01, 0.5])),
            y=alt.Y("value:Q", title="a_p", scale=alt.Scale(type="log")),
            tooltip=["p:Q", "value:Q"],
        )
        .properties(title="Lower support point a_p", width=360, height=300)
    )

    b_chart = (
        alt.Chart({"values": support_rows})
        .transform_filter((alt.datum.support_point == "b") & (alt.datum.p >= 0.01))
        .mark_line(color="#dc2626")
        .encode(
            x=alt.X("p:Q", title="Probability p", scale=alt.Scale(type="log", domain=[0.01, 0.5])),
            y=alt.Y("value:Q", title="b_p", scale=alt.Scale(zero=False)),
            tooltip=["p:Q", "value:Q"],
        )
        .properties(title="Upper support point b_p", width=360, height=300)
    )

    gap_chart = (
        alt.Chart({"values": gap_rows})
        .mark_line(color="#c2410c")
        .encode(
            x=alt.X("p:Q", title="Probability p", scale=alt.Scale(type="log", domain=[p_min, 0.5])),
            y=alt.Y("gap:Q", title="Expected AM-GM gap"),
            tooltip=["p:Q", "gap:Q"],
        )
        .properties(title=f"Expected AM-GM gap for n = {n_value}", width="container", height=300)
    )

    mo.vstack([
        mo.hstack([a_chart, b_chart], widths="equal", gap=1),
        gap_chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo, q):
    mo.md(fr"""
    As we can see the gap between the expectation of AM and GM can be arbitrarily close to 0. This means that without further restraints on the distribution of $X$ it is not possible to measure the gap in any interesting or meaningful way.

    For that reason, we now also assume that there exist $q$ = {q} and $M$ such that

    \[
    \mathbb{{E}}\left[X^{{-q}}\right]\le M.
    \]

    The selected value is $q = {q.amount:.2f}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo, mu_ell_tangle, n, q):
    mo.md(fr"""
    For the selected values $\mu_\ell$ = {mu_ell_tangle.amount:.2f}, $n$ = {n.amount:.0f}, and $q$ = {q.amount:.2f}, the maximizer has a two-point structure:

    $$
    \mathbb{{P}}(X=a)=p,\qquad \mathbb{{P}}(X=b)=1-p,
    $$

    where

    $$
    p = \frac{{b-1}}{{b-a}}.
    $$

    For each value of $M$, the support points $a$ and $b$ are obtained by solving the two equations from the proof:

    $$
    \frac{{b-1}}{{b-a}}\log a + \frac{{1-a}}{{b-a}}\log b = {mu_ell_tangle.amount:.2f},
    $$

    $$
    \frac{{b-1}}{{b-a}}a^{{-q}} + \frac{{1-a}}{{b-a}}b^{{-q}} = M.
    $$

    For some it might not be so obvious whether $X$ with such distribution exists. This problem has been solved and characterized in Lemma 4.2 of my thesis. For our parameters we must enforce: 

    $$
    M > e^{{-q\mu_\ell}} = e^{{-({q.amount:.2f})({mu_ell_tangle.amount:.2f})}}
    $$

    so the $X$ exists.
    """)
    return


@app.cell(hide_code=True)
def _(alt, gap_sequence_rows, mu_ell_tangle, n, np, q, so):
    mu_ell_system = float(mu_ell_tangle.amount)
    q_system = float(q.amount)
    n_system = int(n.amount)

    # Lemma 4.2 in the thesis: feasibility requires M > exp(-q * mu_ell).
    # The inequality is strict, so start the numerical grid just above the boundary.
    _theoretical_M_min = float(np.exp(-q_system * mu_ell_system))
    _numerical_margin = 1e-6
    _M_plot_min = _theoretical_M_min * (1.0 + _numerical_margin)

    # There is no finite theoretical upper bound on M. Use a parameter-dependent,
    # numerically manageable ceiling and let the axis follow the solved data exactly.
    _M_plot_max = max(100.0, 100.0 * _theoretical_M_min)
    _system_log_M_grid = np.linspace(
        np.log(_M_plot_min),
        np.log(_M_plot_max),
        120,
    )

    _seed_log_M = []
    _seed_values = []
    for _row in gap_sequence_rows:
        _log_M = np.logaddexp(
            np.log(_row["p"]) - q_system * _row["log_a"],
            np.log1p(-_row["p"]) - q_system * np.log(_row["b"]),
        )
        _seed_log_M.append(_log_M)
        _seed_values.append([_row["log_a"], np.log(_row["b"] - 1)])

    extremizer_system_rows = []

    for _target_log_M in _system_log_M_grid:
        _seed_index = int(np.argmin(np.abs(np.asarray(_seed_log_M) - _target_log_M)))
        _initial = _seed_values[_seed_index]

        def _system_equations(_variables):
            _log_a, _log_b_minus_one = _variables
            _a = np.exp(_log_a)
            _b = 1 + np.exp(_log_b_minus_one)
            _p = (_b - 1) / (_b - _a)
            _log_moment = _p * _log_a + (1 - _p) * np.log(_b)
            _log_negative_moment = np.logaddexp(
                np.log(_p) - q_system * _log_a,
                np.log1p(-_p) - q_system * np.log(_b),
            )
            return [_log_moment - mu_ell_system, _log_negative_moment - _target_log_M]

        _solution = so.least_squares(
            _system_equations,
            _initial,
            bounds=([-100000.0, -50.0], [-1e-12, 20.0]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=1500,
        )
        if not _solution.success or np.linalg.norm(_system_equations(_solution.x)) > 1e-7:
            continue

        _log_a, _log_b_minus_one = _solution.x
        _a = max(np.exp(_log_a), np.finfo(float).tiny)
        _b = 1 + np.exp(_log_b_minus_one)
        _p = (_b - 1) / (_b - _a)
        _gap = 1 - (_p * _a ** (1 / n_system) + (1 - _p) * _b ** (1 / n_system)) ** n_system
        extremizer_system_rows.append({
            "M": np.exp(_target_log_M),
            "a": _a,
            "b": _b,
            "p": _p,
            "gap": _gap,
            "mean_residual": _system_equations(_solution.x)[0],
            "log_moment_residual": _system_equations(_solution.x)[1],
        })

    extremizer_system_rows.sort(key=lambda _row: _row["M"])
    _computed_M_values = [_row["M"] for _row in extremizer_system_rows]
    _computed_gap_values = [_row["gap"] for _row in extremizer_system_rows]
    _computed_M_domain = [min(_computed_M_values), max(_computed_M_values)]
    _computed_gap_domain = [min(_computed_gap_values), max(_computed_gap_values)]
    if _computed_gap_domain[0] == _computed_gap_domain[1]:
        _computed_gap_domain = [
            _computed_gap_domain[0] - 1e-12,
            _computed_gap_domain[1] + 1e-12,
        ]

    gap_y_min, gap_y_max = _computed_gap_domain

    system_gap_chart = (
        alt.Chart({"values": extremizer_system_rows})
        .mark_line(color="#7c3aed")
        .encode(
            x=alt.X("M:Q", title="M", scale=alt.Scale(domain=_computed_M_domain)),
            y=alt.Y(
                "gap:Q",
                title="Expected AM-GM gap lower bound",
                scale=alt.Scale(domain=_computed_gap_domain),
            ),
            tooltip=["M:Q", "gap:Q", "a:Q", "b:Q", "p:Q"],
        )
        .properties(
            title=f"AM-GM lower bound from the two-equation system (q = {q_system:.2f})",
            width="container",
            height=350,
        )
    )

    system_gap_chart
    return (extremizer_system_rows,)


@app.cell(hide_code=True)
def _(TangleSlider, extremizer_system_rows, mo, np):
    _M_values = np.asarray([_row["M"] for _row in extremizer_system_rows], dtype=float)
    _M_step = max((_M_values[-1] - _M_values[0]) / 1000.0, 0.01)
    M = mo.ui.anywidget(
        TangleSlider(
            amount=float(_M_values[len(_M_values) // 2]),
            min_value=float(_M_values[0]),
            max_value=float(_M_values[-1]),
            step=float(_M_step),
            digits=2,
        )
    )
    return (M,)


@app.cell(hide_code=True)
def _(M, extremizer_system_rows, np):
    _selected_M_values = np.asarray([_row["M"] for _row in extremizer_system_rows], dtype=float)
    _selected_M_index = int(np.argmin(np.abs(_selected_M_values - float(M.amount))))
    selected_M_row = extremizer_system_rows[_selected_M_index]
    return (selected_M_row,)


@app.cell(hide_code=True)
def _(M, mo, selected_M_row):
    selected_M_display = (
        float(M.amount),
        selected_M_row["a"],
        selected_M_row["b"],
        selected_M_row["p"],
        selected_M_row["gap"],
    )

    mo.md(fr"""
    Choose the value of $M$ using {M}. The selected value is $M = {M.amount:.2f}$.

    For this value, solving the equation system gives the two-point distribution

    \[
    \mathbb{{P}}(X=a) = p, \qquad \mathbb{{P}}(X=b) = 1-p,
    \]

    with

    \[
    a = {selected_M_row["a"]:.6g}, \qquad
    b = {selected_M_row["b"]:.6g}, \qquad
    p = {selected_M_row["p"]:.6g}.
    \]

    Therefore, explicitly,

    \[
    \mathbb{{P}}\left(X = {selected_M_row["a"]:.6g}\right) = {selected_M_row["p"]:.6g},
    \qquad
    \mathbb{{P}}\left(X = {selected_M_row["b"]:.6g}\right) = {1-selected_M_row["p"]:.6g}.
    \]

    The corresponding AM--GM lower-bound gap is

    \[
    1 - \left(p\,a^{{1/n}} + (1-p)\,b^{{1/n}}\right)^n
    = {selected_M_row["gap"]:.6g}.
    \]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This summarises the most important facts from my thesis. Hopefully there isn't too much theory. The construction used in my work are quite complicated, but not difficult. The proofs should also be quite easy to follow. The only issue is that it's only available in Polish.
    """)
    return


if __name__ == "__main__":
    app.run()
