"""Does a surrogate's apparent accuracy depend on the experiment you pick?

Aerodynamic surrogates are normally scored against the solver that produced
their training data. This study instead scores predictions against *experiment*,
and separates the experiments by transition state.

The result is that the same model, on the same airfoil at the same Reynolds
number, appears roughly three times more accurate against one experiment than
the other. The difference is not model quality: it is whether the experiment's
boundary layer was tripped, which is a physical condition that surrogate
benchmarks do not record.

Drag is the discriminating quantity. Lift is set by circulation and is largely
insensitive to transition, so lift-based metrics hide the effect entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .experimental import ExperimentalDataset, load_abbott, load_ladson

ALPHA_SWEEP = np.arange(-4.0, 11.01, 0.5)


@dataclass
class ErrorSummary:
    """Prediction error against one experimental dataset."""

    dataset: str
    transition: str
    n: int
    mean_abs_pct: float
    bias_pct: float

    def __str__(self) -> str:
        return (
            f"{self.dataset:<26} [{self.transition:>7}]  n={self.n:>3}  "
            f"mean|err|={self.mean_abs_pct:>5.1f}%  bias={self.bias_pct:>+6.1f}%"
        )


def _scalar(value) -> float:
    return float(np.atleast_1d(np.asarray(value)).ravel()[0])


def neuralfoil_polar(reynolds: float, alphas=ALPHA_SWEEP, model_size: str = "xxlarge"):
    """Return (cl, cd) arrays for a NACA0012 across ``alphas``."""

    import aerosandbox as asb
    import neuralfoil as nf

    coords = asb.Airfoil("naca0012").coordinates
    cl, cd = [], []
    for alpha in alphas:
        aero = nf.get_aero_from_coordinates(
            coordinates=coords, alpha=float(alpha), Re=reynolds, model_size=model_size
        )
        cl.append(_scalar(aero["CL"]))
        cd.append(_scalar(aero["CD"]))
    return np.array(cl), np.array(cd)


def drag_error_at_matched_lift(
    pred_cl: np.ndarray,
    pred_cd: np.ndarray,
    dataset: ExperimentalDataset,
) -> ErrorSummary:
    """Compare predicted and measured drag at matched lift coefficient.

    Matching on Cl rather than angle of attack removes any offset caused by
    tunnel-wall corrections or a small difference in effective incidence, so the
    comparison isolates the drag physics.
    """

    if dataset.cd is None:
        raise ValueError(f"{dataset.name} has no drag data")

    inside = (dataset.cl >= pred_cl.min()) & (dataset.cl <= pred_cl.max())
    exp_cl, exp_cd = dataset.cl[inside], dataset.cd[inside]
    if len(exp_cl) == 0:
        raise ValueError("No overlap between prediction and experiment in Cl")

    order = np.argsort(pred_cl)
    predicted = np.interp(exp_cl, pred_cl[order], pred_cd[order])
    pct = 100.0 * (predicted - exp_cd) / exp_cd

    return ErrorSummary(
        dataset=dataset.name,
        transition=dataset.transition,
        n=len(exp_cl),
        mean_abs_pct=float(np.mean(np.abs(pct))),
        bias_pct=float(np.mean(pct)),
    )


def lift_error_vs_alpha(dataset: ExperimentalDataset, reynolds: float) -> ErrorSummary:
    """Lift error against angle of attack, for contrast with the drag error."""

    if dataset.alpha is None:
        raise ValueError(f"{dataset.name} has no alpha data")

    mask = (dataset.alpha >= -6) & (dataset.alpha <= 12) & (np.abs(dataset.cl) > 0.05)
    alphas, exp_cl = dataset.alpha[mask], dataset.cl[mask]
    pred_cl, _ = neuralfoil_polar(reynolds, alphas)
    pct = 100.0 * (pred_cl - exp_cl) / np.abs(exp_cl)

    return ErrorSummary(
        dataset=dataset.name + " (lift)",
        transition=dataset.transition,
        n=len(alphas),
        mean_abs_pct=float(np.mean(np.abs(pct))),
        bias_pct=float(np.mean(pct)),
    )


def run() -> dict:
    ladson, abbott = load_ladson(), load_abbott()
    pred_cl, pred_cd = neuralfoil_polar(6.0e6)

    drag_tripped = drag_error_at_matched_lift(pred_cl, pred_cd, ladson)
    drag_free = drag_error_at_matched_lift(pred_cl, pred_cd, abbott)
    lift_tripped = lift_error_vs_alpha(ladson, 6.0e6)

    return {
        "drag_tripped": drag_tripped,
        "drag_free": drag_free,
        "lift_tripped": lift_tripped,
        "drag_ratio": drag_tripped.mean_abs_pct / drag_free.mean_abs_pct,
        "lift_drag_ratio": drag_tripped.mean_abs_pct / lift_tripped.mean_abs_pct,
    }


def main() -> None:
    r = run()
    print("NeuralFoil vs experiment -- NACA0012, Re=6e6\n")
    print("Drag error at matched lift:")
    print(f"  {r['drag_tripped']}")
    print(f"  {r['drag_free']}")
    print(f"\n  ratio: apparent error is {r['drag_ratio']:.1f}x larger against the")
    print( "  tripped experiment than the untripped one, for the same model.")
    print("\nLift error for contrast:")
    print(f"  {r['lift_tripped']}")
    print(f"\n  Drag error is {r['lift_drag_ratio']:.1f}x the lift error. A benchmark")
    print( "  reporting lift or field MSE would not reveal this.")
    print(
        "\nInterpretation: NeuralFoil inherits XFOIL's transition model, so it"
        "\npredicts a laminar run over the forward chord and therefore less drag"
        "\nthan a tripped experiment can show. The bias is negative by"
        "\nconstruction. Neither the model nor the benchmark records which"
        "\ncondition is intended."
    )


if __name__ == "__main__":
    main()
