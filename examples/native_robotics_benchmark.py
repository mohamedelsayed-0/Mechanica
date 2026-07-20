"""Benchmark batched Torch and C++ rigid-body algorithms."""

from argparse import ArgumentParser
from statistics import median
from time import perf_counter

import torch

from mechanica import (
    forward_dynamics_aba,
    inverse_dynamics_rnea,
    load_urdf,
    mass_matrix_crba,
    native_kernels_status,
)


def chain_urdf(links: int) -> str:
    bodies = ['<robot name="benchmark">', '<link name="base"/>']
    for index in range(links):
        bodies.append(
            f'<link name="link{index}"><inertial><origin xyz="0.2 0 0"/>'
            '<mass value="1"/><inertia ixx="0.02" ixy="0" ixz="0" '
            'iyy="0.05" iyz="0" izz="0.05"/></inertial></link>'
        )
        parent = "base" if index == 0 else f"link{index - 1}"
        bodies.append(
            f'<joint name="joint{index}" type="revolute"><parent link="{parent}"/>'
            f'<child link="link{index}"/><origin xyz="0.4 0 0"/>'
            '<axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14"/></joint>'
        )
    return "".join([*bodies, "</robot>"])


def timing(fn, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = perf_counter()
        fn()
        samples.append(perf_counter() - start)
    return median(samples)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--links", type=int, default=7)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument("--runs", type=int, default=25)
    args = parser.parse_args()

    available, reason = native_kernels_status()
    if not available:
        raise SystemExit(f"native kernels unavailable: {reason}")

    torch.set_num_threads(1)
    model = load_urdf(chain_urdf(args.links), dtype=torch.float32)
    gravity = torch.tensor([0.0, 0.0, -9.80665])
    algorithms = {
        "RNEA": lambda q, qdot, value, native: inverse_dynamics_rnea(
            model, q, qdot, value, gravity=gravity, use_native=native
        ),
        "CRBA": lambda q, _qdot, _value, native: mass_matrix_crba(
            model, q, use_native=native
        ),
        "ABA": lambda q, qdot, value, native: forward_dynamics_aba(
            model, q, qdot, value, gravity=gravity, use_native=native
        ),
    }

    print(f"{args.links}-DoF chain, float32, one CPU thread")
    print(f"{'algorithm':<10}{'batch':>8}{'torch ms':>12}{'native ms':>12}{'speedup':>10}")
    for batch in args.batch:
        q = torch.randn(batch, model.dof) * 0.2
        qdot = torch.randn_like(q) * 0.1
        value = torch.randn_like(q) * 0.1
        for name, algorithm in algorithms.items():
            def pure():
                return algorithm(q, qdot, value, False)

            def native():
                return algorithm(q, qdot, value, True)

            expected, actual = pure(), native()
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            pure_time = timing(pure, 3, args.runs)
            native_time = timing(native, 3, args.runs)
            print(
                f"{name:<10}{batch:>8}{pure_time * 1e3:>12.3f}"
                f"{native_time * 1e3:>12.3f}{pure_time / native_time:>9.2f}x"
            )


if __name__ == "__main__":
    main()
