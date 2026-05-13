import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_snr_table(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"arm_length_m", "test_mass_kg", "snr_year"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain columns: arm_length_m, test_mass_kg, snr_year"
            )

        for row in reader:
            rows.append(
                {
                    "l": float(row["arm_length_m"]),
                    "m": float(row["test_mass_kg"]),
                    "snr": float(row["snr_year"]),
                }
            )
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")
    return rows

def plot_2d(rows: list[dict[str, float]], output_path: Path, show: bool) -> None:
    y_l = [r["l"] for r in rows]
    z_snr = [r["snr"] for r in rows]
    fig = plt.figure(figsize=(9, 7))
    plt.scatter(y_l, z_snr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure: {output_path}")


def plot_3d(rows: list[dict[str, float]], output_path: Path, show: bool) -> None:
    x_m = [r["m"] for r in rows]
    y_l = [r["l"] for r in rows]
    z_snr = [r["snr"] for r in rows]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        x_m, y_l, z_snr, c=z_snr, cmap="viridis", s=45, alpha=0.95, depthshade=True
    )

    ax.set_title("SNR (Year) Relying on Test Mass and Arm Length")
    ax.set_xlabel("m (kg)")
    ax.set_ylabel("l (m)")
    ax.set_zlabel("SNR (Year)")
    fig.colorbar(scatter, ax=ax, shrink=0.75, pad=0.1, label="SNR (Year)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved figure: {output_path}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a 3D SNR figure from data/snr_year_table.csv with axes m, l, and snr."
    )
    parser.add_argument(
        "--input",
        default="data/snr_year_table.csv",
        help="Input CSV path. Default: data/snr_year_table.csv",
    )
    parser.add_argument(
        "--output",
        default=f"images/SNR (3D).png",
        help="Output figure path. Default: images/SNR (3D).png",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    args = parser.parse_args()

    rows = load_snr_table(Path(args.input))
    plot_3d(rows, Path(args.output), show=args.show)


if __name__ == "__main__":
    main()
    # rows = load_snr_table(Path("data/snr_year_table.csv"))
    # plot_2d(rows, Path("images/SNR (2D).png"), show=True)
