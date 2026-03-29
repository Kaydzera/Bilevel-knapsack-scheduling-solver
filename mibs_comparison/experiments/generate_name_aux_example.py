"""Generate a small MibS-compatible MPS + name-based AUX example."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from formulation.bilevel_model import BilevelInstance
from formulation.mps_name_aux_generator import generate_mps_name_aux_files


def main() -> None:
    instance = BilevelInstance(
        n_job_types=4,
        n_machines=2,
        durations=[10, 20, 15, 25],
        prices=[5, 8, 6, 10],
        budget=20,
        seed=42,
        metadata={"source": "name_aux_example", "repetition": 0},
    )

    output_dir = Path(__file__).parent / "test_output"
    mps_path, aux_path = generate_mps_name_aux_files(instance, str(output_dir))

    print(f"Generated MPS: {mps_path}")
    print(f"Generated AUX: {aux_path}")
    print("\nRun with MibS:")
    print(f"mibs -Alps_instance {mps_path} -MibS_auxiliaryInfoFile {aux_path}")


if __name__ == "__main__":
    main()
