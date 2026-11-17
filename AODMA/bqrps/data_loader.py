
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityRecord:
    candidate_id: str
    sampler: str
    nfe: int
    fid: float
    kid_mean: float
    kid_std: float
    sample_count: int
    source_file: str


def load_quality_report(path: str | Path) -> list[QualityRecord]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Quality report not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if "results" not in report:
        raise ValueError("Invalid quality report: missing 'results'.")

    records = []
    seen = set()

    for item in report["results"]:
        candidate = item["candidate"]

        sampler = str(candidate["sampler"]).lower()
        requested_steps = int(candidate["steps"])

        # DDPM always executes the complete reverse process
        # in the current implementation.
        if sampler == "ddpm":
            if requested_steps != 1000:
                continue
            actual_nfe = 1000
        else:
            actual_nfe = requested_steps

        candidate_id = f"{sampler}_{actual_nfe}"

        # Do not silently count duplicate configurations.
        if candidate_id in seen:
            raise ValueError(
                f"Duplicate candidate configuration: {candidate_id}"
            )

        seen.add(candidate_id)

        kid = item["kid"]

        record = QualityRecord(
            candidate_id=candidate_id,
            sampler=sampler,
            nfe=actual_nfe,
            fid=float(item["fid"]),
            kid_mean=float(kid["mean"]),
            kid_std=float(kid["std"]),
            sample_count=int(item["sample_count"]),
            source_file=str(path),
        )

        records.append(record)

    if not records:
        raise ValueError("No valid candidate records found.")

    return records


def print_quality_records(records: list[QualityRecord]) -> None:
    print("\nAODMA — IMPORTED QUALITY PROFILES")
    print("=" * 75)

    print(
        f"{'Candidate':<22}"
        f"{'NFE':<10}"
        f"{'FID':<16}"
        f"{'KID Mean':<16}"
        f"{'Samples':<10}"
    )

    print("-" * 75)

    for record in records:
        print(
            f"{record.candidate_id:<22}"
            f"{record.nfe:<10}"
            f"{record.fid:<16.4f}"
            f"{record.kid_mean:<16.6f}"
            f"{record.sample_count:<10}"
        )

    print("-" * 75)
    print(f"Total valid candidates: {len(records)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    profiles = load_quality_report(args.report)
    print_quality_records(profiles)
