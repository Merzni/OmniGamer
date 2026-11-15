"""Select a region-change threshold from labelled before/after frame pairs."""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from omnigamer.profiles import GameProfile, load_profile
from omnigamer.personality import configure_stdio
from omnigamer.skills.basic import region_delta


def choose_threshold(samples, margin=0.01):
    negatives = [float(value) for label, value in samples if label == "no_change"]
    positives = [float(value) for label, value in samples if label == "target_change"]
    if not negatives or not positives:
        raise ValueError("Need at least one no_change and one target_change sample")
    max_negative = max(negatives)
    min_positive = min(positives)
    separable = min_positive > max_negative
    threshold = ((max_negative + min_positive) / 2 if separable
                 else min(1.0, max_negative + max(0.001, margin)))
    false_positives = sum(value >= threshold for value in negatives)
    false_negatives = sum(value < threshold for value in positives)
    return {
        "threshold": threshold,
        "separable": separable,
        "no_change_samples": len(negatives),
        "target_change_samples": len(positives),
        "no_change_max": max_negative,
        "target_change_min": min_positive,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_positive_rate": false_positives / len(negatives),
        "false_negative_rate": false_negatives / len(positives),
        "threshold_margin_over_no_change": threshold - max_negative,
        "mean_no_change": statistics.mean(negatives),
        "mean_target_change": statistics.mean(positives),
    }


def load_samples(manifest, region):
    manifest = Path(manifest).resolve()
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.DictReader(stream), 2):
            label = row.get("class", "").strip()
            if label not in {"no_change", "target_change"}:
                raise ValueError(f"{manifest.name}:{number}: class must be no_change or target_change")
            before = Image.open(manifest.parent / row["before"]).convert("RGB")
            after = Image.open(manifest.parent / row["after"]).convert("RGB")
            rows.append({"sample_id": row.get("sample_id", str(number - 1)),
                         "class": label,
                         "before": row["before"], "after": row["after"],
                         "delta": region_delta(before, after, region)})
    return rows


def write_threshold(profile_path, profile, threshold):
    path = Path(profile_path)
    raw = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else (
        profile.model_dump(mode="json", exclude={"comment"}))
    raw.setdefault("thresholds", {})["region_change"] = float(threshold)
    GameProfile.model_validate(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    configure_stdio()
    parser = argparse.ArgumentParser(description="Calibrate normalized region-change threshold")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--profile", default="")
    parser.add_argument("--region", required=True)
    parser.add_argument("--manifest", required=True, help="CSV: sample_id,before,after,class")
    parser.add_argument("--output", default="", help="JSON report path")
    parser.add_argument("--accept", action="store_true", help="Write selected threshold to the profile")
    args = parser.parse_args(argv)
    profile = load_profile(args.game_id, args.profile or None)
    if args.region not in profile.regions:
        parser.error(f"Unknown profile region {args.region!r}")
    rows = load_samples(args.manifest, profile.regions[args.region])
    report = choose_threshold([(row["class"], row["delta"]) for row in rows])
    report.update(game_id=profile.game_id, region=args.region, metric="brightness_color_p97_v2",
                  noise_floor=4, samples=rows)
    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parents[1] / "data" / "diagnostics" / "region_metric" /
        f"{profile.game_id}_{args.region}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"},
                     ensure_ascii=False, indent=2))
    print(f"Отчёт: {output}")
    if args.accept:
        profile_path = Path(args.profile) if args.profile else (
            Path(__file__).resolve().parents[1] / "data" / "profiles" / f"{args.game_id}.json")
        write_threshold(profile_path, profile, report["threshold"])
        print(f"Порог записан в профиль: {profile_path}")


if __name__ == "__main__":
    main()
