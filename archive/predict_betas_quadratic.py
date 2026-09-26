"""
Predict four stage-wise beta values from a completed
stage-isolated validation landscape.

Uses quadratic interpolation around the highest
coarse validation-accuracy point for each stage.

No test accuracy.
No learned SW-CT beta targets.
No 0.78 anchor.
"""

import argparse
import json


BETA_MIN = 0.70
BETA_MAX = 0.99


def quadratic_peak(
    beta_left,
    beta_center,
    beta_right,
    acc_left,
    acc_center,
    acc_right,
):
    h = beta_center - beta_left

    denominator = (
        acc_left
        - 2.0 * acc_center
        + acc_right
    )

    # If curve is nearly flat, just use
    # the best coarse point.
    if abs(denominator) < 1e-12:
        return beta_center

    predicted = (
        beta_center
        + (h / 2.0)
        * (
            acc_left
            - acc_right
        )
        / denominator
    )

    return max(
        BETA_MIN,
        min(
            BETA_MAX,
            predicted,
        ),
    )


def predict_stage(records):

    betas = [
        record["beta"]
        for record in records
    ]

    accuracies = [
        record["validation_accuracy"]
        for record in records
    ]

    best_index = max(
        range(len(records)),
        key=lambda i: accuracies[i],
    )

    # Boundary case:
    # if coarse optimum is at edge,
    # keep the edge value.
    if best_index == 0:
        return betas[0]

    if best_index == len(records) - 1:
        return betas[-1]

    return quadratic_peak(
        betas[best_index - 1],
        betas[best_index],
        betas[best_index + 1],
        accuracies[best_index - 1],
        accuracies[best_index],
        accuracies[best_index + 1],
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_json",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    with open(
        args.result_json,
        "r",
    ) as handle:
        data = json.load(handle)

    predicted = []

    print("=" * 70)
    print("QUADRATIC STAGE-WISE BETA PREDICTION")
    print("=" * 70)

    for stage in range(1, 5):

        records = data[
            "stages"
        ][str(stage)][
            "records"
        ]

        beta = predict_stage(
            records
        )

        predicted.append(beta)

        print(
            f"Stage {stage}: "
            f"{beta:.6f}"
        )

    print(
        "\nPredicted beta vector:"
    )

    print([
        round(x, 6)
        for x in predicted
    ])

    print("=" * 70)


if __name__ == "__main__":
    main()