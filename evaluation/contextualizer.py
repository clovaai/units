import argparse

import numpy as np
from text_evaluation import TextEvaluator

# Dictionary mapping eval_db argument to dataset_name
dataset_mapping = {
    "icdar2015": ["icdar2015"],
    "totaltext": ["totaltext"],
    "ctw1500": ["ctw1500"],
    "textocr": ["textocr"],
    "custom": ["custom"],
}

# Dictionary mapping evaluation types for icdar2015
eval_type_mapping = {"generic": 1, "weak": 2, "strong": 3}


def evaluate_text(args):
    cfg = {}
    eval_dataset = args.eval_db

    if eval_dataset in dataset_mapping:
        dataset_name = dataset_mapping[eval_dataset]
    else:
        raise ValueError("Invalid eval_db argument")

    if eval_dataset == "icdar2015":
        if args.generic:
            cfg["EVAL_TYPE"] = eval_type_mapping["generic"]
        elif args.weak:
            cfg["EVAL_TYPE"] = eval_type_mapping["weak"]
        elif args.strong:
            cfg["EVAL_TYPE"] = eval_type_mapping["strong"]
        else:
            cfg["EVAL_TYPE"] = None
    else:
        cfg["EVAL_TYPE"] = args.full

    cfg["WORD_SPOTTING"] = args.word_spotting

    dets, e2e_none, e2e_full = dict(), dict(), dict()
    start_perplexity_threshold = 0.80
    end_perp_threshold = 0.99
    thresh_increment = 0.01
    end_perp_threshold += 0.001

    print("=" * 100)

    for threshold in np.arange(
        start_perplexity_threshold, end_perp_threshold, thresh_increment
    ):
        cfg["INFERENCE_TH_TEST"] = threshold

        e = TextEvaluator(dataset_name, cfg, output_dir=args.output)
        res = e.evaluate()

        print("Threshold: {:.2f}".format(threshold))

        for key, value in res.items():
            print(f"{key}: Calculated!{value}")

        print("=" * 100)

        dets[threshold] = res["DETECTION_ONLY_RESULTS"]
        e2e_none[threshold] = res["None-E2E_RESULTS"]
        e2e_full[threshold] = res["Full-E2E_RESULTS"]

    th_dets = max(dets, key=lambda x: dets[x]["hmean"])
    th_e2e_none = max(e2e_none, key=lambda x: e2e_none[x]["hmean"])
    th_e2e_full = max(e2e_full, key=lambda x: e2e_full[x]["hmean"])

    print(
        "[Contextualizer] BEST SCORE || Det: {:.4f} @{:.2f} || E2E: {:.4f} @{:.2f} || E2E FULL: {:.4f} @{:.2f}".format(
            dets[th_dets]["hmean"],
            th_dets,
            e2e_none[th_e2e_none]["hmean"],
            th_e2e_none,
            e2e_full[th_e2e_full]["hmean"],
            th_e2e_full,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add an argument for setting the output file name
    parser.add_argument(
        "--output",
        type=str,
        default="text_results.json",
        help="Specify the output file name",
    )

    # Add an argument for selecting the dataset
    parser.add_argument(
        "--eval_db",
        type=str,
        choices=dataset_mapping.keys(),
        default="totaltext",
        help="Select the dataset for evaluation",
    )

    # Add arguments for selecting the vocab type
    parser.add_argument(
        "-g", "--generic", default=False, action="store_true", help="Use generic vocab"
    )
    parser.add_argument(
        "-w", "--weak", default=False, action="store_true", help="Use weak vocab"
    )
    parser.add_argument(
        "-s", "--strong", default=False, action="store_true", help="Use strong vocab"
    )

    # Add an argument for enabling full vocab for TotalText dataset
    parser.add_argument(
        "-f",
        "--full",
        default=False,
        action="store_true",
        help="Use full vocab for TotalText dataset",
    )

    # Add an argument for enabling word spotting
    parser.add_argument(
        "--word_spotting",
        default=False,
        action="store_true",
        help="Word spotting evaluation protocol",
    )

    args = parser.parse_args()

    evaluate_text(args)
