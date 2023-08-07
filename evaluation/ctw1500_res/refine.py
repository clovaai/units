import argparse
import glob as glob
import json
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, default="res_output")
    parser.add_argument("--outputs", type=str, default="output.json")
    parser.add_argument("--num_pts", type=int, default=16)
    args = parser.parse_args()

    filenames = glob.glob(os.path.join(args.inputs, "*"))
    filenames = sorted(filenames)
    results = []

    for file_id, filename in enumerate(filenames):
        input_file = open(filename, "r")
        img_id = filename.split("/")[-1].split(".")[0][4:]
        lines = input_file.readlines()
        for line in lines:
            line = line.split(",")
            x = int(np.mean(list(map(int, line[0 : 2 * args.num_pts : 2]))))
            y = int(np.mean(list(map(int, line[1 : 2 * args.num_pts : 2]))))
            score = float(line[2 * args.num_pts])
            text = ",".join(line[2 * args.num_pts + 1 :]).replace("\n", "")

            polys = [
                [x, y]
                for x, y in zip(
                    line[0 : 2 * args.num_pts : 2], line[1 : 2 * args.num_pts : 2]
                )
            ]
            result = {"image_id": img_id, "polys": polys, "rec": text, "score": score}
            results.append(result)

    with open(args.outputs, "w") as f:
        json.dump(results, f)
