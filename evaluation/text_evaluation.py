import copy
import glob
import json
import logging
import os
import re
import shutil
import zipfile
from collections import OrderedDict

import editdistance
import text_eval_script
import text_eval_script_ic15
import torch
from shapely.geometry import LinearRing, Polygon


class TextEvaluator:
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # use dataset_name to decide eval_gt_path
        self.lexicon_type = cfg["EVAL_TYPE"]
        if "totaltext" in dataset_name:
            self._text_eval_gt_path = "datasets/gt/gt_totaltext.zip"
            self._word_spotting = cfg["WORD_SPOTTING"]
            self.dataset_name = "totaltext"
        elif "ctw1500" in dataset_name:
            self._text_eval_gt_path = "datasets/gt/gt_ctw1500.zip"
            self._word_spotting = cfg["WORD_SPOTTING"]
            self.dataset_name = "ctw1500"
        elif "icdar2015" in dataset_name:
            self._text_eval_gt_path = "datasets/gt/gt_icdar2015.zip"
            self._word_spotting = cfg["WORD_SPOTTING"]
            self.dataset_name = "icdar2015"
        elif "textocr" in dataset_name:
            self._text_eval_gt_path = "datasets/gt/gt_textocr.zip"
            self._word_spotting = cfg["WORD_SPOTTING"]
            self.dataset_name = "textocr"
        elif "custom" in dataset_name:
            self._text_eval_gt_path = "datasets/gt/gt_custom.zip"
            self._word_spotting = cfg["WORD_SPOTTING"]

        self._text_eval_confidence = cfg["INFERENCE_TH_TEST"]

    def to_eval_format(self, file_path, temp_dir="temp_det_results", cf_th=0.5):
        def fis_ascii(s):
            a = (ord(c) < 128 for c in s)
            return all(a)

        def de_ascii(s):
            a = [c for c in s if ord(c) < 128]
            outa = ""
            for i in a:
                outa += i
            return outa

        with open(file_path, "r") as f:
            data = json.load(f)
            with open("temp_all_det_cors.txt", "w") as f2:
                for ix in range(len(data)):
                    if data[ix]["score"] > 0.1:
                        outstr = "{}: ".format(data[ix]["image_id"])
                        xmin = 1000000
                        ymin = 1000000
                        xmax = 0
                        ymax = 0
                        for i in range(len(data[ix]["polys"])):
                            outstr = (
                                outstr
                                + str(int(data[ix]["polys"][i][0]))
                                + ","
                                + str(int(data[ix]["polys"][i][1]))
                                + ","
                            )
                        # ass = de_ascii(data[ix]['rec'])
                        ass = str(data[ix]["rec"])
                        if len(ass) >= 0:  #
                            outstr = (
                                outstr
                                + str(round(data[ix]["score"], 3))
                                + ",####"
                                + ass
                                + "\n"
                            )
                            f2.writelines(outstr)
                f2.close()
        dirn = temp_dir
        lsc = [cf_th]
        fres = open("temp_all_det_cors.txt", "r").readlines()
        for isc in lsc:
            if not os.path.isdir(dirn):
                os.mkdir(dirn)

            for line in fres:
                line = line.strip()
                s = line.split(": ")
                if "textocr" in self.dataset_name:
                    filename = "{}.txt".format(s[0])
                else:
                    filename = "{:07d}.txt".format(int(s[0]))
                outName = os.path.join(dirn, filename)
                with open(outName, "a") as fout:
                    ptr = s[1].strip().split(",####")
                    score = ptr[0].split(",")[-1]
                    if float(score) < isc:
                        continue
                    if "icdar2015" in self.dataset_name and float(score) < 0.45:
                        continue
                    cors = ",".join(e for e in ptr[0].split(",")[:-1])
                    fout.writelines(cors + ",####" + str(ptr[1]) + "\n")
        os.remove("temp_all_det_cors.txt")

    def sort_detection(self, temp_dir):
        origin_file = temp_dir
        output_file = "final_" + temp_dir
        output_file_full = "full_final_" + temp_dir
        if not os.path.isdir(output_file_full):
            os.mkdir(output_file_full)
        if not os.path.isdir(output_file):
            os.mkdir(output_file)
        files = glob.glob(origin_file + "*.txt")
        files.sort()
        if "totaltext" in self.dataset_name:
            if not self.lexicon_type == None:
                lexicon_path = "datasets/vocab/totaltext/weak_voc_new.txt"
                lexicon_fid = open(lexicon_path, "r")
                pair_list = open("datasets/vocab/totaltext/weak_voc_pair_list.txt", "r")
                pairs = dict()
                for line in pair_list.readlines():
                    line = line.strip()
                    word = line.split(" ")[0].upper()
                    word_gt = line[len(word) + 1 :]
                    pairs[word] = word_gt
                lexicon_fid = open(lexicon_path, "r")
                lexicon = []
                for line in lexicon_fid.readlines():
                    line = line.strip()
                    lexicon.append(line)
        elif "ctw1500" in self.dataset_name:
            if not self.lexicon_type == None:
                lexicon_path = "datasets/vocab/ctw1500/weak_voc_new.txt"
                lexicon_fid = open(lexicon_path, "r")
                pair_list = open("datasets/vocab/ctw1500/weak_voc_pair_list.txt", "r")
                pairs = dict()
                lexicon_fid = open(lexicon_path, "r")
                lexicon = []
                for line in lexicon_fid.readlines():
                    line = line.strip()
                    lexicon.append(line)
                    pairs[line.upper()] = line
        elif "icdar2015" in self.dataset_name:
            if self.lexicon_type == 1:
                # generic lexicon
                lexicon_path = "datasets/vocab/icdar2015/GenericVocabulary_new.txt"
                lexicon_fid = open(lexicon_path, "r")
                pair_list = open(
                    "datasets/vocab/icdar2015/GenericVocabulary_pair_list.txt", "r"
                )
                pairs = dict()
                for line in pair_list.readlines():
                    line = line.strip()
                    word = line.split(" ")[0].upper()
                    word_gt = line[len(word) + 1 :]
                    pairs[word] = word_gt
                lexicon_fid = open(lexicon_path, "r")
                lexicon = []
                for line in lexicon_fid.readlines():
                    line = line.strip()
                    lexicon.append(line)
            if self.lexicon_type == 2:
                # weak lexicon
                lexicon_path = "datasets/vocab/icdar2015/ch4_test_vocabulary_new.txt"
                lexicon_fid = open(lexicon_path, "r")
                pair_list = open(
                    "datasets/vocab/icdar2015/ch4_test_vocabulary_pair_list.txt", "r"
                )
                pairs = dict()
                for line in pair_list.readlines():
                    line = line.strip()
                    word = line.split(" ")[0].upper()
                    word_gt = line[len(word) + 1 :]
                    pairs[word] = word_gt
                lexicon_fid = open(lexicon_path, "r")
                lexicon = []
                for line in lexicon_fid.readlines():
                    line = line.strip()
                    lexicon.append(line)

        def find_match_word(rec_str, pairs, lexicon=None):
            rec_str = rec_str.upper()
            dist_min = 100
            dist_min_pre = 100
            match_word = ""
            match_dist = 100
            for word in lexicon:
                word = word.upper()
                ed = editdistance.eval(rec_str, word)
                length_dist = abs(len(word) - len(rec_str))
                dist = ed
                if dist < dist_min:
                    dist_min = dist
                    match_word = pairs[word]
                    match_dist = dist
            return match_word, match_dist

        for i in files:
            if "icdar2015" in self.dataset_name:
                out = (
                    output_file
                    + "res_img_"
                    + str(int(i.split("/")[-1].split(".")[0]))
                    + ".txt"
                )
                out_full = (
                    output_file_full
                    + "res_img_"
                    + str(int(i.split("/")[-1].split(".")[0]))
                    + ".txt"
                )
                if self.lexicon_type == 3:
                    lexicon_path = (
                        "datasets/vocab/icdar2015/new_strong_lexicon/new_voc_img_"
                        + str(int(i.split("/")[-1].split(".")[0]))
                        + ".txt"
                    )
                    lexicon_fid = open(lexicon_path, "r")
                    pair_list = open(
                        "datasets/vocab/icdar2015/new_strong_lexicon/pair_voc_img_"
                        + str(int(i.split("/")[-1].split(".")[0]))
                        + ".txt"
                    )
                    pairs = dict()
                    for line in pair_list.readlines():
                        line = line.strip()
                        word = line.split(" ")[0].upper()
                        word_gt = line[len(word) + 1 :]
                        pairs[word] = word_gt
                    lexicon_fid = open(lexicon_path, "r")
                    lexicon = []
                    for line in lexicon_fid.readlines():
                        line = line.strip()
                        lexicon.append(line)
            else:
                out = i.replace(origin_file, output_file)
                out_full = i.replace(origin_file, output_file_full)
            fin = open(i, "r").readlines()
            fout = open(out, "w")
            fout_full = open(out_full, "w")
            for iline, line in enumerate(fin):
                ptr = line.strip().split(",####")
                rec = ptr[1]
                cors = ptr[0].split(",")
                assert len(cors) % 2 == 0, "cors invalid."
                pts = [(int(cors[j]), int(cors[j + 1])) for j in range(0, len(cors), 2)]
                try:
                    pgt = Polygon(pts)
                except Exception as e:
                    print(e)
                    print(
                        "An invalid detection in {} line {} is removed ... ".format(
                            i, iline
                        )
                    )
                    continue
                # if not pgt.is_valid:
                #     print('An invalid detection in {} line {} is removed ... '.format(i, iline))
                #     continue

                pRing = LinearRing(pts)
                if not "icdar2015" in self.dataset_name:
                    if pRing.is_ccw:
                        pts.reverse()
                outstr = ""
                for ipt in pts[:-1]:
                    outstr += str(int(ipt[0])) + "," + str(int(ipt[1])) + ","
                outstr += str(int(pts[-1][0])) + "," + str(int(pts[-1][1]))
                pts = outstr
                if "icdar2015" in self.dataset_name:
                    outstr = outstr + "," + rec
                else:
                    outstr = outstr + ",####" + rec
                fout.writelines(outstr + "\n")
                if self.lexicon_type is None:
                    rec_full = rec
                else:
                    match_word, match_dist = find_match_word(rec, pairs, lexicon)
                    if match_dist < 1.5:
                        rec_full = match_word
                        if "icdar2015" in self.dataset_name:
                            pts = pts + "," + rec_full
                        else:
                            pts = pts + ",####" + rec_full
                        fout_full.writelines(pts + "\n")
            fout.close()
            fout_full.close()

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    ziph.write(os.path.join(root, file))

        if "icdar2015" in self.dataset_name:
            os.system("zip -r -q -j " + "det.zip" + " " + output_file + "/*")
            os.system("zip -r -q -j " + "det_full.zip" + " " + output_file_full + "/*")
            shutil.rmtree(origin_file)
            shutil.rmtree(output_file)
            shutil.rmtree(output_file_full)
            return "det.zip", "det_full.zip"
        else:
            os.chdir(output_file)
            zipf = zipfile.ZipFile("../det.zip", "w", zipfile.ZIP_DEFLATED)
            zipdir("./", zipf)
            zipf.close()
            os.chdir("../")

            os.chdir(output_file_full)
            zipf_full = zipfile.ZipFile("../det_full.zip", "w", zipfile.ZIP_DEFLATED)
            zipdir("./", zipf_full)
            zipf_full.close()
            os.chdir("../")
            # clean temp files

            shutil.rmtree(origin_file)
            shutil.rmtree(output_file)
            shutil.rmtree(output_file_full)
            return "det.zip", "det_full.zip"

    def evaluate_with_official_code(self, result_path, gt_path):
        if "icdar2015" in self.dataset_name:
            return text_eval_script_ic15.text_eval_main_ic15(
                det_file=result_path,
                gt_file=gt_path,
                is_word_spotting=self._word_spotting,
            )
        else:
            return text_eval_script.text_eval_main(
                det_file=result_path,
                gt_file=gt_path,
                is_word_spotting=self._word_spotting,
            )

    def evaluate(self):
        # file_path = os.path.join(self._output_dir, "text_results.json")
        file_path = os.path.join(self._output_dir)

        self._results = OrderedDict()
        # eval text
        if not self._text_eval_gt_path:
            return copy.deepcopy(self._results)
        temp_dir = "temp_det_results/"
        self.to_eval_format(file_path, temp_dir, self._text_eval_confidence)
        result_path, result_path_full = self.sort_detection(temp_dir)
        text_result = self.evaluate_with_official_code(
            result_path, self._text_eval_gt_path
        )  # None
        text_result["e2e_method"] = "None-" + text_result["e2e_method"]
        text_result_full = self.evaluate_with_official_code(
            result_path_full, self._text_eval_gt_path
        )  # with lexicon
        text_result_full["e2e_method"] = "Full-" + text_result_full["e2e_method"]

        os.remove(result_path)
        os.remove(result_path_full)
        # parse
        template = "(\S+): (\S+): (\S+), (\S+): (\S+), (\S+): (\S+)"
        result = text_result["det_only_method"]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {
            groups[i * 2 + 1]: float(groups[(i + 1) * 2]) for i in range(3)
        }
        result = text_result["e2e_method"]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {
            groups[i * 2 + 1]: float(groups[(i + 1) * 2]) for i in range(3)
        }
        result = text_result_full["e2e_method"]
        groups = re.match(template, result).groups()
        self._results[groups[0]] = {
            groups[i * 2 + 1]: float(groups[(i + 1) * 2]) for i in range(3)
        }

        return copy.deepcopy(self._results)
