import sys
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
from lvos.lvos_seperate import LVOS
from lvos.metrics import db_eval_boundary, db_eval_iou
from lvos import utils
from lvos.results import Results
from scipy.optimize import linear_sum_assignment


class LVOSEvaluation(object):
    def __init__(
        self, lvos_root, task, gt_set, mp_procs=2, codalab=False, use_cache=False
    ):
        """
        Class to evaluate LVOS sequences from a certain set and for a certain task
        :param lvos_root: Path to the LVOS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.lvos_root = lvos_root
        self.task = task
        self.use_cache = use_cache
        self.dataset = LVOS(root=lvos_root, task=task, subset=gt_set, codalab=codalab)
        self.mp_procs = mp_procs

        sys.path.append(".")
        if codalab:
            self.unseen_videos = os.path.join(lvos_root, "unseen_videos.txt")
        else:
            self.unseen_videos = "./unseen_videos.txt"

        self.unseen_videos = open(self.unseen_videos, mode="r").readlines()
        for vi in range(len(self.unseen_videos)):
            self.unseen_videos[vi] = self.unseen_videos[vi].strip()

        if codalab:
            self.unsup_videos = os.path.join(lvos_root, "unsupervised_videos.txt")
        else:
            self.unsup_videos = "./unsupervised_videos.txt"

        self.unsup_videos = open(self.unsup_videos, mode="r").readlines()
        for vi in range(len(self.unsup_videos)):
            self.unsup_videos[vi] = self.unsup_videos[vi].strip()

    def _evaluate_semisupervised(self, seq, results, all_void_masks, metric):
        seq_name = list(seq.keys())[0]
        seq = seq[seq_name]

        objs = list(seq.keys())
        j_metrics_res = dict()
        f_metrics_res = dict()
        for oi in range(len(objs)):
            _obj = objs[oi]
            _frame_num = seq[_obj]["frame_range"]["frame_nums"]
            j_metrics_res[str(_obj)] = np.zeros((1, int(_frame_num) - 2))
            f_metrics_res[str(_obj)] = np.zeros((1, int(_frame_num) - 2))

        for oi in range(len(objs)):
            _obj = objs[oi]
            _frame_num = seq[_obj]["frame_range"]["frame_nums"]
            start_frame = seq[_obj]["frame_range"]["start"]
            end_frame = seq[_obj]["frame_range"]["end"]

            oidx = 0
            for ii in range(int(start_frame), int(end_frame), 5):
                if oidx == 0 or oidx == (int(_frame_num) - 1):
                    oidx = oidx + 1
                    continue
                gt_mask, _ = self.dataset.get_mask(seq_name, "{0:08d}".format(ii), _obj)
                res_mask = results.read_mask(seq_name, "{0:08d}".format(ii), _obj)
                if "J" in metric:
                    j_metrics_res[str(_obj)][0, oidx - 1] = db_eval_iou(
                        gt_mask, res_mask, all_void_masks
                    )
                if "F" in metric:
                    f_metrics_res[str(_obj)][0, oidx - 1] = db_eval_boundary(
                        gt_mask,
                        res_mask,
                        all_void_masks,
                        video_name=seq_name,
                        frame_name="{0:08d}".format(ii),
                        eval_obj=_obj,
                        use_cache=self.use_cache,
                    )
                oidx = oidx + 1

        return j_metrics_res, f_metrics_res

    def _evaluate_unsupervised_single(self, seq, results, all_void_masks, metric):
        seq_name = list(seq.keys())[0]
        seq = seq[seq_name]

        objs = list(seq.keys())
        j_metrics_res = dict()
        f_metrics_res = dict()
        for oi in range(len(objs)):
            _obj = objs[oi]
            _frame_num = seq[_obj]["frame_range"]["frame_nums"]
            j_metrics_res[str(_obj)] = np.zeros((1, int(_frame_num)))
            f_metrics_res[str(_obj)] = np.zeros((1, int(_frame_num)))

        for oi in range(len(objs)):
            _obj = objs[oi]
            _frame_num = seq[_obj]["frame_range"]["frame_nums"]
            start_frame = seq[_obj]["frame_range"]["start"]
            end_frame = seq[_obj]["frame_range"]["end"]

            oidx = 0
            for ii in range(int(start_frame), int(end_frame), 5):
                gt_mask, _ = self.dataset.get_mask(seq_name, "{0:08d}".format(ii), _obj)
                res_mask = results.read_mask_salient(seq_name, "{0:08d}".format(ii))
                if "J" in metric:
                    j_metrics_res[str(_obj)][0, oidx - 1] = db_eval_iou(
                        gt_mask, res_mask, all_void_masks
                    )
                if "F" in metric:
                    f_metrics_res[str(_obj)][0, oidx - 1] = db_eval_boundary(
                        gt_mask,
                        res_mask,
                        all_void_masks,
                        video_name=seq_name,
                        frame_name="{0:08d}".format(ii),
                        eval_obj=_obj,
                        use_cache=self.use_cache,
                    )
                oidx = oidx + 1

        return j_metrics_res, f_metrics_res

    def _evaluate_unsupervised_multiple(
        self, seq, results, all_void_masks, metric, max_n_proposals=20
    ):
        seq_name = list(seq.keys())[0]
        seq = seq[seq_name]

        objs = list(seq.keys())
        j_metrics_res_all = dict()
        f_metrics_res_all = dict()
        j_metrics_res = dict()
        f_metrics_res = dict()
        all_metrics = np.zeros((max_n_proposals, len(objs)))

        for oi in range(len(objs)):
            _obj = objs[oi]
            _frame_num = seq[_obj]["frame_range"]["frame_nums"]
            j_metrics_res_all[str(_obj)] = np.zeros((max_n_proposals, int(_frame_num)))
            f_metrics_res_all[str(_obj)] = np.zeros((max_n_proposals, int(_frame_num)))
        for oi in range(len(objs)):
            _obj = objs[oi]
            _frame_num = seq[_obj]["frame_range"]["frame_nums"]
            start_frame = seq[_obj]["frame_range"]["start"]
            end_frame = seq[_obj]["frame_range"]["end"]

            oidx = 0
            for ii in range(int(start_frame), int(end_frame), 5):
                gt_mask, _ = self.dataset.get_mask(seq_name, "{0:08d}".format(ii), _obj)
                res_mask_all = results.read_mask_seperate(
                    seq_name, "{0:08d}".format(ii), max_n_proposals
                )
                for pi in range(max_n_proposals):
                    res_mask = res_mask_all[0, pi][None,]
                    if "J" in metric:
                        j_metrics_res_all[str(_obj)][pi, oidx] = db_eval_iou(
                            gt_mask, res_mask, all_void_masks
                        )
                    if "F" in metric:
                        f_metrics_res_all[str(_obj)][pi, oidx] = db_eval_boundary(
                            gt_mask,
                            res_mask,
                            all_void_masks,
                            video_name=seq_name,
                            frame_name="{0:08d}".format(ii),
                            eval_obj=_obj,
                            use_cache=self.use_cache,
                        )
                oidx = oidx + 1
        for oi in range(len(objs)):
            if "J" in metric and "F" in metric:
                all_metrics[:, oi] = (
                    np.mean(j_metrics_res_all[str(objs[oi])], axis=1)
                    + np.mean(f_metrics_res_all[str(objs[oi])], axis=1)
                ) / 2
            else:
                all_metrics[:, oi] = (
                    np.mean(j_metrics_res_all[str(objs[oi])], axis=1)
                    if "J" in metric
                    else np.mean(f_metrics_res_all[str(objs[oi])], axis=1)
                )
        row_ind, col_ind = linear_sum_assignment(-all_metrics)
        for oi in range(len(objs)):
            _obj = objs[oi]
            j_metrics_res[str(_obj)] = j_metrics_res_all[str(objs[col_ind[oi]])][
                row_ind[oi],
            ][
                None,
            ]
            f_metrics_res[str(_obj)] = f_metrics_res_all[str(objs[col_ind[oi]])][
                row_ind[oi],
            ][
                None,
            ]

        return j_metrics_res, f_metrics_res

    def _evaluate(self, seq):
        seq = self.dataset.get_sequence(seq)

        _seq_name = list(seq.keys())[0]
        objs = list(seq[_seq_name])
        if self.task == "semi-supervised":
            j_metrics_res, f_metrics_res = self._evaluate_semisupervised(
                seq, self.results, None, self.metric
            )
        elif self.task == "unsupervised_multiple":
            j_metrics_res, f_metrics_res = self._evaluate_unsupervised_multiple(
                seq, self.results, None, self.metric
            )
        elif self.task == "unsupervised_single":
            j_metrics_res, f_metrics_res = self._evaluate_unsupervised_single(
                seq, self.results, None, self.metric
            )

        for ii in range(len(objs)):
            _obj = objs[ii]
            seq_name = f"{_seq_name}_{ii+1}"
            is_unseen = False
            if _seq_name in self.unseen_videos:
                is_unseen = True
            if "J" in self.metric:
                [JM, JR, JD] = utils.db_statistics(j_metrics_res[str(_obj)])
                self.metrics_res["J"]["M"].append(JM)
                self.metrics_res["J"]["R"].append(JR)
                self.metrics_res["J"]["D"].append(JD)

                self.metrics_res["J"]["M_per_object"][seq_name] = JM

                if is_unseen:
                    self.pmetrics_res_unseen["J"]["M"].append(JM)
                    self.pmetrics_res_unseen["J"]["R"].append(JR)
                    self.pmetrics_res_unseen["J"]["D"].append(JD)

                    self.pmetrics_res_unseen["J"]["M_per_object"][seq_name] = JM

                else:
                    self.pmetrics_res_seen["J"]["M"].append(JM)
                    self.pmetrics_res_seen["J"]["R"].append(JR)
                    self.pmetrics_res_seen["J"]["D"].append(JD)

                    self.pmetrics_res_seen["J"]["M_per_object"][seq_name] = JM

            if "F" in self.metric:
                [FM, FR, FD] = utils.db_statistics(f_metrics_res[str(_obj)])
                self.metrics_res["F"]["M"].append(FM)
                self.metrics_res["F"]["R"].append(FR)
                self.metrics_res["F"]["D"].append(FD)

                self.metrics_res["F"]["M_per_object"][seq_name] = FM

                if is_unseen:
                    self.pmetrics_res_unseen["F"]["M"].append(FM)
                    self.pmetrics_res_unseen["F"]["R"].append(FR)
                    self.pmetrics_res_unseen["F"]["D"].append(FD)

                    self.pmetrics_res_unseen["F"]["M_per_object"][seq_name] = FM

                else:
                    self.pmetrics_res_seen["F"]["M"].append(FM)
                    self.pmetrics_res_seen["F"]["R"].append(FR)
                    self.pmetrics_res_seen["F"]["D"].append(FD)

                    self.pmetrics_res_seen["F"]["M_per_object"][seq_name] = FM

            if "V" in self.metric and "J" in self.metric and "F" in self.metric:
                VM = utils.db_statistics_var(
                    j_metrics_res[str(_obj)], f_metrics_res[str(_obj)]
                )
                self.metrics_res["V"]["M"] = VM
                self.metrics_res["V"]["M_per_object"][seq_name] = VM

                if is_unseen:
                    self.pmetrics_res_unseen["V"]["M"].append(VM)
                    self.pmetrics_res_unseen["V"]["M_per_object"][seq_name] = VM
                else:
                    self.pmetrics_res_seen["V"]["M"].append(VM)
                    self.pmetrics_res_unseen["V"]["M_per_object"][seq_name] = VM

        self.pbar.update()

    def adjust(self):
        if "J" in self.metric:
            self.pmetrics_res["J"]["M"] = self.metrics_res["J"]["M"]
            self.pmetrics_res["J"]["R"] = self.metrics_res["J"]["R"]
            self.pmetrics_res["J"]["D"] = self.metrics_res["J"]["D"]
        if "F" in self.metric:
            self.pmetrics_res["F"]["M"] = self.metrics_res["F"]["M"]
            self.pmetrics_res["F"]["R"] = self.metrics_res["F"]["R"]
            self.pmetrics_res["F"]["D"] = self.metrics_res["F"]["D"]
        if "V" in self.metric:
            self.pmetrics_res["V"]["M"] = self.metrics_res["V"]["M"]
        for seq in self.eval_sequeneces:
            seq = self.dataset.get_sequence(seq)
            _seq_name = list(seq.keys())[0]
            objs = list(seq[_seq_name])
            for ii in range(len(objs)):
                _obj = objs[ii]
                seq_name = f"{_seq_name}_{ii+1}"
                if "J" in self.metric:
                    self.pmetrics_res["J"]["M_per_object"][seq_name] = self.metrics_res[
                        "J"
                    ]["M_per_object"][seq_name]
                if "F" in self.metric:
                    self.pmetrics_res["F"]["M_per_object"][seq_name] = self.metrics_res[
                        "F"
                    ]["M_per_object"][seq_name]
                if "V" in self.metric:
                    self.pmetrics_res["V"]["M_per_object"][seq_name] = self.metrics_res[
                        "V"
                    ]["M_per_object"][seq_name]

    def evaluate(self, res_path, metric=("J", "F", "V"), debug=False):
        metric = (
            metric
            if isinstance(metric, tuple) or isinstance(metric, list)
            else [metric]
        )
        if "T" in metric:
            raise ValueError("Temporal metric not supported!")
        if "J" not in metric and "F" not in metric:
            raise ValueError("Metric possible values are J for IoU or F for Boundary")

        # Containers
        self.metrics_res = dict()
        self.pmetrics_res = dict()
        self.pmetrics_res_seen = dict()
        self.pmetrics_res_unseen = dict()
        if "J" in metric:
            self.metrics_res["J"] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            self.pmetrics_res["J"] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            self.pmetrics_res_seen["J"] = {
                "M": [],
                "R": [],
                "D": [],
                "M_per_object": {},
            }
            self.pmetrics_res_unseen["J"] = {
                "M": [],
                "R": [],
                "D": [],
                "M_per_object": {},
            }
        if "F" in metric:
            self.metrics_res["F"] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            self.pmetrics_res["F"] = {"M": [], "R": [], "D": [], "M_per_object": {}}
            self.pmetrics_res_seen["F"] = {
                "M": [],
                "R": [],
                "D": [],
                "M_per_object": {},
            }
            self.pmetrics_res_unseen["F"] = {
                "M": [],
                "R": [],
                "D": [],
                "M_per_object": {},
            }
        if "V" in metric:
            self.metrics_res["V"] = {"M": [], "M_per_object": {}}
            self.pmetrics_res["V"] = {"M": [], "M_per_object": {}}
            self.pmetrics_res_seen["V"] = {"M": [], "M_per_object": {}}
            self.pmetrics_res_unseen["V"] = {"M": [], "M_per_object": {}}

        # Sweep all sequences
        self.results = Results(root_dir=res_path)
        self.metric = metric
        if self.task == "semi-supervised":
            eval_sequeneces = list(self.dataset.get_sequences())
        elif self.task == "unsupervised_multiple":
            eval_sequeneces = list(self.dataset.get_sequences())
        elif self.task == "unsupervised_single":
            eval_sequeneces = list(self.unsup_videos)
        else:
            raise NotImplementedError("Unknown task.")
        self.eval_sequeneces = eval_sequeneces
        self.pbar = tqdm(total=len(eval_sequeneces))
        self.pbar.set_description("Eval Long-Term VOS")

        with ThreadPoolExecutor(max_workers=self.mp_procs) as pool:
            pool.map(self._evaluate, eval_sequeneces)

        self.adjust()
        return self.pmetrics_res, self.pmetrics_res_seen, self.pmetrics_res_unseen
