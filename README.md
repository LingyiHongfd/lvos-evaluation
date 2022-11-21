# LVOS Semi-supervised evaluation package

This package is used to evaluate semi-supervised long-term video multi-object segmentation models for the <a href="https://lingyihongfd.github.io/lvos.github.io/" target="_blank">LVOS</a> dataset.

This tool is also used to evaluate the submissions in the Codalab site for the <a href="https://codalab.lisn.upsaclay.fr/competitions/8767" target="_blank">Semi-supervised LVOS Challenge</a>.

### Installation

```bash
# Download the code
git clone https://github.com/LingyiHongfd/lvos-evaluation.git && cd lvos-evaluation
# Install it - Python 3.6 or higher required
python setup.py install
```

If you don't want to specify the DAVIS path every time, you can modify the default value in the variable `default_lvos_path` in `evaluation_method.py`(the following examples assume that you have set it).
Otherwise, you can specify the path in every call using using the flag `--lvos_path /path/to/LVOS` when calling `evaluation_method.py`.

Once the evaluation has finished, two different CSV files will be generated inside the folder with the results:

- `global_results.csv` contains the overall results.
- `per-sequence_results.csv` contain the per sequence.

If a folder that contains the previous files is evaluated again, the results will be read from the CSV files instead of recomputing them.

## Evaluate LVOS Semi-supervised

In order to evaluate your semi-supervised method in LVOS, execute the following command substituting `results/semi-supervised/ddmemory` by the folder path that contains your results:

```bash
python evaluation_method.py --task semi-supervised --results_path results/semi-supervised/ddmemory
```

The semi-supervised results have been generated using DDMemory.

For some reason, the result of DDMemory is unavailable temporarily. So we provide the result of <a href="https://github.com/yoxu515/aot-benchmark" target="_blank"> AOT-T </a> as an alternative. You can download the result <a href="https://drive.google.com/drive/folders/1bGbyNUdbvmQBBezVv_3Fp-5LITMsY2EG?usp=share_link" target="_blank"> here </a> and unzip the file. After putting the unziped file under the folder `results/semi-supervised/aott`, please use the following command to evaluate AOT-T result.

```bash
python evaluation_method.py --task semi-supervised --results_path results/semi-supervised/aott
```

## Acknowledgement

The codes are modified from <a href="https://github.com/davisvideochallenge/davis2017-evaluation"> DAVIS 2017 Semi-supervised and Unsupervised evaluation package</a>.

## Citation

Please cite both papers in your publications if LVOS or this code helps your research.

```latex
@article{Caelles_arXiv_2019,
  author = {Sergi Caelles and Jordi Pont-Tuset and Federico Perazzi and Alberto Montes and Kevis-Kokitsi Maninis and Luc {Van Gool}},
  title = {The 2019 DAVIS Challenge on VOS: Unsupervised Multi-Object Segmentation},
  journal = {arXiv},
  year = {2019}
}
```
