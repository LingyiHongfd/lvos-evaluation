# LVOS evaluation package

This package is used to evaluate long-term video object segmentation models for the <a href="https://lingyihongfd.github.io/lvos.github.io/" target="_blank">LVOS</a> dataset.

This tool is also used to evaluate the submissions in the Codalab site for the <a href="https://codalab.lisn.upsaclay.fr/competitions/8767" target="_blank">Semi-supervised LVOS Challenge</a>.

### Installation

```bash
# Download the code
git clone https://github.com/LingyiHongfd/lvos-evaluation.git && cd lvos-evaluation
# Install it - Python 3.6 or higher required
python setup.py install
```

If you don't want to specify the LVOS path every time, you can modify the default value in the variable `default_lvos_path` in `evaluation_method.py`(the following examples assume that you have set it).

Note: `default_lvos_path` is the valid split path.

Otherwise, you can specify the path in every call by using the flag `--lvos_path /path/to/LVOS` when calling `evaluation_method.py`.

Once the evaluation has finished, two different CSV files will be generated inside the folder with the results:

- `global_results.csv` contains the overall results.
- `per-sequence_results.csv` contain the per sequence.

If a folder that contains the previous files is evaluated again, the results will be read from the CSV files instead of recomputing them.

## Evaluate Your VOS Model on LVOS

In order to evaluate your vos method on LVOS , execute the following command substituting `results_path` by the folder path that contains your results:

```bash
python evaluation_method.py --task semi-supervised --results_path results_path --mp_nums 1
```

<!-- For some reason, the result of DDMemory is unavailable temporarily. So we provide the result of <a href="https://github.com/yoxu515/aot-benchmark" target="_blank"> AOT-T </a> as an alternative. You can download the result <a href="https://drive.google.com/drive/folders/1bGbyNUdbvmQBBezVv_3Fp-5LITMsY2EG?usp=share_link" target="_blank"> here </a> and unzip the file. After putting the unziped file under the folder `results/semi-supervised/aott`, please use the following command to evaluate AOT-T result. -->

`mp_nums` is set as 1 by default. Because the score computing process in serial mode is time-consuming, you can set `mp_nums` larger than 1 (such as 2) to enable multiple processing and speed up the evaluation. But we suggest that `mp_nums` should be set to less than 8 on a regular server.

Besides, you can also choose to use multiple process or multiple thread to speed up the evaluation process by set `m_class`. 

To accelerate the evaluation, we also introduce the cache technique and the Cython code. Setting `use_cache` as `True`, our code will automatically cache the intermediate results and uses the cached results in subsequent evaluations to reduce redundant computation. It's worth noting that the cache data requires 1G disk space. Moreover, you can compile the ops to further speed up the evaluation. 

```bash
cd ./lvos/metric_ops/_get_binary_c
python setup.py build_ext --inplace
```

In addition to the semi-supervised vos task, we also support unsupervised vos tasks. You can set `task` as `unsupervised_multiple` or `unsupervised_single`.

Here is an example to evaluate the semi-supervised vos model with 64 processes.

```bash
python evaluation_method.py --task semi-supervised --results_path results_path --mp_nums 64 --m_class mp --use_cache
```

Our code can be used to evaluate both LVOS V1 and LVOS V2 by setting `lvos_path`.





## APIs

We released the tools and test scripts in this <a href="https://github.com/LingyiHongfd/LVOS-api"> repository</a>. Click on this link for more information.

## Acknowledgement

The codes are modified from <a href="https://github.com/davisvideochallenge/davis2017-evaluation"> DAVIS 2017 Semi-supervised and Unsupervised evaluation package</a>.

## Citation

Please cite both papers in your publications if LVOS or this code helps your research.

```latex
# for LVOS V2
@article{hong2024lvos,
  title={LVOS: A Benchmark for Large-scale Long-term Video Object Segmentation},
  author={Hong, Lingyi and Liu, Zhongying and Chen, Wenchao and Tan, Chenzhi and Feng, Yuang and Zhou, Xinyu and Guo, Pinxue and Li, Jinglun and Chen, Zhaoyu and Gao, Shuyong and others},
  journal={arXiv preprint arXiv:2404.19326},
  year={2024}
}
# for LVOS V1
@inproceedings{hong2023lvos,
  title={Lvos: A benchmark for long-term video object segmentation},
  author={Hong, Lingyi and Chen, Wenchao and Liu, Zhongying and Zhang, Wei and Guo, Pinxue and Chen, Zhaoyu and Zhang, Wenqiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13480--13492},
  year={2023}
}
```
