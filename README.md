# Introduction

This replication package contains the code and instructions necessary to reproduce the results for the paper _Idioms: Neural Decompilation With Joint Code and Type Prediction_.

There are five main sections:
- Building the realtype dataset
- Adapting the exebench dataset 
- Training
- Evaluation
- LLM4Decompile Baseline

The datasets and trained models and adapters are available [here](https://zenodo.org/records/14797017).

With the provided datasets, you can skip to the Training section.
With the provided datasets and pretrained models/adapters, you can skip to the Evaluation section.
The evaluator script assumes the models are in a folder named `runs` rather than `models` so if you plan on rerunning the evaluation, run `mv models runs`.

All sections make use of the `idioms` conda environment. It can be installed with:
```
conda env create -f env.yml
```
In addition, some sections require installing additional packages or supplementary conda environments.

# Building the Realtype Dataset

The first steps for building the dataset involve the ghcc-idioms directory, a modified version of the [original GHCC](https://github.com/huzecong/ghcc).

First, build set up the required python environment, which differs from the `idioms` python environment.
```
cd ghcc-idioms
conda env create -f env.yml
conda activate ghcc
```

For safe building of binaries, ghcc provides a docker container.
To build it, run
```
docker build -t gcc-custom .
```

To build the binaries for the dataset, run
```
python main.py --repo-list-file repo_list.txt --n-procs 5
```
You may also download the binaries [here](https://huggingface.co/datasets/ejschwartz/idioms-realtype-binaries).


To preprocess the code, run
```
python extract_original.py binaries/ archives/ preprocessed
```
The latter two are very expensive commands, and may take up to several weeks to run.

We use scripts from DIRTY [1] to do the decompilation. For our main experiments, we used the original version which uses the proprietary Hex-Rays decompiler. There is also a version based on the open-source Ghidra decompiler. We use this for our baselining experiments.

Clone these repositories with
```
git clone https://github.com/CMUSTRUDEL/DIRTY.git
cd DIRTY
git checkout 5f8e9494d52ef8788f0834342097f36462eeb2eb
cd ..

git clone https://github.com/edmcman/DIRTY-Ghidra.git
cd DIRTY-Ghidra
git checkout 82aadac50fab7e8842da6d2fd356bcc088f34994
cd ..
```

DIRTY and DIRTY-Ghidra require their own python environments. We provide an environment file for both of them. To use it, run
```
conda env create -f dirty_env.yml 
conda activate dirty
```

The relevant script is `generate.py`, which can be found in `DIRTY/dataset-gen` and `DIRTY-Ghidra/dataset-gen-ghidra`

The commands to run decompilation with Hex-Rays and Ghidra respectively are
```
python generate.py --ida /path/to/idat64 -b /path/to/binaries/ -o /path/to/decompiled-hex-rays

python generate.py --ghidra /path/to/ghidra_11.2.1_PUBLIC/support/analyzeHeadless -b /path/to/binaries/ -o /path/to/decompiled-ghidra
```
The decomiled-hex-rays and decompiled-ghidra folders will be created by the script.

All subsequent steps use the `idioms` conda environment.
```
conda activate idioms
```


Duplicate projects are common in open-source projects, and failing to remove them can lead to data leakage. Find duplicates with
```
python deduplicate.py /path/to/archives/ dedup-clusters.json --lexical --workers 8
```

After these steps, there is enough information to build the full realtype dataset. To do this, run
```
python prepare.py /path/to/decompiled-hex-rays /path/to/binaries/ /path/to/preprocessed/ dedup-clusters.json idioms_dataset --holdout-set-size 0.02 --valid-max-bins-per-repo 1 --test-max-bins-per-repo 25 --shard-size 500
```

# Adapting the Exebench Dataset

The exebench dataset contains source code and the corresponding assembly, but no decompiled code.
To add it, we first compile each example.
```
conda activate idioms
mkdir exebench
python compile_exebench.py exebench/compiled-O0 --workers 8
```

Next, we use DIRTY and DIRTY-GHIDRA to decompile each partition.
```
cp exebench_decompile/gen-exebench.py /path/to/DIRTY/dataset-gen
cp exebench_decompile/gen-exebench.py /path/to/DIRTY-Ghidra/dataset-gen-ghidra
mkdir exebench

conda activate dirty

cd /path/to/DIRTY/dataset-gen
python gen-exebench.py /path/to/exebench/ O0 --decompiler /path/to/ida-7.5/idat64

cd /path/to/DIRTY-Ghidra/dataset-gen-ghidra
python gen-exebench.py /path/to/exebench/ O0 --decompiler /path/to/ghidra_11.2.1_PUBLIC/support/analyzeHeadless --eval-only
```

With the code decompiled, build the dataset. There are two different forms of the dataset that are used in the experiments. For convenience, we move the datasets to the current directory for the experiments.
```
conda activate idioms
python update_exebench.py exebench idioms O0
python update_exebench.py exebench huggingface O0 --eval-only

mv exebench/exebench-idioms-O0-hex-rays exebench-idioms-O0
mv exebench/exebench-hf-O0-eval/ .
```

The `parity-exebench` experiments require a subsample of the dataset that is the same size as the realtype dataset.
To create this, run
```
python downsample_exebench.py exebench-idioms-O0/ idioms_dataset/ parity-exebench-idioms-O0
```


# Training

Model training requires the `unsloth` package, which both accelerates training and reduces GPU memory consumption.
Unsloth is installed in a conda environment but separately from the environment file.
We used
```
conda activate idioms
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```
Unsloth is only fully supported on GPUs with compute capability 7.0 or greater.
For more information on using unsloth, see [the pypi page](https://pypi.org/project/unsloth/).


To train the models, do the following:

For the CodeQwen2.5 experiments

```
python train.py exebench-idioms-O0 qwen-0.5b-exebench-O0 function finetune --epochs 8.0 --model-type Qwen/Qwen2.5-Coder-0.5B --batch-size 8 --gradient-accumulation 8
python train.py parity-exebench-idioms-O0 qwen-0.5b-parity-exebench-O0 function finetune --epochs 8.0 --model-type Qwen/Qwen2.5-Coder-0.5B --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset qwen-0.5b-functions-idioms function finetune --epochs 8.0 --model-type Qwen/Qwen2.5-Coder-0.5B --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset qwen-0.5b-neighbors-idioms neighbors finetune --epochs 8.0 --model-type Qwen/Qwen2.5-Coder-0.5B --batch-size 4 --gradient-accumulation 16 --nhops 4 --max-length 4096

```

For the modified LLM4Decompile experiments

```
python train.py exebench-idioms-O0 llm4decompile-1.3b-exebench-O0 function adapter --epochs 2.0 --model-type LLM4Binary/llm4decompile-1.3b-v2 --batch-size 8 --gradient-accumulation 8
python train.py parity-exebench-idioms-O0 llm4decompile-1.3b-parity-exebench-O0 function adapter --epochs 8.0 --model-type LLM4Binary/llm4decompile-1.3b-v2 --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset llm4decompile-1.3b-functions-idioms function adapter --epochs 8.0 --model-type LLM4Binary/llm4decompile-1.3b-v2 --batch-size 16 --gradient-accumulation 4
python train.py idioms_dataset llm4decompile-1.3b-neighbors-idioms neighbors adapter --epochs 8.0 --model-type LLM4Binary/llm4decompile-1.3b-v2 --batch-size 16 --gradient-accumulation 4 --nhops 4 --max-length 4096

```

For the CodeGemma-2b experiments

```
python train.py exebench-idioms-O0 codegemma-2b-exebench-O0 function adapter --epochs 2.0 --model-type unsloth/codegemma-2b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py parity-exebench-idioms-O0 codegemma-2b-parity-exebench-O0 function adapter --epochs 8.0 --model-type unsloth/codegemma-2b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset codegemma-2b-functions-idioms function adapter --epochs 8.0 --model-type unsloth/codegemma-2b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset codegemma-2b-neighbors-idioms neighbors adapter --epochs 8.0 --model-type unsloth/codegemma-2b-bnb-4bit --batch-size 4 --gradient-accumulation 16 --nhops 4 --max-length 4096

```


For the CodeGemma-7b experiments

```
python train.py exebench-idioms-O0 codegemma-7b-exebench-O0 function adapter --epochs 1.0 --model-type unsloth/codegemma-7b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py parity-exebench-idioms-O0 codegemma-7b-parity-exebench-O0 function adapter --epochs 8.0 --model-type unsloth/codegemma-7b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset codegemma-7b-functions-idioms function adapter --epochs 8.0 --model-type unsloth/codegemma-7b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset codegemma-7b-neighbors-idioms neighbors adapter --epochs 8.0 --model-type unsloth/codegemma-7b-bnb-4bit --batch-size 4 --gradient-accumulation 16 --nhops 4 --max-length 4096
```

For the CodeLlama-7b experiments

```
python train.py exebench-idioms-O0 codellama-7b-exebench-O0 function adapter --epochs 1.0 --model-type unsloth/codellama-7b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py parity-exebench-idioms-O0 codellama-7b-parity-exebench-O0 function adapter --epochs 8.0 --model-type unsloth/codellama-7b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset codellama-7b-functions-idioms function adapter --epochs 8.0 --model-type unsloth/codellama-7b-bnb-4bit --batch-size 8 --gradient-accumulation 8
python train.py idioms_dataset codellama-7b-neighbors-idioms neighbors adapter --epochs 8.0 --model-type unsloth/codellama-7b-bnb-4bit --batch-size 4 --gradient-accumulation 16 --nhops 4 --max-length 4096
```

# Evaluation

Evaluation requires the `codealign` package for computing most metrics. To install, run
```
git clone https://github.com/squaresLab/codealign.git
cd codealign
pip install .
cd ..
```

Most experiments are trained to 8 epochs, but it's possible that some models will start to overfit before then.
To account for this, run validation on each model that has eight or more epochs of training using
```
python eval_all.py
```

The `eval_all.py` script has arguments to filter the training runs for which validation is performed.
This allows the validation runs to be divided up and run in parallel.

With all necessary runs validated, run
```
python compare_checkpoints.py
```
This will output the results for each validation experiment in a tabular form.
It will also write a json file to `results/` describing the checkpoint that performed best on the validation set for each model/experimental conditions combo.

With this, test the best checkpoints by running
```
python eval_all.py --test-best-checkpoints
```
This does not include runs for which selecting a best checkpoint is unnecessary (because those runs were only trained for 1 or 2 epochs).
To run evaluation in these four cases, run:
```
python evaluator.py runs/llm4decompile-1.3b-exebench-O0/ --eval-partition test --batch-size 32 --dataset exebench-hf-O0-eval --exebench-subpartition real --no-exebench-tests
python evaluator.py runs/codegemma-2b-exebench-O0/ --eval-partition test --batch-size 32 --dataset exebench-hf-O0-eval --exebench-subpartition real --no-exebench-tests
python evaluator.py runs/codegemma-7b-exebench-O0/ --eval-partition test --batch-size 32 --dataset exebench-hf-O0-eval --exebench-subpartition real --no-exebench-tests
python evaluator.py runs/codellama-7b-exebench-O0/ --eval-partition test --batch-size 32 --dataset exebench-hf-O0-eval --exebench-subpartition real --no-exebench-tests
```

Models evaluated on the exebench test run also need their unit tests to be run; the fraction of runs which pass all of the unit tests is an additional metric.
Running the unit tests tooks around 6 hours per model on our machine vs around a minute
for all other metrics combined, so `eval_all.py` skips running them.
Running arbitrary machine-learning-generated code can be dangerous, so we provide a docker image for containers in which the tests are actually run.
To build the image and compute exebench metrics, run
```
docker build -t exebench-test exebench_docker/
python test_exebench.py 
```

Finally, summarize and display the final results with
```
python visualizer.py run metric
```
If you used the pretrained models rather than training your own, instead run
```
python visualizer.py run metric --checkpoint-type last
```
Note that numbers may vary slightly.

# LLM4Decompile Baseline

We also include the script to perform the LLM4Decopmile baseline.
We use the `exebench-hf-O0-eval` dataset from the "Adapting the Exebench Dataset" section.
The script runs exebench tests, so it requires the docker image described in the "Evaluation" section.
To build that docker image, run
```
docker build -t exebench-test exebench_docker/
```

The main script depends on `clang-format`.
LLM4Decompile by default is trained on Ghidra decompilation as input, so we provide that here rather than the Hex-Rays decompilation from the original version of DIRTY.
DIRTY-Ghidra stores the decompiled code in a manner with whitespace stripped, which affects the tokenization.
LLM4Decopmile was trained on formatted code.
The reformatting the code provides a small but noticable accuracy boost.

To run the LLM4Decompile experiment run
```
python llm4decompile.py LLM4Binary/llm4decompile-1.3b-v2 exebench-hf-O0-eval --dataset-split test_real
```

The results will be in `baselines/llm4decompile-1.3b-v2/exebench-hf-O0-eval/test_real_scores.json`.

# Reference

[1] Chen, Qibin, et al. "Augmenting decompiler output with learned variable names and types." 31st USENIX Security Symposium (USENIX Security 22). 2022.
