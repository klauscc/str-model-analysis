## Scene Text Recognition: an Analysis of Sequence to Sequence Model and Parallel Decoding Model

Official Tensorflow implementation for our paper.

#### Pretrained models and results: 

| [Pretrained Models](https://drive.google.com/drive/folders/1WzsLBl5Ex-7QmIjrxkAPghLqQeR7ydk4?usp=sharing) | IIIT5K | SVT  | IC03 | IC13 | IC15 | SVTP | CUTE |
|-----------------------------------------------------------------------------------------------------------|--------|------|------|------|------|------|------|
| PD-PVAM                                                                                                   | 90.2   | 88.4 | 96.2 | 94.6 | 81.9 | 85.0 | 83.3 |
| Seq2seq-Trans                                                                                             | 90.8   | 92.4 | 97.0 | 96.4 | 83.7 | 89.1 | 86.5 |
| E-CAM-Seq                                                                                                 | 91.9   | 92.9 | 97.6 | 96.8 | 84.2 | 89.6 | 87.2 |

### 1. project structure

```
├── str  # scene text recognition
│   ├── common/    # common libs
│   ├── ctc/       # ctc model
│   ├── parallel_decoder/   # pd models
│   └── transformer/        # transformer models
└── tfbp # personal toolbox in progress
    └── tfbp
        ├── callbacks
        ├── tests
        └── train
```

clone the repo:

```
git clone --recurse-submodules git@github.com:klauscc/str-model-analysis.git
# or with https
git clone --recurse-submodules git@github.com:klauscc/str-model-analysis.git
```


### 2. Dependencies

- tensorflow==2.2.0
- tensorflow_io==0.14.0
- opencv-python==4.2.0.32
- opencv-contrib-python==4.4.0.44
- lmdb==0.98

you may install with `pip install tensorflow_gpu==2.2.0 tensorflow_io==0.14.0 opencv-python==4.2.0.32 opencv-contrib-python==4.4.0.44 lmdb==0.98`


### 3. Datasets

The datasets used for training, validation and test are adopted from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

Download lmdb dataset for traininig and evaluation from [their shared google drive](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt).

data_lmdb_release.zip contains below. <br>
training datasets : [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/)[1] and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)[2] \
validation datasets : the union of the training sets [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6].\
evaluation datasets : benchmark evaluation datasets, consist of [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[5], [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)[6], [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)[7], [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4)[4], [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf)[8], and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html)[9].

### 4. Models

Currently implemented models:

```
├── str  # scene text recognition
    ├── ctc/       # ctc model
    ├── parallel_decoder/   # pd models
    └── transformer/        # transformer models
  
```

<font color='red'>Important:</font> Before traing/evaluation, you need to run the script to add PYTHONPATH:

```
# in the project root dir.
source prepare_env.sh
```

#### 4.1 E-CAM-Seq

Transformer with ealy fusion CAM.

- Train:

```
# in the project root dir.
python -m str.transformer.train \
    --workspace /path/to/your/workspace \
    --config str/transformer/config_large.yml \
    --gpu_id 0,1,2,3 \
    --label_smooth 0.1 \
    --with_parallel_visual_attention True \
    --pd_method dan_cam \
    --pd_early_fusion True \
    --pd_late_fusion False \
```

- Evaluate:

```
# in the project root dir.
python -m str.transformer.evaluate \
    --workspace /path/to/your/workspace \
    --config str/transformer/config_large.yml \
    --label_smooth 0.1 \
    --with_parallel_visual_attention True \
    --pd_method dan_cam \
    --pd_early_fusion True \
    --pd_late_fusion: False \
    --mode eval \
    --beam_size 5 \
    --load_ckpt path/to/your/checkpoint \
```

- **Note**:
    - The learning rate and schedules are tuned based on training on 4 GPUs with total batch size 256. You may need to re-tune these parameters if you use a different batch size.

##### Arguments:

**All the Arguments are defined in the `str/transformer/config_large.yml`. You can override them by specifying them in the training scripts.**

For example, in the `.../config_large.yml`, you have `label_smooth 0`; in the training script, `--label_smooth 0.1`. The arguments in the training script will take effect.

* --workspace: The path where the logs and checkpoints will be saved. Auto resume training
* --db_dir: training set path. <font color='red'> You must modify the path in `config_large.yml` or override it in the training script in order to find the dataset.</font>
* --test_db_dir: validation set path. <font color='red'> You must modify the path in `config_large.yml` or override it in the training script in order to find the dataset.</font>
* --eval_db_dir: evaluation(test) set path.<font color='red'> You must modify the path in `config_large.yml` or override it in the training script in order to find the dataset.</font>
* --encode_training_mode: Whether to freeze BN mean/var. Options: 
    * `encoder_only`:  Freeze the BN mean/var to 0/1. The BN's gamma/beta will be trainable.
    * `all`: Train the BN normally.
* --with_image_transform: Whether to apply perspective transform augmentation. `True` or `False`.
* --with_tps: Whether to apply TPS transform augmentation. `True` or `False`.
* --with_parallel_visual_attention: Whether to fuse Transformer with PD-models. If `False`, the model is `Seq2seq Trans`.
* --pd_method: The hint proposal module. Options: `semantic_reasoning` (PVAM) and `dan_cam` (CAM).
* --pd_early_fusion: Whether to perform early fusion. `True` or `False`.
* --pd_late_fusion:  Whether to perform late fusion. `True` or `False`.

#### 4.2 Seq2seq-Trans

Transformer-based model.

- Train:

```
# in the project root dir.
python -m str.transformer.train \
    --workspace /path/to/your/workspace \
    --config str/transformer/config_large.yml \
    --gpu_id 0,1,2,3 \
    --label_smooth 0.1 \
```

- Evaluate:

```
# in the project root dir.
python -m str.transformer.evaluate \
    --workspace /path/to/your/workspace \
    --config str/transformer/config_large.yml \
    --label_smooth 0.1 \
    --mode eval \
    --beam_size 5 \
    --load_ckpt path/to/your/checkpoint \
```

#### 4.3 Parallel Decoding models

- Train:

```
python -m str.parallel_decoder.train \
    --workspace  $workspace \
    --config str/parallel_decoder/config_large.yml \
    --voc_type LOWERCASE \
    --gpu_id 1,2,3,4 \
    --encode_training_mode encoder_only  \
    --label_smooth 0.1 \
```

- Evaluate:

```
python -m str.parallel_decoder.train \
    --workspace  $workspace \
    --config str/parallel_decoder/config_large.yml \
    --voc_type LOWERCASE \
    --encode_training_mode encoder_only  \
    --label_smooth 0.1 \
    --mode eval \
    --load_ckpt path/to/your/ckpt \
```
