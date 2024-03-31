The following steps need to be followed to perform a performance evaluation.

#### (1) Train teacher models by fine-tuning and grid search according to the original Bert paper.
This requires three arguments: task_name is the dataset name in the GLUE benchmark, metric_name is the metric name for the task and the model_name is the model backbone in [bert-base-uncased, bert-large-uncased]:
    
    'QNLI': 'eval_accuracy',
    'RTE': 'eval_accuracy',
    'CoLA': 'eval_matthews_correlation',
    'STSB': 'eval_combined_score',
    'MRPC': 'eval_f1',
    
For instance, to train a BERT base teacher in RTE:

    python train_teacher.py --task_name RTE --metric_name eval_accuracy --model_name bert-base-uncased
    
The output model can be found in ./tmp/{exp_name}/{task_name}/{model_name}. Please manually move the best model to ~/checkpoints/exp/[task_name] for further training (i.e. baseline, SecFormer).

#### (2) Train baseline models with approximations
To train a baseline with gelu and 2quad approximation on RTE, run:

    python baseline.py --task_name RTE --model_path ~/checkpoints/exp/RTE --baseline_type S1 --hidden_act gelu --softmax_act 2quad

#### (3) Run distillation process
Main training scripts of SecFormer. It uses a minimized implementation of Transformer in this directory (i.e. changes in the outmost directory will not affect the behavior here). To reproduce results:

(1) Download Glue data using download_glue_data.py

(2) train a teacher model (pretraining + fine-tuning on downstream tasks) and put in ~/checkpoints/exp/[task_name]. We provide our script in ../secformer/text-classification/train_teacher.py.

(3) run SecFormer distillation process, e.g. for STSB and gelu+2quad approximation run:

    python exp2.py --task_name STSB --teacher_dir ~/secformer/checkpoints/exp/STSB --student_dir ~/secformer/checkpoints/exp/STSB --hidden_act gelu --softmax_act 2quad

Potential issues:

(1) In case of dataset minor mismatch, such as "STS-B" and "STSB", please change the data dir as written in task_distill.py. This is due to different naming convention
    of HuggingFace and other Repos.

(2) Hyper-parameters can be overwrite in task_distill.py.