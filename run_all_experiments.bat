@echo off
REM Enhanced batch script for running multiple GNN experiments

REM Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%

REM Create experiment directories
mkdir experiments 2>nul
mkdir experiments\logs 2>nul
mkdir experiments\models 2>nul

REM 1. Run experiments with different cancer types
echo Running experiments with different cancer types...
python train.py --cancer_type BRCA --experiment_name "BRCA_baseline"
python train.py --cancer_type LUAD --experiment_name "LUAD_baseline"
python train.py --cancer_type LUSC --experiment_name "LUSC_baseline"

REM 2. Run experiments with different model architectures
echo Running experiments with different model architectures...
python train.py --model GCN --hidden_channels 128 --num_layers 4 --experiment_name "GCN_deep"
python train.py --model GraphSAGE --hidden_channels 256 --num_layers 6 --experiment_name "GraphSAGE_wide"
python train.py --model GAT --hidden_channels 64 --num_heads 8 --experiment_name "GAT_attention"

REM 3. Run experiments with different hyperparameters
echo Running experiments with different hyperparameters...
python train.py --learning_rate 0.0001 --weight_decay 1e-4 --experiment_name "lr_1e4_wd_1e4"
python train.py --learning_rate 0.01 --weight_decay 1e-6 --experiment_name "lr_1e2_wd_1e6"
python train.py --batch_size 32 --dropout 0.3 --experiment_name "batch32_dropout03"

REM 4. Run cross-validation experiments
echo Running cross-validation experiments...
python train.py --cross_validate --n_splits 5 --experiment_name "5fold_cv"

REM 5. Run ablation studies
echo Running ablation studies...
python train.py --no_attention --experiment_name "no_attention"
python train.py --no_residual --experiment_name "no_residual"
python train.py --no_batch_norm --experiment_name "no_batch_norm"

REM 6. Run experiments with different data preprocessing
echo Running experiments with different data preprocessing...
python train.py --use_smote --experiment_name "with_smote"
python train.py --use_feature_selection --experiment_name "with_feature_selection"
python train.py --use_graph_augmentation --experiment_name "with_graph_aug"

REM 7. Run ensemble experiments
echo Running ensemble experiments...
python train.py --ensemble --experiment_name "ensemble_all_models"

REM 8. Run experiments with different loss functions
echo Running experiments with different loss functions...
python train.py --loss focal --experiment_name "focal_loss"
python train.py --loss weighted --experiment_name "weighted_loss"

REM 9. Run experiments with different optimizers
echo Running experiments with different optimizers...
python train.py --optimizer adamw --experiment_name "adamw_optimizer"
python train.py --optimizer sgd --experiment_name "sgd_optimizer"

REM 10. Run experiments with different learning rate schedulers
echo Running experiments with different learning rate schedulers...
python train.py --scheduler cosine --experiment_name "cosine_scheduler"
python train.py --scheduler step --experiment_name "step_scheduler"

echo All experiments completed! 