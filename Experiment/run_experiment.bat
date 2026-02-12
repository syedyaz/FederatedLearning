@echo off
REM Run CIFAR-10 Federated Learning Experiment (Windows)

echo FedEdge-Accel Experiment Runner
echo ================================

REM Check if datasets are prepared
if not exist "data\cifar10" (
    echo Preparing CIFAR-10 dataset...
    python scripts\prepare_datasets.py --dataset cifar10
)

REM Run experiment
echo Running CIFAR-10 experiment...
python experiments\cifar10_experiment.py

REM Generate plots (find latest results directory)
echo Generating plots...
for /f "delims=" %%i in ('dir /b /ad /o-d results\cifar10_* 2^>nul') do (
    python scripts\generate_plots.py --results_dir "results\%%i" --output_dir plots\
    echo Results saved to: results\%%i
    echo Plots saved to: plots\
    goto :done
)

:done
echo Experiment complete!
pause
