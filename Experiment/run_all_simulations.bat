@echo off
setlocal
set PYTHONPATH=.

echo ============================================================
echo FedEdge-Accel: Federated Learning Simulation Runner
echo ============================================================

echo PHASE 1: Preparing Datasets...
python scripts/prepare_datasets.py --dataset cifar10
if %errorlevel% neq 0 (
    echo Error preparing CIFAR-10 dataset.
)

python scripts/prepare_datasets.py --dataset femnist
if %errorlevel% neq 0 (
    echo Error preparing FEMNIST dataset.
)

python scripts/prepare_datasets.py --dataset har
if %errorlevel% neq 0 (
    echo Error preparing UCI HAR dataset.
)

echo.
echo PHASE 2: Running Experiments...
echo.
echo Running FEMNIST Experiment...
python experiments/femnist_experiment.py --config configs/training_config.yaml
if %errorlevel% neq 0 (
    echo Error running FEMNIST experiment.
)

echo.
echo Running UCI HAR Experiment...
python experiments/har_experiment.py --config configs/training_config.yaml
if %errorlevel% neq 0 (
    echo Error running UCI HAR experiment.
)

echo.
echo Running CIFAR-10 Experiment...
python experiments/cifar10_experiment.py --config configs/training_config.yaml
if %errorlevel% neq 0 (
    echo Error running CIFAR-10 experiment.
)

echo.
echo PHASE 3: Generating Plots...
python scripts/generate_plots.py --results_dir results
if %errorlevel% neq 0 (
    echo Error generating plots.
)

echo ============================================================
echo All simulations completed! Check the 'results' and 'plots' 
echo directories for the generated data.
echo ============================================================
pause
endlocal
