#!/bin/bash

# Script to train and compare different configurations of the digit classifier

# Create a directory for experiment results
mkdir -p experiments
timestamp=$(date +%Y%m%d_%H%M%S)
exp_dir="experiments/run_$timestamp"
mkdir -p $exp_dir

# Function to train model with specific parameters
train_model() {
    config_name=$1
    dropout=$2
    batch_norm=$3
    
    echo "=========================================="
    echo "Training configuration: $config_name"
    echo "Dropout: $dropout, Batch Normalization: $batch_norm"
    echo "=========================================="
    
    # Create directory for this configuration
    config_dir="$exp_dir/$config_name"
    mkdir -p $config_dir
    
    # Prepare command
    cmd="python digit_classifier.py --epochs 5 --batch-size 128 --lr 0.01"
    
    # Add configuration flags
    if [ "$dropout" = "False" ]; then
        cmd="$cmd --no-dropout"
    fi
    
    if [ "$batch_norm" = "False" ]; then
        cmd="$cmd --no-batch-norm"
    fi
    
    # Add CUDA if available
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected, using CUDA"
    else
        echo "No GPU detected, using CPU"
        cmd="$cmd --no-cuda"
    fi
    
    # Save model
    cmd="$cmd --save-model"
    
    # Run the command and save output
    echo "$cmd" > "$config_dir/command.txt"
    TIMEFORMAT='%3R'
    echo "Starting training at $(date)"
    start_time=$(date +%s)
    
    # Execute the command and capture output
    $cmd > "$config_dir/output.log" 2>&1
    
    end_time=$(date +%s)
    training_time=$((end_time - start_time))
    echo "Training completed in $training_time seconds"
    echo "$training_time" > "$config_dir/training_time.txt"
    
    # Move generated images to config directory
    if [ -f "predictions.png" ]; then
        mv predictions.png "$config_dir/"
    fi
    
    if [ -f "training_metrics.png" ]; then
        mv training_metrics.png "$config_dir/"
    fi
    
    # Move saved model if it exists
    if [ -f "mnist_cnn.pt" ]; then
        mv mnist_cnn.pt "$config_dir/"
    fi
}

# Train with different configurations
echo "Starting training experiments at $(date)"

# 1. Base model (with dropout and batch norm)
train_model "full_model" "True" "True"

# 2. No dropout
train_model "no_dropout" "False" "True"

# 3. No batch normalization
train_model "no_batchnorm" "True" "False"

# 4. No dropout and no batch normalization
train_model "baseline" "False" "False"

echo "All experiments completed at $(date)"

# Generate summary report
echo "Generating summary report..."
{
    echo "# Digit Classifier Training Summary"
    echo "Date: $(date)"
    echo ""
    echo "## Training Times"
    echo "| Configuration | Training Time (s) |"
    echo "|---------------|------------------|"
    
    for config in "full_model" "no_dropout" "no_batchnorm" "baseline"; do
        time_file="$exp_dir/$config/training_time.txt"
        if [ -f "$time_file" ]; then
            time_val=$(cat "$time_file")
            echo "| $config | $time_val |"
        else
            echo "| $config | N/A |"
        fi
    done
    
    echo ""
    echo "## Results"
    echo "Review the output.log files in each configuration directory for detailed metrics."
    echo ""
    echo "## Configuration Details"
    echo "- full_model: Dropout + Batch Normalization"
    echo "- no_dropout: Only Batch Normalization"
    echo "- no_batchnorm: Only Dropout"
    echo "- baseline: No Dropout, No Batch Normalization"
} > "$exp_dir/summary.md"

echo "Summary saved to $exp_dir/summary.md"
echo "All experiment results saved to $exp_dir"