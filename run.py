#!/usr/bin/env python
"""
Main execution script for Predictive Maintenance System
Complete pipeline: preprocess → train → evaluate → predict
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE, CHECKPOINTS_DIR, RESULTS_DIR
from src.data_preprocessing import preprocess_dataset, DataPreprocessor
from src.dataset import create_data_loaders
from src.models import create_model
from src.train import Trainer
from src.evaluate import evaluate_model
from src.predict import RULPredictor, create_sample_engine_data
from src.utils import set_random_seed, print_model_info, print_comparison_table


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance System Pipeline"
    )
    
    parser.add_argument(
        'command',
        choices=['preprocess', 'train', 'evaluate', 'predict', 'all'],
        help='Command to execute'
    )
    
    parser.add_argument(
        '--model',
        choices=['lstm', 'transformer', 'both'],
        default='both',
        help='Model to train (default: both)'
    )
    
    parser.add_argument(
        '--dataset',
        choices=['FD001', 'FD002', 'FD003', 'FD004'],
        default='FD001',
        help='Dataset variant (default: FD001)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from checkpoint'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"PREDICTIVE MAINTENANCE SYSTEM")
    print(f"{'='*70}")
    print(f"Command: {args.command}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")
    
    # Set random seed
    set_random_seed()
    
    # ============================================================
    # PREPROCESS
    # ============================================================
    
    if args.command in ['preprocess', 'all']:
        print(f"\n{'='*70}")
        print("STEP 1: DATA PREPROCESSING")
        print(f"{'='*70}\n")
        
        try:
            processed_data = preprocess_dataset(
                dataset_name=args.dataset,
                reload=True
            )
            print(f"\n✓ Data preprocessing complete!")
            print(f"  Train sequences: {processed_data['train_sequences'].shape}")
            print(f"  Test sequences: {processed_data['test_sequences'].shape}")
        except Exception as e:
            print(f"✗ Error during preprocessing: {e}")
            return
    
    # ============================================================
    # TRAIN
    # ============================================================
    
    if args.command in ['train', 'all']:
        print(f"\n{'='*70}")
        print("STEP 2: MODEL TRAINING")
        print(f"{'='*70}\n")
        
        try:
            # Load processed data
            processed_data = preprocess_dataset(dataset_name=args.dataset, reload=False)
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                processed_data,
                batch_size=args.batch_size,
                val_split=0.2,
                shuffle=True
            )
            
            # Prepare config
            config_dict = {
                'LSTM_INPUT_SIZE': processed_data['train_sequences'].shape[2],
                'LSTM_HIDDEN_SIZE': 128,
                'LSTM_NUM_LAYERS': 2,
                'LSTM_DROPOUT': 0.2,
                'LSTM_OUTPUT_SIZE': 1,
                'TRANSFORMER_INPUT_SIZE': processed_data['train_sequences'].shape[2],
                'TRANSFORMER_D_MODEL': 128,
                'TRANSFORMER_NHEAD': 8,
                'TRANSFORMER_NUM_LAYERS': 2,
                'TRANSFORMER_DIM_FEEDFORWARD': 256,
                'TRANSFORMER_DROPOUT': 0.2,
                'TRANSFORMER_OUTPUT_SIZE': 1,
            }
            
            models_to_train = ['lstm', 'transformer'] if args.model == 'both' else [args.model]
            
            for model_name in models_to_train:
                print(f"\n--- Training {model_name.upper()} ---")
                
                # Create model
                model = create_model(model_name, config_dict, DEVICE)
                print_model_info(model, model_name)
                
                # Create trainer
                trainer = Trainer(
                    model,
                    train_loader,
                    val_loader,
                    model_name,
                    config_dict
                )
                
                # Train
                history = trainer.train(
                    num_epochs=args.epochs,
                    resume_from_checkpoint=args.resume
                )
                
                # Save history
                trainer.save_training_history()
                
                print(f"✓ {model_name.upper()} training complete!")
            
        except Exception as e:
            print(f"✗ Error during training: {e}")
            return
    
    # ============================================================
    # EVALUATE
    # ============================================================
    
    if args.command in ['evaluate', 'all']:
        print(f"\n{'='*70}")
        print("STEP 3: MODEL EVALUATION")
        print(f"{'='*70}\n")
        
        try:
            # Load data
            processed_data = preprocess_dataset(dataset_name=args.dataset, reload=False)
            train_loader, val_loader, test_loader = create_data_loaders(processed_data)
            
            # Prepare config
            config_dict = {
                'LSTM_INPUT_SIZE': processed_data['train_sequences'].shape[2],
                'LSTM_HIDDEN_SIZE': 128,
                'LSTM_NUM_LAYERS': 2,
                'LSTM_DROPOUT': 0.2,
                'LSTM_OUTPUT_SIZE': 1,
                'TRANSFORMER_INPUT_SIZE': processed_data['train_sequences'].shape[2],
                'TRANSFORMER_D_MODEL': 128,
                'TRANSFORMER_NHEAD': 8,
                'TRANSFORMER_NUM_LAYERS': 2,
                'TRANSFORMER_DIM_FEEDFORWARD': 256,
                'TRANSFORMER_DROPOUT': 0.2,
                'TRANSFORMER_OUTPUT_SIZE': 1,
            }
            
            all_metrics = {}
            
            models_to_eval = ['lstm', 'transformer'] if args.model == 'both' else [args.model]
            
            for model_name in models_to_eval:
                print(f"\n--- Evaluating {model_name.upper()} ---")
                
                # Create and load model
                model = create_model(model_name, config_dict, DEVICE)
                checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"
                
                if not checkpoint_path.exists():
                    print(f"⚠ Checkpoint not found: {checkpoint_path}")
                    continue
                
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Evaluate
                metrics, evaluator = evaluate_model(
                    model,
                    test_loader,
                    model_name,
                    generate_plots=True
                )
                
                all_metrics[model_name.upper()] = metrics
            
            # Compare models
            if len(all_metrics) > 1:
                print(f"\n{'='*70}")
                print("MODEL COMPARISON")
                print(f"{'='*70}")
                print_comparison_table(all_metrics)
        
        except Exception as e:
            print(f"✗ Error during evaluation: {e}")
            return
    
    # ============================================================
    # PREDICT
    # ============================================================
    
    if args.command in ['predict', 'all']:
        print(f"\n{'='*70}")
        print("STEP 4: MAKING PREDICTIONS")
        print(f"{'='*70}\n")
        
        try:
            # Load data
            processed_data = preprocess_dataset(dataset_name=args.dataset, reload=False)
            scaler = processed_data['scaler']
            
            # Create sample data
            print("Creating sample engine data...")
            engine_data = create_sample_engine_data(num_cycles=100)
            
            models_to_pred = ['lstm', 'transformer'] if args.model == 'both' else [args.model]
            
            for model_name in models_to_pred:
                print(f"\n--- Predictions with {model_name.upper()} ---")
                
                # Prepare config
                config_dict = {
                    'LSTM_INPUT_SIZE': 24,
                    'LSTM_HIDDEN_SIZE': 128,
                    'LSTM_NUM_LAYERS': 2,
                    'LSTM_DROPOUT': 0.2,
                    'LSTM_OUTPUT_SIZE': 1,
                    'TRANSFORMER_INPUT_SIZE': 24,
                    'TRANSFORMER_D_MODEL': 128,
                    'TRANSFORMER_NHEAD': 8,
                    'TRANSFORMER_NUM_LAYERS': 2,
                    'TRANSFORMER_DIM_FEEDFORWARD': 256,
                    'TRANSFORMER_DROPOUT': 0.2,
                    'TRANSFORMER_OUTPUT_SIZE': 1,
                }
                
                # Load model
                model = create_model(model_name, config_dict, DEVICE)
                checkpoint_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"
                
                if not checkpoint_path.exists():
                    print(f"⚠ Checkpoint not found: {checkpoint_path}")
                    continue
                
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Make predictions
                predictor = RULPredictor(model, scaler)
                trajectory = predictor.predict_engine_trajectory(engine_data)
                health = predictor.get_health_status(trajectory['current_rul'])
                
                # Display results
                print(f"Current RUL: {trajectory['current_rul']:.2f} cycles")
                print(f"Health Status: {health['status']}")
                print(f"Recommendation: {health['recommendation']}")
        
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
            return
    
    print(f"\n{'='*70}")
    print("✓ Pipeline execution complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
