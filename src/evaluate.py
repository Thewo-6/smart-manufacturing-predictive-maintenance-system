"""
Evaluation module for Turbofan Engine RUL Prediction
Includes metrics (MAE, RMSE, R²) and NASA scoring function
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import DEVICE, ALPHA1, ALPHA2, RESULTS_DIR, FIGURES_DIR, PLOT_STYLE


class Evaluator:
    """
    Evaluator class for model evaluation and visualization
    """
    
    def __init__(self, model, test_loader, model_name, device=DEVICE):
        """
        Initialize evaluator
        
        Args:
            model (nn.Module): Trained model
            test_loader (DataLoader): Test data loader
            model_name (str): Name of model
            device (torch.device): Device to use
        """
        self.model = model
        self.test_loader = test_loader
        self.model_name = model_name
        self.device = device
        
        self.predictions = None
        self.targets = None
        self.engine_ids = None
    
    def predict(self):
        """
        Generate predictions on test set
        
        Returns:
            Tuple of (predictions, targets, engine_ids)
        """
        self.model.eval()
        predictions = []
        targets = []
        engine_ids = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                if len(batch) == 3:
                    sequences, rul_values, ids = batch
                else:
                    sequences, rul_values = batch
                    ids = None
                
                sequences = sequences.to(self.device)
                rul_values = rul_values.to(self.device)
                
                pred = self.model(sequences)
                
                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(rul_values.cpu().numpy().flatten())
                
                if ids is not None:
                    engine_ids.extend(ids.cpu().numpy().flatten())
        
        self.predictions = np.array(predictions)
        self.targets = np.array(targets)
        self.engine_ids = np.array(engine_ids) if engine_ids else None
        
        return self.predictions, self.targets, self.engine_ids
    
    def calculate_mae(self):
        """Calculate Mean Absolute Error"""
        if self.predictions is None:
            raise ValueError("Predictions not generated. Call predict() first.")
        return mean_absolute_error(self.targets, self.predictions)
    
    def calculate_rmse(self):
        """Calculate Root Mean Squared Error"""
        if self.predictions is None:
            raise ValueError("Predictions not generated. Call predict() first.")
        return np.sqrt(mean_squared_error(self.targets, self.predictions))
    
    def calculate_r2(self):
        """Calculate R² Score"""
        if self.predictions is None:
            raise ValueError("Predictions not generated. Call predict() first.")
        return r2_score(self.targets, self.predictions)
    
    def calculate_mape(self):
        """Calculate Mean Absolute Percentage Error"""
        if self.predictions is None:
            raise ValueError("Predictions not generated. Call predict() first.")
        
        # Avoid division by zero
        mask = self.targets != 0
        mape = np.mean(np.abs((self.targets[mask] - self.predictions[mask]) / self.targets[mask])) * 100
        return mape
    
    def calculate_nasa_score(self, alpha1=ALPHA1, alpha2=ALPHA2):
        """
        Calculate NASA Prognostics Center of Excellence scoring function
        
        The NASA scoring function penalizes late predictions more heavily than early ones:
        - If predicted RUL < true RUL: score = exp(-error / alpha1) - 1
        - If predicted RUL >= true RUL: score = exp(error / alpha2) - 1
        
        Args:
            alpha1 (int): Penalty multiplier for early predictions
            alpha2 (int): Penalty multiplier for late predictions
            
        Returns:
            float: Total NASA score (lower is better)
        """
        errors = self.predictions - self.targets
        
        scores = np.where(
            errors < 0,
            np.exp(-errors / alpha1) - 1,
            np.exp(errors / alpha2) - 1
        )
        
        total_score = np.sum(scores)
        return total_score
    
    def get_all_metrics(self):
        """
        Calculate all evaluation metrics
        
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            'MAE': self.calculate_mae(),
            'RMSE': self.calculate_rmse(),
            'R2': self.calculate_r2(),
            'MAPE': self.calculate_mape(),
            'NASA_Score': self.calculate_nasa_score()
        }
        return metrics
    
    def print_metrics(self):
        """Print all evaluation metrics"""
        metrics = self.get_all_metrics()
        
        print(f"\n{'='*60}")
        print(f"Evaluation Metrics for {self.model_name.upper()}")
        print(f"{'='*60}")
        print(f"Mean Absolute Error (MAE):      {metrics['MAE']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
        print(f"R² Score:                       {metrics['R2']:.4f}")
        print(f"Mean Absolute % Error (MAPE):   {metrics['MAPE']:.2f}%")
        print(f"NASA Score:                     {metrics['NASA_Score']:.4f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def plot_predictions_vs_actual(self, save=True):
        """
        Plot predicted vs actual RUL values
        
        Args:
            save (bool): Whether to save figure
        """
        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Scatter plot
        axes[0].scatter(self.targets, self.predictions, alpha=0.5, s=20)
        axes[0].plot([self.targets.min(), self.targets.max()], 
                     [self.targets.min(), self.targets.max()], 
                     'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('True RUL')
        axes[0].set_ylabel('Predicted RUL')
        axes[0].set_title(f'{self.model_name.upper()} - Predicted vs Actual RUL')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = self.predictions - self.targets
        axes[1].scatter(self.targets, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('True RUL')
        axes[1].set_ylabel('Prediction Error')
        axes[1].set_title(f'{self.model_name.upper()} - Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = FIGURES_DIR / f"{self.model_name}_predictions_vs_actual.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved figure: {save_path}")
        
        plt.show()
    
    def plot_error_distribution(self, save=True):
        """
        Plot error distribution
        
        Args:
            save (bool): Whether to save figure
        """
        plt.style.use(PLOT_STYLE)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        errors = self.predictions - self.targets
        
        # Histogram
        axes[0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero error')
        axes[0].set_xlabel('Prediction Error (RUL cycles)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{self.model_name.upper()} - Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot([errors], labels=[self.model_name])
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_ylabel('Prediction Error (RUL cycles)')
        axes[1].set_title(f'{self.model_name.upper()} - Error Box Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = FIGURES_DIR / f"{self.model_name}_error_distribution.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved figure: {save_path}")
        
        plt.show()
    
    def plot_engine_degradation(self, num_engines=5, save=True):
        """
        Plot health degradation curves for sample engines
        
        Args:
            num_engines (int): Number of engines to plot
            save (bool): Whether to save figure
        """
        if self.engine_ids is None:
            print("Engine IDs not available for degradation plot")
            return
        
        plt.style.use(PLOT_STYLE)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        unique_engines = np.unique(self.engine_ids)[:num_engines]
        
        for engine_id in unique_engines:
            mask = self.engine_ids == engine_id
            engine_pred = self.predictions[mask]
            engine_target = self.targets[mask]
            cycles = np.arange(len(engine_target))
            
            ax.plot(cycles, engine_target, 'o-', label=f'Engine {engine_id} (True)', linewidth=2)
            ax.plot(cycles, engine_pred, 's--', label=f'Engine {engine_id} (Pred)', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Cycle/Time Step')
        ax.set_ylabel('Remaining Useful Life (RUL)')
        ax.set_title(f'{self.model_name.upper()} - Engine Health Degradation')
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = FIGURES_DIR / f"{self.model_name}_degradation_curves.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved figure: {save_path}")
        
        plt.show()
    
    def save_results_csv(self):
        """Save predictions and targets to CSV file"""
        results_df = pd.DataFrame({
            'True_RUL': self.targets,
            'Predicted_RUL': self.predictions,
            'Error': self.predictions - self.targets,
            'Absolute_Error': np.abs(self.predictions - self.targets)
        })
        
        if self.engine_ids is not None:
            results_df['Engine_ID'] = self.engine_ids
        
        save_path = RESULTS_DIR / f"{self.model_name}_predictions.csv"
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to: {save_path}")
        
        return results_df


def evaluate_model(model, test_loader, model_name, generate_plots=True):
    """
    Complete evaluation pipeline
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        model_name (str): Name of model
        generate_plots (bool): Whether to generate visualization plots
        
    Returns:
        dict: Evaluation metrics
    """
    
    print(f"\nEvaluating {model_name.upper()} model...")
    
    evaluator = Evaluator(model, test_loader, model_name)
    
    # Generate predictions
    print("Generating predictions...")
    evaluator.predict()
    
    # Calculate metrics
    metrics = evaluator.print_metrics()
    
    # Save results
    evaluator.save_results_csv()
    
    # Generate plots
    if generate_plots:
        print("Generating plots...")
        evaluator.plot_predictions_vs_actual(save=True)
        evaluator.plot_error_distribution(save=True)
        evaluator.plot_engine_degradation(num_engines=5, save=True)
    
    return metrics, evaluator


if __name__ == "__main__":
    from src.models import create_model
    from src.dataset import create_data_loaders
    from src.data_preprocessing import preprocess_dataset
    from config import DEVICE
    
    # Load processed data
    processed_data = preprocess_dataset(dataset_name="FD001")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(processed_data)
    
    # Create config dict
    config_dict = {
        'LSTM_INPUT_SIZE': len(processed_data['train_sequences'][0][0]),
        'LSTM_HIDDEN_SIZE': 128,
        'LSTM_NUM_LAYERS': 2,
        'LSTM_DROPOUT': 0.2,
        'LSTM_OUTPUT_SIZE': 1,
    }
    
    # Create and evaluate LSTM model
    model = create_model('lstm', config_dict, DEVICE)
    
    # Load best checkpoint
    checkpoint_path = Path("checkpoints/lstm_best.pt")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Evaluate
    metrics, evaluator = evaluate_model(model, test_loader, 'lstm', generate_plots=True)
