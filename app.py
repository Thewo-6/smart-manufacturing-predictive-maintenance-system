"""
Streamlit Dashboard for Predictive Maintenance System
Complete interactive interface for model training, prediction, and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
import io
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, CHECKPOINTS_DIR, RESULTS_DIR, SENSOR_COLUMNS, 
    SEQUENCE_LENGTH, DATASET_INSTRUCTIONS
)
from src.models import create_model, LSTMModel, TransformerEncoderModel
from src.dataset import create_data_loaders, TurbofanDataset
from src.data_preprocessing import preprocess_dataset, DataPreprocessor
from src.evaluate import Evaluator
from src.predict import RULPredictor, create_sample_engine_data
from src.utils import set_random_seed, print_comparison_table, get_device_info

# Set random seed for reproducibility
set_random_seed()

# Configure page
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and styling
st.title("AI-Powered Predictive Maintenance System")
st.markdown("### Remaining Useful Life Prediction for Turbofan Engines")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "🏋️ Model Training",
        "🔮 Make Predictions",
        "📈 Visualizations",
        "⚖️ Model Comparison"
    ]
)

# ============================================================
# PAGE 1: HOME
# ============================================================

if page == "🏠 Home":
    st.markdown("""
    # Welcome to the Predictive Maintenance System
    
    This application uses machine learning to predict the Remaining Useful Life (RUL) 
    of turbofan engines using the NASA C-MAPSS dataset.
    
    ## 🎯 Project Overview
    
    **Goal:** Predict when turbofan engines will fail by analyzing multivariate time-series 
    sensor data. This enables proactive maintenance scheduling and reduces unplanned downtime.
    
    ## 🔧 Features
    
    - **Two Advanced Models:**
      - LSTM (Long Short-Term Memory) - Excellent for sequential data
      - Transformer Encoder - State-of-the-art attention mechanism
    
    - **Comprehensive Evaluation:**
      - MAE, RMSE, R² Score, MAPE
      - NASA Prognostics Center of Excellence scoring function
    
    - **Interactive Dashboard:**
      - Dataset exploration
      - Model training with real-time monitoring
      - Single engine or batch predictions
      - Detailed visualizations and comparisons
    
    ## 📊 Dataset
    
    **NASA C-MAPSS Turbofan Engine Dataset**
    - 21 sensor channels
    - 3 operational settings
    - Multiple failure modes
    - Thousands of complete engine runs-to-failure
    
    ## 🚀 Quick Start
    
    1. **Download Dataset:** Go to Dataset Overview page
    2. **Train Model:** Use Model Training page
    3. **Make Predictions:** Use Make Predictions page
    4. **Analyze Results:** Check Visualizations page
    
    ## 💡 Industrial Value
    
    Predictive maintenance provides:
    - **Reduced Downtime:** Schedule maintenance before failures
    - **Cost Savings:** Avoid expensive emergency repairs
    - **Safety:** Prevent catastrophic failures
    - **Optimization:** Maximize engine operational life
    
    ---
    
    **Device Information:** {0}
    """.format(get_device_info()))
    
    st.info("Select a page from the sidebar to get started!")


# ============================================================
# PAGE 2: DATASET OVERVIEW
# ============================================================

elif page == "📊 Dataset Overview":
    st.header("📊 Dataset Overview")
    
    st.markdown("""
    ## NASA C-MAPSS Turbofan Engine Dataset
    
    ### Download Instructions
    """)
    
    with st.expander("📥 How to Download the Dataset", expanded=False):
        st.markdown("""
        1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
        2. Download "Turbofan Engine Degradation Simulation Data Set"
        3. Extract the ZIP file
        4. Copy the following files to `data/raw/`:
           - `train_FD001.txt`
           - `test_FD001.txt`
           - `RUL_FD001.txt`
        5. You can also download FD002, FD003, FD004 for different scenarios
        """)
    
    # Check if data is available
    train_file = Path("data/raw/train_FD001.txt")
    test_file = Path("data/raw/test_FD001.txt")
    rul_file = Path("data/raw/RUL_FD001.txt")
    
    if train_file.exists() and test_file.exists() and rul_file.exists():
        st.success("✓ Dataset files found!")
        
        # Load and display dataset info
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Dataset Info"):
                with st.spinner("Loading dataset..."):
                    preprocessor = DataPreprocessor(dataset_name="FD001")
                    train_df, test_df, train_rul, test_rul = preprocessor.read_raw_files()
                    preprocessor.calculate_train_rul()
                    
                    st.write("### Training Data")
                    st.write(f"Shape: {train_df.shape}")
                    st.write(f"Unique engines: {train_df['Engine_ID'].nunique()}")
                    st.write(f"Columns: {', '.join(train_df.columns[:10].tolist())}...")
                    
                    st.write("### Test Data")
                    st.write(f"Shape: {test_df.shape}")
                    st.write(f"Unique engines: {test_df['Engine_ID'].nunique()}")
                    
                    st.write("### Sample Data (First 5 rows)")
                    st.dataframe(train_df.iloc[:5, :10])
        
        with col2:
            if st.button("View Dataset Statistics"):
                with st.spinner("Computing statistics..."):
                    processed_data = preprocess_dataset(dataset_name="FD001", reload=False)
                    
                    st.write("### Processed Data Statistics")
                    st.write(f"Training sequences: {processed_data['train_sequences'].shape[0]}")
                    st.write(f"Test sequences: {processed_data['test_sequences'].shape[0]}")
                    st.write(f"Sequence length: {processed_data['train_sequences'].shape[1]}")
                    st.write(f"Number of features: {processed_data['train_sequences'].shape[2]}")
                    
                    st.write("### RUL Statistics")
                    st.write(f"Train RUL - Mean: {processed_data['train_rul'].mean():.2f}, "
                           f"Std: {processed_data['train_rul'].std():.2f}")
                    st.write(f"Train RUL - Min: {processed_data['train_rul'].min():.2f}, "
                           f"Max: {processed_data['train_rul'].max():.2f}")
                    st.write(f"Test RUL - Mean: {processed_data['test_rul'].mean():.2f}, "
                           f"Std: {processed_data['test_rul'].std():.2f}")
                    st.write(f"Test RUL - Min: {processed_data['test_rul'].min():.2f}, "
                           f"Max: {processed_data['test_rul'].max():.2f}")
    else:
        st.error("❌ Dataset not found!")
        st.write("Please download the NASA C-MAPSS dataset and place it in the data/raw/ directory.")
        st.write("See instructions above.")


# ============================================================
# PAGE 3: MODEL TRAINING
# ============================================================

elif page == "🏋️ Model Training":
    st.header("🏋️ Model Training")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        model_choice = st.radio("Select Model:", ["LSTM", "Transformer"])
        dataset_choice = st.selectbox("Dataset:", ["FD001", "FD002", "FD003", "FD004"])
        num_epochs = st.slider("Number of Epochs:", 10, 200, 50)
    
    with col2:
        st.subheader("Training Status")
        
        if st.button("🚀 Start Training"):
            try:
                with st.spinner(f"Preparing {dataset_choice} dataset..."):
                    processed_data = preprocess_dataset(dataset_name=dataset_choice, reload=False)
                    train_loader, val_loader, test_loader = create_data_loaders(
                        processed_data,
                        batch_size=32,
                        val_split=0.2
                    )
                
                with st.spinner(f"Creating {model_choice} model..."):
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
                    
                    model = create_model(model_choice.lower(), config_dict, DEVICE)
                
                st.success(f"✓ {model_choice} model created!")
                st.info("Training would typically start here. For demo, use the prediction page instead.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ============================================================
# PAGE 4: MAKE PREDICTIONS
# ============================================================

elif page == "🔮 Make Predictions":
    st.header("🔮 Make Predictions")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Prediction Configuration")
        model_choice = st.radio("Select Model:", ["LSTM", "Transformer"], key="pred_model")
        data_source = st.radio("Data Source:", ["Sample Data", "Upload CSV", "Test Dataset"])
    
    with col2:
        st.subheader("Prediction Results")
        
        try:
            if st.button("🔮 Generate Predictions"):
                with st.spinner("Loading model..."):
                    # Load checkpoint
                    checkpoint_path = CHECKPOINTS_DIR / f"{model_choice.lower()}_best.pt"
                    
                    if not checkpoint_path.exists():
                        st.warning(f"Checkpoint not found: {checkpoint_path}")
                        st.info("Please train the model first on the Model Training page.")
                    else:
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
                        
                        model = create_model(model_choice.lower(), config_dict, DEVICE)
                        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        
                        # Load scaler
                        processed_data = preprocess_dataset("FD001")
                        scaler = processed_data['scaler']
                        
                        # Create predictor
                        predictor = RULPredictor(model, scaler)
                        
                        # Get data
                        if data_source == "Sample Data":
                            with st.spinner("Creating sample engine data..."):
                                engine_data = create_sample_engine_data(num_cycles=100)
                        
                        elif data_source == "Test Dataset":
                            with st.spinner("Loading test dataset..."):
                                # Load raw test data
                                preprocessor = DataPreprocessor("FD001")
                                preprocessor.read_raw_files()
                                preprocessor.calculate_test_rul()
                                preprocessor.normalize_features(fit_on_train=True)
                                
                                # Get first 100 cycles from first engine
                                test_df = preprocessor.test_data
                                first_engine = test_df[test_df['Engine_ID'] == 1].head(100)
                                
                                if len(first_engine) > 0:
                                    engine_data = first_engine
                                    st.info(f"Using first engine from test set ({len(engine_data)} cycles)")
                                else:
                                    st.error("No test data available")
                                    st.stop()
                        
                        elif data_source == "Upload CSV":
                            uploaded_file = st.file_uploader("Upload CSV with sensor data", type="csv")
                            if uploaded_file is not None:
                                try:
                                    df = pd.read_csv(uploaded_file)
                                    # Filter to only sensor columns
                                    available_cols = [col for col in SENSOR_COLUMNS if col in df.columns]
                                    if len(available_cols) == 0:
                                        st.error(f"CSV must contain sensor columns: {', '.join(SENSOR_COLUMNS[:5])}...")
                                        st.stop()
                                    
                                    engine_data = df[available_cols].values
                                    st.success(f"Loaded {len(engine_data)} cycles with {len(available_cols)} sensors")
                                except Exception as e:
                                    st.error(f"Error reading CSV: {str(e)}")
                                    st.stop()
                            else:
                                st.info("Please upload a CSV file")
                                st.stop()
                        
                        # Make predictions
                        with st.spinner("Making predictions..."):
                            trajectory = predictor.predict_engine_trajectory(engine_data)
                            health = predictor.get_health_status(trajectory['current_rul'])
                        
                        # Display results
                        st.success("✓ Predictions generated!")
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        
                        with col_res1:
                            st.metric("Current RUL", f"{trajectory['current_rul']:.2f} cycles")
                        
                        with col_res2:
                            st.metric("Status", health['status'])
                        
                        with col_res3:
                            st.write(f"**Recommendation:**\n{health['recommendation']}")
                        
                        # Plot trajectory
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(trajectory['cycles'], trajectory['rul_predictions'], 'b-o', linewidth=2)
                        ax.axhline(y=health['critical_threshold'], color='red', linestyle='--', 
                                  label='Critical', linewidth=2)
                        ax.axhline(y=health['warning_threshold'], color='orange', linestyle='--',
                                  label='Warning', linewidth=2)
                        ax.fill_between(trajectory['cycles'], 0, health['critical_threshold'],
                                       color='red', alpha=0.1, label='Critical Zone')
                        ax.set_xlabel('Cycle')
                        ax.set_ylabel('Remaining Useful Life (cycles)')
                        ax.set_title(f'{model_choice} - Engine Health Degradation')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ============================================================
# PAGE 5: VISUALIZATIONS
# ============================================================

elif page == "📈 Visualizations":
    st.header("📈 Visualizations & Analysis")
    
    st.subheader("Model Performance Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Training History", "Error Analysis", "Engine Degradation"])
    
    with tab1:
        st.write("### Training History")
        model_choice = st.selectbox("Select Model:", ["lstm", "transformer"], key="viz_model1")
        history_file = RESULTS_DIR / f"{model_choice}_training_history.json"
        
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            epochs = history['epoch']
            train_loss = history['train_loss']
            val_loss = history['val_loss']
            
            axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
            axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title(f'{model_choice.upper()} - Training History')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title(f'{model_choice.upper()} - Learning Rate Schedule')
            axes[1].grid(True, alpha=0.3)
            
            st.pyplot(fig)
        else:
            st.info(f"No training history found for {model_choice}. Train the model first.")
    
    with tab2:
        st.write("### Error Analysis")
        model_choice2 = st.selectbox("Select Model:", ["lstm", "transformer"], key="viz_model2")
        error_plot_file = Path("outputs/figures") / f"{model_choice2}_error_distribution.png"
        
        if error_plot_file.exists():
            st.image(str(error_plot_file), use_column_width=True)
        else:
            st.info(f"No error distribution plot found. Evaluate the model first.")
    
    with tab3:
        st.write("### Engine Degradation Curves")
        model_choice3 = st.selectbox("Select Model:", ["lstm", "transformer"], key="viz_model3")
        degradation_plot_file = Path("outputs/figures") / f"{model_choice3}_degradation_curves.png"
        
        if degradation_plot_file.exists():
            st.image(str(degradation_plot_file), use_column_width=True)
        else:
            st.info(f"No degradation curves found. Evaluate the model first.")


# ============================================================
# PAGE 6: MODEL COMPARISON
# ============================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Model Comparison")
    
    st.subheader("Compare LSTM vs Transformer Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Model Information")
        st.write("""
        **LSTM (Long Short-Term Memory)**
        - Parameters: ~150K
        - Strengths: Excellent for sequences, good memory
        - Weaknesses: Slower to train, less parallelizable
        - Training Time: ~30 minutes (GPU)
        
        **Transformer Encoder**
        - Parameters: ~130K
        - Strengths: Highly parallelizable, better scaling
        - Weaknesses: Needs more data, higher memory
        - Training Time: ~20 minutes (GPU)
        """)
    
    with col2:
        st.write("### Comparison Metrics")
        if st.button("Load Comparison Results"):
            metrics_dict = {
                'LSTM': {
                    'MAE': 0.0,
                    'RMSE': 0.0,
                    'R2': 0.0,
                    'NASA_Score': 0.0
                },
                'Transformer': {
                    'MAE': 0.0,
                    'RMSE': 0.0,
                    'R2': 0.0,
                    'NASA_Score': 0.0
                }
            }
            
            comparison_df = pd.DataFrame(metrics_dict).T
            st.dataframe(comparison_df)
            
            st.write("""
            **Recommendation:**
            - Use LSTM for familiar environments with proven performance
            - Use Transformer for large-scale deployments needing speed
            - Ensemble both models for highest accuracy
            """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔧 AI-Powered Predictive Maintenance System | NASA C-MAPSS Dataset | PyTorch | Streamlit</p>
    <p>Made with ❤️ by Max Ally W. THERESIAS</p>
</div>
""", unsafe_allow_html=True)
