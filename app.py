############################################################
#   Streamlit Visualization for Augmented Population Based Training
#   Interactive dashboard for training visualization
############################################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'source'))

from apbt import APBT
from ann import ANN
import threading
import time

# Page configuration
st.set_page_config(
    page_title="APBT Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3240;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    h1 {
        color: #00d4ff;
    }
    h2 {
        color: #00b4d8;
    }
    h3 {
        color: #0096c7;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🧬 Augmented Population-Based Training Dashboard")
st.markdown("""
**Population-Based Training** with **Neural Architecture Search**  
Evolving neural networks through competitive training and architecture optimization
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Dataset selection
dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Tennis", "Identity"],
    index=0
)

# Map dataset to file paths
dataset_map = {
    "Iris": {
        "attr": "data/iris/iris-attr.txt",
        "train": "data/iris/iris-train.txt",
        "test": "data/iris/iris-test.txt"
    },
    "Tennis": {
        "attr": "data/tennis/tennis-attr.txt",
        "train": "data/tennis/tennis-train.txt",
        "test": "data/tennis/tennis-test.txt"
    },
    "Identity": {
        "attr": "data/identity/identity-attr.txt",
        "train": "data/identity/identity-train.txt",
        "test": None
    }
}

# Training parameters
st.sidebar.subheader("Training Parameters")
k_inds = st.sidebar.slider("Population Size (k)", 10, 500, 40, 10)
epochs = st.sidebar.slider("Number of Epochs", 10, 1000, 100, 10)

# Calculate default readiness as 5% of epochs
default_readiness = int(epochs * 0.05)

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    lr_min = st.number_input("Min Learning Rate", 0.0001, 0.1, 0.0001, format="%.4f")
    lr_max = st.number_input("Max Learning Rate", 0.001, 1.0, 0.1, format="%.4f")
    
    st.markdown("---")
    
    st.info(f"💡 Recommended Readiness: {default_readiness} epochs (5% of total)")
    readiness = st.slider("Readiness Threshold", 10, 500, default_readiness, 10)
    truncation = st.slider("Truncation %", 10, 50, 20, 5) / 100

# Initialize session state
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0
if 'history' not in st.session_state:
    st.session_state.history = {
        'epoch': [],
        'best_perf': [],
        'best_acc': [],
        'best_size': [],
        'most_acc': [],
        'avg_perf': [],
        'avg_acc': [],
        'learning_rate': [],
        'momentum': [],
        'decay': [],
        'best_topology': [],  # Topology of best performer over time
        'most_acc_topology': []  # Topology of most accurate over time
    }
if 'apbt' not in st.session_state:
    st.session_state.apbt = None
if 'best_net' not in st.session_state:
    st.session_state.best_net = None

# Training control buttons
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    start_button = st.button("▶️ Start Training", disabled=st.session_state.training_started)

with col2:
    if st.button("🔄 Reset"):
        st.session_state.training_started = False
        st.session_state.training_complete = False
        st.session_state.current_epoch = 0
        st.session_state.history = {
            'epoch': [],
            'best_perf': [],
            'best_acc': [],
            'best_size': [],
            'most_acc': [],
            'avg_perf': [],
            'avg_acc': [],
            'learning_rate': [],
            'momentum': [],
            'decay': [],
            'best_topology': [],
            'most_acc_topology': []
        }
        st.session_state.apbt = None
        st.session_state.best_net = None
        st.rerun()

# Initialize APBT when start is clicked
if start_button and not st.session_state.training_started:
    st.session_state.training_started = True
    
    logger.info("="*80)
    logger.info("TRAINING SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  - Dataset: {dataset}")
    logger.info(f"  - Population Size: {k_inds}")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Readiness Threshold: {readiness}")
    logger.info(f"  - Truncation: {truncation*100:.0f}%")
    logger.info(f"  - Learning Rate Range: [{lr_min:.4f}, {lr_max:.4f}]")
    
    with st.spinner("Initializing population..."):
        try:
            # Get dataset paths
            paths = dataset_map[dataset]
            
            # Initialize APBT
            logger.info("Initializing population...")
            apbt = APBT(
                k=k_inds,
                end_training=epochs,
                training=paths["train"],
                testing=paths["test"],
                attributes=paths["attr"],
                debug=False
            )
            
            # Override some parameters from UI
            apbt.LR_RANGE = (lr_min, lr_max)
            apbt.READINESS = readiness
            apbt.TRUNC = truncation
            
            st.session_state.apbt = apbt
            st.success("✅ Population initialized!")
            logger.info(f"✓ Population of {k_inds} networks initialized successfully")
            logger.info(f"  - Input units: {apbt.input_units}")
            logger.info(f"  - Output units: {apbt.output_units}")
            logger.info(f"  - Training examples: {len(apbt.training)}")
            logger.info(f"  - Validation examples: {len(apbt.validation)}")
            
        except Exception as e:
            logger.error(f"✗ Error initializing APBT: {str(e)}")
            st.error(f"❌ Error initializing APBT: {str(e)}")
            st.session_state.training_started = False

# Show progress bar if training has started
if st.session_state.training_started:
    progress = st.session_state.current_epoch / max(epochs, 1)
    st.progress(progress)
    
    if st.session_state.training_complete:
        st.success("✅ Training Complete!")
    else:
        st.info(f"🔄 Training... Epoch {st.session_state.current_epoch}/{epochs}")

# Main content area with tabs (BEFORE training so they display during training)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Training Progress", 
    "🏆 Leaderboard", 
    "🧠 Architecture", 
    "📈 Hyperparameters",
    "🎯 Fitness Landscape"
])

# Display results
with tab1:
    st.header("📊 Training Progress")
    
    if st.session_state.training_started and len(st.session_state.history['epoch']) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Best Performance", 
                f"{st.session_state.history['best_perf'][-1]:.2f}",
                delta=f"{st.session_state.history['best_perf'][-1] - st.session_state.history['best_perf'][0]:.2f}" if len(st.session_state.history['best_perf']) > 1 else None
            )
        
        with col2:
            st.metric(
                "Best Accuracy", 
                f"{st.session_state.history['best_acc'][-1]*100:.1f}%",
                delta=f"{(st.session_state.history['best_acc'][-1] - st.session_state.history['best_acc'][0])*100:.1f}%" if len(st.session_state.history['best_acc']) > 1 else None
            )
        
        with col3:
            st.metric(
                "Model Size", 
                f"{st.session_state.history['best_size'][-1]} params"
            )
        
        with col4:
            st.metric(
                "Current Epoch", 
                f"{st.session_state.current_epoch}/{epochs}"
            )
        
        st.markdown("---")
        
        # Performance over time
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Evolution', 'Accuracy Evolution', 
                          'Model Size Evolution', 'Average Population Stats'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Performance
        fig.add_trace(
            go.Scatter(x=st.session_state.history['epoch'], 
                      y=st.session_state.history['best_perf'],
                      name='Best Performance',
                      line=dict(color='#00d4ff', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=st.session_state.history['epoch'], 
                      y=st.session_state.history['avg_perf'],
                      name='Avg Performance',
                      line=dict(color='#ff6b6b', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Accuracy
        fig.add_trace(
            go.Scatter(x=st.session_state.history['epoch'], 
                      y=[a*100 for a in st.session_state.history['best_acc']],
                      name='Best Accuracy',
                      line=dict(color='#4ecdc4', width=3)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=st.session_state.history['epoch'], 
                      y=[a*100 for a in st.session_state.history['most_acc']],
                      name='Most Accurate',
                      line=dict(color='#95e1d3', width=2, dash='dot')),
            row=1, col=2
        )
        
        # Size
        fig.add_trace(
            go.Scatter(x=st.session_state.history['epoch'], 
                      y=st.session_state.history['best_size'],
                      name='Model Parameters',
                      line=dict(color='#ffd93d', width=3),
                      fill='tozeroy'),
            row=2, col=1
        )
        
        # Average population
        fig.add_trace(
            go.Scatter(x=st.session_state.history['epoch'], 
                      y=[a*100 for a in st.session_state.history['avg_acc']],
                      name='Avg Accuracy',
                      line=dict(color='#a29bfe', width=3)),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Performance", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        fig.update_yaxes(title_text="Parameters", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
        
        fig.update_layout(
            height=700,
            showlegend=True,
            template="plotly_dark",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Configure parameters and click 'Start Training' to begin")

with tab2:
    st.header("🏆 Population Leaderboard")
    
    if st.session_state.apbt and len(st.session_state.history['epoch']) > 0:
        apbt = st.session_state.apbt
        
        # Create leaderboard dataframe
        leaderboard_data = []
        for rank, idx in enumerate(apbt.leaderboard[:20]):  # Top 20
            net = apbt.population[idx]
            hyperparams = apbt.hyperparams[idx]
            perf = apbt.perfs[idx]
            acc = apbt.accuracies[idx]
            
            leaderboard_data.append({
                'Rank': rank + 1,
                'Network ID': idx,
                'Performance': f"{perf:.3f}",
                'Accuracy': f"{acc*100:.2f}%",
                'Size': net.num_params(),
                'Topology': str(net.topology),
                'Learning Rate': f"{hyperparams['learning_rate']:.4f}",
                'Momentum': f"{hyperparams['momentum']:.3f}",
                'Decay': f"{hyperparams['decay']:.4f}"
            })
        
        df = pd.DataFrame(leaderboard_data)
        
        # Highlight top 3
        def highlight_top3(row):
            if row['Rank'] == 1:
                return ['background-color: #ffd700'] * len(row)  # Gold
            elif row['Rank'] == 2:
                return ['background-color: #c0c0c0'] * len(row)  # Silver
            elif row['Rank'] == 3:
                return ['background-color: #cd7f32'] * len(row)  # Bronze
            return [''] * len(row)
        
        st.dataframe(df, use_container_width=True, height=600)
        
        # Population distribution
        st.subheader("Population Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance distribution
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Histogram(
                x=apbt.perfs,
                nbinsx=20,
                name='Performance',
                marker_color='#00d4ff'
            ))
            fig_perf.update_layout(
                title="Performance Distribution",
                xaxis_title="Performance",
                yaxis_title="Count",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_perf, use_container_width=True)
        
        with col2:
            # Size distribution
            sizes = [net.num_params() for net in apbt.population]
            fig_size = go.Figure()
            fig_size.add_trace(go.Histogram(
                x=sizes,
                nbinsx=20,
                name='Model Size',
                marker_color='#ffd93d'
            ))
            fig_size.update_layout(
                title="Model Size Distribution",
                xaxis_title="Parameters",
                yaxis_title="Count",
                template="plotly_dark",
                height=300
            )
            st.plotly_chart(fig_size, use_container_width=True)
    else:
        st.info("Start training to see the leaderboard")

with tab3:
    st.header("🧠 Neural Architecture Evolution")
    
    if len(st.session_state.history.get('best_topology', [])) > 0:
        
        def draw_topology(topology, title):
            """Helper function to draw network topology"""
            max_units = max(topology)
            layers = len(topology)
            
            fig = go.Figure()
            
            # Draw nodes
            for layer_idx, num_units in enumerate(topology):
                x_pos = layer_idx
                y_offset = (max_units - num_units) / 2
                
                for unit_idx in range(num_units):
                    y_pos = unit_idx + y_offset
                    
                    # Color based on layer type
                    if layer_idx == 0:
                        color = '#00d4ff'  # Input
                        label = 'Input'
                    elif layer_idx == layers - 1:
                        color = '#4ecdc4'  # Output
                        label = 'Output'
                    else:
                        color = '#ffd93d'  # Hidden
                        label = f'Hidden {layer_idx}'
                    
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode='markers',
                        marker=dict(size=30, color=color, line=dict(width=2, color='white')),
                        name=label,
                        showlegend=unit_idx == 0,
                        hovertemplate=f'Layer {layer_idx}<br>Unit {unit_idx}<extra></extra>'
                    ))
            
            # Draw connections between layers
            for layer_idx in range(layers - 1):
                num_units_current = topology[layer_idx]
                num_units_next = topology[layer_idx + 1]
                
                y_offset_current = (max_units - num_units_current) / 2
                y_offset_next = (max_units - num_units_next) / 2
                
                # Draw sample connections (not all to avoid clutter on large networks)
                max_connections_per_layer = min(8, num_units_current)
                for i in range(max_connections_per_layer):
                    for j in range(min(8, num_units_next)):
                        fig.add_trace(go.Scatter(
                            x=[layer_idx, layer_idx + 1],
                            y=[i + y_offset_current, j + y_offset_next],
                            mode='lines',
                            line=dict(width=1.5, color='rgba(100,200,255,0.5)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            fig.update_layout(
                title=title,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark",
                height=400,
                hovermode='closest'
            )
            
            return fig
        
        # Best Performer (by fitness) Evolution
        st.subheader("🏆 Best Performer Evolution (Optimal Fitness)")
        
        num_epochs_best = len(st.session_state.history['best_topology'])
        
        if num_epochs_best > 0:
            # Only show slider if there's more than 1 epoch
            if num_epochs_best > 1:
                epoch_best = st.slider(
                    "Select Epoch (Best Performer)",
                    min_value=0,
                    max_value=num_epochs_best - 1,
                    value=num_epochs_best - 1,
                    key='slider_best'
                )
            else:
                epoch_best = 0
                st.info("Slider will appear after more epochs complete")
            
            best_topology = st.session_state.history['best_topology'][epoch_best]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Epoch:** {epoch_best}")
                st.markdown(f"**Topology:** {best_topology}")
                st.markdown(f"**Total Parameters:** {sum((best_topology[i] * best_topology[i-1]) for i in range(1, len(best_topology)))}")
                st.markdown(f"**Accuracy:** {st.session_state.history['best_acc'][epoch_best]*100:.2f}%")
                st.markdown(f"**Fitness:** {st.session_state.history['best_perf'][epoch_best]:.2f}")
                
                st.markdown("---")
                st.markdown("**Layer Details:**")
                for i, units in enumerate(best_topology):
                    if i == 0:
                        st.text(f"Input: {units} units")
                    elif i == len(best_topology) - 1:
                        st.text(f"Output: {units} units")
                    else:
                        st.text(f"Hidden {i}: {units} units")
            
            with col2:
                fig_best = draw_topology(best_topology, f"Best Performer at Epoch {epoch_best}")
                st.plotly_chart(fig_best, use_container_width=True)
        
        st.markdown("---")
        
        # Most Accurate Network Evolution
        st.subheader("🎯 Most Accurate Network Evolution (Highest Accuracy)")
        
        num_epochs_acc = len(st.session_state.history['most_acc_topology'])
        
        if num_epochs_acc > 0:
            # Only show slider if there's more than 1 epoch
            if num_epochs_acc > 1:
                epoch_acc = st.slider(
                    "Select Epoch (Most Accurate)",
                    min_value=0,
                    max_value=num_epochs_acc - 1,
                    value=num_epochs_acc - 1,
                    key='slider_acc'
                )
            else:
                epoch_acc = 0
                st.info("Slider will appear after more epochs complete")
            
            acc_topology = st.session_state.history['most_acc_topology'][epoch_acc]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Epoch:** {epoch_acc}")
                st.markdown(f"**Topology:** {acc_topology}")
                st.markdown(f"**Total Parameters:** {sum((acc_topology[i] * acc_topology[i-1]) for i in range(1, len(acc_topology)))}")
                st.markdown(f"**Accuracy:** {st.session_state.history['most_acc'][epoch_acc]*100:.2f}%")
                if epoch_acc < len(st.session_state.history['best_perf']):
                    # Find this network's fitness (approximate)
                    st.markdown(f"**Fitness:** ~{st.session_state.history['best_perf'][epoch_acc]:.2f}")
                
                st.markdown("---")
                st.markdown("**Layer Details:**")
                for i, units in enumerate(acc_topology):
                    if i == 0:
                        st.text(f"Input: {units} units")
                    elif i == len(acc_topology) - 1:
                        st.text(f"Output: {units} units")
                    else:
                        st.text(f"Hidden {i}: {units} units")
            
            with col2:
                fig_acc = draw_topology(acc_topology, f"Most Accurate at Epoch {epoch_acc}")
                st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("Start training to visualize network architecture evolution")

with tab4:
    st.header("📈 Hyperparameter Evolution")
    
    if len(st.session_state.history['epoch']) > 0:
        # Create subplots for hyperparameters
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Learning Rate', 'Momentum', 'Weight Decay'),
            vertical_spacing=0.12
        )
        
        # Learning Rate
        fig.add_trace(
            go.Scatter(
                x=st.session_state.history['epoch'],
                y=st.session_state.history['learning_rate'],
                name='Learning Rate',
                line=dict(color='#00d4ff', width=3),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        # Momentum
        fig.add_trace(
            go.Scatter(
                x=st.session_state.history['epoch'],
                y=st.session_state.history['momentum'],
                name='Momentum',
                line=dict(color='#4ecdc4', width=3),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Decay
        fig.add_trace(
            go.Scatter(
                x=st.session_state.history['epoch'],
                y=st.session_state.history['decay'],
                name='Decay',
                line=dict(color='#ffd93d', width=3),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        fig.update_xaxes(title_text="Epoch", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=False,
            template="plotly_dark",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hyperparameter statistics
        st.subheader("Hyperparameter Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Learning Rate",
                f"{st.session_state.history['learning_rate'][-1]:.4f}",
                delta=f"{st.session_state.history['learning_rate'][-1] - st.session_state.history['learning_rate'][0]:.4f}" if len(st.session_state.history['learning_rate']) > 1 else None
            )
        
        with col2:
            st.metric(
                "Current Momentum",
                f"{st.session_state.history['momentum'][-1]:.3f}",
                delta=f"{st.session_state.history['momentum'][-1] - st.session_state.history['momentum'][0]:.3f}" if len(st.session_state.history['momentum']) > 1 else None
            )
        
        with col3:
            st.metric(
                "Current Decay",
                f"{st.session_state.history['decay'][-1]:.4f}",
                delta=f"{st.session_state.history['decay'][-1] - st.session_state.history['decay'][0]:.4f}" if len(st.session_state.history['decay']) > 1 else None
            )
    else:
        st.info("Start training to see hyperparameter evolution")

with tab5:
    st.header("🎯 Fitness Landscape")
    
    st.markdown("""
    The fitness function balances accuracy against model complexity:
    
    $$f(accuracy, size) = \\frac{X^{accuracy \\times 100}}{Y^{size}}$$
    
    Where X = 1.09 (rewards accuracy) and Y = 1.02 (penalizes model size)
    """)
    
    # Create 3D fitness landscape
    accuracy_range = np.linspace(0.5, 1.0, 50)
    size_range = np.linspace(10, 200, 50)
    
    X, Y = 1.09, 1.02
    
    acc_grid, size_grid = np.meshgrid(accuracy_range, size_range)
    fitness_grid = (X ** (acc_grid * 100)) / (Y ** size_grid)
    
    fig = go.Figure(data=[go.Surface(
        x=accuracy_range * 100,
        y=size_range,
        z=fitness_grid,
        colorscale='Viridis',
        name='Fitness'
    )])
    
    # Add current best point if available
    if st.session_state.apbt and len(st.session_state.history['epoch']) > 0:
        best_acc = st.session_state.history['best_acc'][-1] * 100
        best_size = st.session_state.history['best_size'][-1]
        best_fitness = st.session_state.history['best_perf'][-1]
        
        fig.add_trace(go.Scatter3d(
            x=[best_acc],
            y=[best_size],
            z=[best_fitness],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Current Best'
        ))
    
    fig.update_layout(
        title=dict(
            text="Fitness Landscape: Accuracy vs Model Size",
            font=dict(size=24)
        ),
        scene=dict(
            xaxis=dict(
                title=dict(text='Accuracy (%)', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(text='Model Size (parameters)', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            zaxis=dict(
                title=dict(text='Fitness', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        template="plotly_dark",
        height=900,
        font=dict(size=14)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pareto front if training has started
    if st.session_state.apbt and len(st.session_state.history['epoch']) > 0:
        st.subheader("Population: Accuracy vs Size")
        
        apbt = st.session_state.apbt
        
        scatter_data = []
        for idx, net in enumerate(apbt.population):
            scatter_data.append({
                'Network ID': idx,
                'Accuracy': apbt.accuracies[idx] * 100,
                'Size': net.num_params(),
                'Performance': apbt.perfs[idx],
                'Rank': apbt.leaderboard.index(idx) + 1
            })
        
        df_scatter = pd.DataFrame(scatter_data)
        
        fig_scatter = px.scatter(
            df_scatter,
            x='Size',
            y='Accuracy',
            size='Performance',
            color='Rank',
            hover_data=['Network ID', 'Performance'],
            color_continuous_scale='Turbo',
            labels={'Accuracy': 'Accuracy (%)', 'Size': 'Model Complexity (parameters)'}
        )
        
        fig_scatter.update_layout(
            template="plotly_dark",
            height=700,
            title=dict(
                text="Population Distribution: Accuracy vs Complexity",
                font=dict(size=22)
            ),
            xaxis=dict(
                title=dict(text="Model Complexity (parameters)", font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title=dict(text="Accuracy (%)", font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            font=dict(size=14)
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)

# Training loop - runs AFTER displaying charts for live updates
if st.session_state.training_started and not st.session_state.training_complete:
    apbt = st.session_state.apbt
    
    # Run one epoch at a time for live updates
    if st.session_state.current_epoch < epochs:
        e = st.session_state.current_epoch
        
        # Log epoch start
        if e == 0:
            logger.info("-"*80)
            logger.info("TRAINING STARTED")
            logger.info("-"*80)
        
        exploitation_count = 0
        
        # Train one epoch
        for i in range(apbt.k):
            net = apbt.population[i]
            hyperparams = apbt.hyperparams[i]
            last = apbt.last_ready[i]
            
            # Optimize the net
            net = apbt.step(net)
            perf, accuracy = apbt.evaluate(net)
            
            apbt.perfs[i] = perf
            apbt.accuracies[i] = accuracy
            apbt.update_leaderboard()
            
            # Exploit and explore
            if apbt.is_ready(last, e, i):
                exploitation_count += 1
                new_net, new_hyperparams = apbt.exploit(net, hyperparams)
                if apbt.is_diff(new_net, net):
                    net, hyperparams = apbt.explore(new_net, new_hyperparams)
                    net.set_hyperparameters(hyperparams)
                    perf, accuracy = apbt.evaluate(net)
                    apbt.perfs[i] = perf
                    apbt.accuracies[i] = accuracy
                    apbt.update_leaderboard()
            
            apbt.population[i] = net
            apbt.hyperparams[i] = hyperparams
        
        # Update best
        best = apbt.get_best()
        most_acc = apbt.get_most_accurate()
        
        # Log epoch results
        logger.info(f"Epoch {e+1}/{epochs} | "
                   f"Best Perf: {best[1]:.2f} | "
                   f"Best Acc: {best[2]*100:.2f}% | "
                   f"Most Acc: {most_acc[2]*100:.2f}% | "
                   f"Avg Acc: {np.mean(apbt.accuracies)*100:.2f}% | "
                   f"Size: {best[0].num_params()} params"
                   f"{' | Exploitations: ' + str(exploitation_count) if exploitation_count > 0 else ''}")
        
        # Store history
        st.session_state.history['epoch'].append(e)
        st.session_state.history['best_perf'].append(best[1])
        st.session_state.history['best_acc'].append(best[2])
        st.session_state.history['best_size'].append(best[0].num_params())
        st.session_state.history['most_acc'].append(most_acc[2])
        st.session_state.history['avg_perf'].append(np.mean(apbt.perfs))
        st.session_state.history['avg_acc'].append(np.mean(apbt.accuracies))
        st.session_state.history['learning_rate'].append(best[3]['learning_rate'])
        st.session_state.history['momentum'].append(best[3]['momentum'])
        st.session_state.history['decay'].append(best[3]['decay'])
        st.session_state.history['best_topology'].append(best[0].topology.copy())
        st.session_state.history['most_acc_topology'].append(most_acc[0].topology.copy())
        
        st.session_state.best_net = best[0]
        st.session_state.current_epoch += 1
        
        # Rerun to show updated charts
        st.rerun()
    else:
        if not st.session_state.training_complete:
            # Log training completion (only once)
            logger.info("-"*80)
            logger.info("TRAINING COMPLETE!")
            logger.info("-"*80)
            
            # Get final results
            apbt = st.session_state.apbt
            best = apbt.get_best()
            most_acc = apbt.get_most_accurate()
            
            logger.info("Final Results:")
            logger.info(f"  Best Performer:")
            logger.info(f"    - Topology: {best[0].topology}")
            logger.info(f"    - Fitness: {best[1]:.2f}")
            logger.info(f"    - Accuracy: {best[2]*100:.2f}%")
            logger.info(f"    - Parameters: {best[0].num_params()}")
            logger.info(f"    - Learning Rate: {best[3]['learning_rate']:.4f}")
            logger.info(f"    - Momentum: {best[3]['momentum']:.3f}")
            logger.info(f"    - Decay: {best[3]['decay']:.4f}")
            
            logger.info(f"  Most Accurate:")
            logger.info(f"    - Topology: {most_acc[0].topology}")
            logger.info(f"    - Accuracy: {most_acc[2]*100:.2f}%")
            logger.info(f"    - Fitness: {most_acc[1]:.2f}")
            logger.info(f"    - Parameters: {most_acc[0].num_params()}")
            
            logger.info(f"  Population Stats:")
            logger.info(f"    - Average Accuracy: {np.mean(apbt.accuracies)*100:.2f}%")
            logger.info(f"    - Average Performance: {np.mean(apbt.perfs):.2f}")
            logger.info(f"    - Best/Worst Ratio: {max(apbt.perfs)/max(min(apbt.perfs), 0.001):.2f}x")
            
            logger.info("="*80)
        
        st.session_state.training_complete = True
        st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧬 Augmented Population-Based Training | Evolving Better Neural Networks</p>
</div>
""", unsafe_allow_html=True)

