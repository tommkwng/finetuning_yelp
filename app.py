import streamlit as st
import pandas as pd
import numpy as np
import torch
import zipfile
import os
import tempfile
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
from huggingface_hub import login
import json

# Set page configuration
st.set_page_config(
    page_title="Transformer Model Fine-tuning Tool",
    page_icon="🤖",
    layout="wide"
)

# App title
st.title("🤖 Transformer Model Fine-tuning Tool")
st.markdown("""
This application allows you to fine-tune models using Hugging Face datasets and models.
You can customize the dataset, model, and training parameters.
""")

# Initialize session state variables
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration Settings")
    
    # Hugging Face Token (optional)
    hf_token = st.text_input("Hugging Face Token (optional)", type="password", 
                           help="Enter your Hugging Face token if you need to access private datasets or models")
    
    if hf_token:
        try:
            login(token=hf_token)
            st.success("Hugging Face login successful!")
        except Exception as e:
            st.warning(f"Login failed: {e}. Continuing with public datasets.")
    
    st.divider()
    
    # Dataset Configuration
    st.subheader("📊 Dataset Configuration")
    
    dataset_name = st.text_input(
        "Dataset Name",
        value="yelp_review_full",
        help="Examples: yelp_review_full, imdb, emotion, etc."
    )
    
    # Try to get dataset info
    available_splits = ["train", "test", "validation"]
    try:
        if st.button("🔍 Check Dataset Availability"):
            with st.spinner("Checking dataset availability..."):
                dataset_info = load_dataset(dataset_name, trust_remote_code=True)
                available_splits = list(dataset_info.keys())
                st.success(f"Dataset found: {dataset_name}")
                st.info(f"Available splits: {', '.join(available_splits)}")
    except Exception as e:
        st.error(f"Unable to check dataset: {e}")
        st.info("Using default splits: train, test, validation")
    
    # Dataset split selection
    col1, col2 = st.columns(2)
    with col1:
        train_split = st.selectbox(
            "Train Split",
            available_splits,
            index=0 if "train" in available_splits else 0
        )
    
    with col2:
        test_split = st.selectbox(
            "Test Split",
            available_splits,
            index=1 if len(available_splits) > 1 else 0
        )
    
    # Dataset size settings
    col1, col2 = st.columns(2)
    with col1:
        train_size = st.number_input(
            "Train Dataset Size",
            min_value=100,
            max_value=10000,
            value=5000,
            step=100
        )
    
    with col2:
        test_size = st.number_input(
            "Test Dataset Size",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
    
    st.divider()
    
    # Model Configuration
    st.subheader("🧠 Model Configuration")
    
    model_name = st.text_input(
        "Model Name",
        value="distilbert/distilbert-base-uncased",
        help="Examples: distilbert/distilbert-base-uncased, bert-base-uncased, etc."
    )
    
    tokenizer_name = st.text_input(
        "Tokenizer Name",
        value="distilbert/distilbert-base-uncased",
        help="Usually the same as model name"
    )
    
    st.divider()
    
    # Training Configuration
    st.subheader("🚀 Training Configuration")
    
    epochs = st.slider(
        "Number of Epochs",
        min_value=1,
        max_value=5,
        value=1
    )
    
    batch_size = st.selectbox(
        "Batch Size",
        [8, 16, 32, 64],
        index=1
    )
    
    learning_rate = st.number_input(
        "Learning Rate",
        min_value=1e-6,
        max_value=1e-3,
        value=2e-5,
        format="%.6f"
    )
    
    output_dir = st.text_input(
        "Output Directory",
        value="./fine_tuned_model"
    )

# Main area
tab1, tab2, tab3 = st.tabs(["📥 Data Loading", "⚡ Model Training", "📤 Model Testing"])

with tab1:
    st.header("Data Loading & Preprocessing")
    
    if st.button("🚀 Load Dataset", type="primary"):
        with st.spinner("Loading dataset..."):
            try:
                # Load dataset
                train_dataset = load_dataset(
                    dataset_name, 
                    split=f"{train_split}[:{train_size}]",
                    trust_remote_code=True
                )
                test_dataset = load_dataset(
                    dataset_name, 
                    split=f"{test_split}[:{test_size}]",
                    trust_remote_code=True
                )
                
                # Create DatasetDict
                dataset = DatasetDict({
                    "train": train_dataset,
                    "test": test_dataset
                })
                
                # Show dataset information
                st.success("Dataset loaded successfully!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Samples", len(train_dataset))
                with col2:
                    st.metric("Test Samples", len(test_dataset))
                
                # Display dataset structure
                with st.expander("📋 Dataset Structure", expanded=True):
                    st.json({
                        "train": {
                            "features": list(train_dataset.features.keys()),
                            "num_rows": len(train_dataset)
                        },
                        "test": {
                            "features": list(test_dataset.features.keys()),
                            "num_rows": len(test_dataset)
                        }
                    })
                
                # Detect number of labels
                if "label" in train_dataset.features:
                    num_labels = len(set(train_dataset["label"]))
                    with col3:
                        st.metric("Number of Labels", num_labels)
                    
                    # Show label distribution
                    st.subheader("📊 Label Distribution")
                    
                    # Training set label distribution
                    train_labels = pd.Series(train_dataset["label"]).value_counts().sort_index()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(train_labels, use_container_width=True)
                    with col2:
                        st.dataframe(
                            train_labels.reset_index().rename(
                                columns={"index": "Label", 0: "Count"}
                            ),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    # Save to session state
                    st.session_state.dataset = dataset
                    st.session_state.num_labels = num_labels
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset_name = dataset_name
                else:
                    st.error("No 'label' field found in the dataset")
                    st.session_state.dataset_loaded = False
                    
                # Show sample examples
                st.subheader("🔍 Sample Data Examples")
                sample_data = []
                for i in range(min(5, len(train_dataset))):
                    sample = {k: train_dataset[i][k] for k in train_dataset.features.keys()}
                    sample_data.append(sample)
                
                st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                st.code(str(e), language="bash")
                st.session_state.dataset_loaded = False

with tab2:
    st.header("Model Training & Evaluation")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load a dataset first in the 'Data Loading' tab")
    else:
        # Model Initialization
        st.subheader("🔧 Model Initialization")
        
        if st.button("Initialize Model and Tokenizer", type="primary"):
            with st.spinner("Loading model and tokenizer..."):
                try:
                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=st.session_state.num_labels
                    )
                    
                    st.success("✅ Model and Tokenizer loaded successfully!")
                    
                    # Save to session state
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.session_state.model_name = model_name
                    
                    # Display model information
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", model_name.split("/")[-1])
                    with col2:
                        st.metric("Tokenizer", tokenizer_name.split("/")[-1])
                    with col3:
                        st.metric("Labels", st.session_state.num_labels)
                    
                    # Display model configuration
                    with st.expander("🔍 View Model Configuration"):
                        config_dict = model.config.to_dict()
                        st.json(config_dict)
                        
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.session_state.model_loaded = False
        
        # Data Preprocessing
        if st.session_state.model_loaded:
            st.subheader("🔄 Data Preprocessing")
            
            if st.button("Preprocess Data", type="primary"):
                with st.spinner("Preprocessing data..."):
                    try:
                        # Define tokenize function
                        def tokenize_function(examples):
                            # Check if text field exists, if not use the first field
                            if "text" in examples:
                                text_field = "text"
                            elif "review" in examples:
                                text_field = "review"
                            elif "content" in examples:
                                text_field = "content"
                            else:
                                # Use the first non-label field
                                available_fields = [k for k in examples.keys() if k != "label"]
                                text_field = available_fields[0] if available_fields else list(examples.keys())[0]
                            
                            return st.session_state.tokenizer(
                                examples[text_field],
                                padding="max_length",
                                truncation=True,
                                max_length=512
                            )
                        
                        # Get column names to remove (all except label)
                        columns_to_remove = [
                            col for col in st.session_state.dataset["train"].column_names 
                            if col != "label"
                        ]
                        
                        # Apply tokenization
                        tokenized_datasets = st.session_state.dataset.map(
                            tokenize_function,
                            batched=True,
                            remove_columns=columns_to_remove  # Only remove non-label columns
                        )
                        
                        # Set format for PyTorch
                        tokenized_datasets.set_format("torch")
                        
                        st.success("✅ Data preprocessing completed!")
                        
                        # Show tokenized example
                        with st.expander("🔍 View Tokenized Example"):
                            example = tokenized_datasets["train"][0]
                            st.json({
                                "input_ids (first 10)": example["input_ids"][:10].tolist(),
                                "attention_mask (first 10)": example["attention_mask"][:10].tolist(),
                                "label": example["label"].item() if hasattr(example["label"], 'item') else example["label"]
                            })
                        
                        # Save to session state
                        st.session_state.tokenized_datasets = tokenized_datasets
                        st.session_state.data_preprocessed = True
                        
                    except Exception as e:
                        st.error(f"Error during data preprocessing: {e}")
                        import traceback
                        with st.expander("View Error Details"):
                            st.code(traceback.format_exc())
        
        # Model Training
        if st.session_state.data_preprocessed:
            st.subheader("🏃‍♂️ Model Training")
            
            # Training parameters summary
            with st.expander("📋 Training Parameters Summary"):
                params_table = {
                    "Parameter": ["Dataset", "Model", "Train Size", "Test Size", 
                                 "Epochs", "Batch Size", "Learning Rate", "Labels"],
                    "Value": [str(st.session_state.dataset_name),  # Convert to string
                             str(st.session_state.model_name.split("/")[-1]),
                             str(len(st.session_state.tokenized_datasets["train"])),  # Convert to string
                             str(len(st.session_state.tokenized_datasets["test"])),
                             str(epochs),  # Convert to string
                             str(batch_size),
                             str(learning_rate),  # Convert to string
                             str(st.session_state.num_labels)]  # Convert to string
                }
                # Create DataFrame with explicit dtype
                df_params = pd.DataFrame(params_table)
                # Ensure all values are strings
                df_params['Value'] = df_params['Value'].astype(str)
                st.table(df_params)
            
            if st.button("Start Fine-tuning", type="primary"):
                # Create output directory
                os.makedirs(output_dir, exist_ok=True)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Define compute metrics function
                    def compute_metrics(eval_pred):
                        logits, labels = eval_pred
                        predictions = np.argmax(logits, axis=-1)
                        accuracy_metric = evaluate.load("accuracy")
                        return accuracy_metric.compute(predictions=predictions, references=labels)
                    
                    # Set training arguments

                    # Set training arguments
                    training_args = TrainingArguments(
                        output_dir=output_dir,
                        num_train_epochs=epochs,
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        learning_rate=learning_rate,
                        eval_strategy="epoch",  # Changed from evaluation_strategy
                        save_strategy="epoch",  # Keep this as is
                        logging_dir=f'{output_dir}/logs',
                        logging_steps=10,
                        load_best_model_at_end=True,
                        metric_for_best_model="accuracy",
                        greater_is_better=True,
                        report_to="none",
                        save_total_limit=2,
                        fp16=torch.cuda.is_available(),
                    )
                                    
                    # Create Trainer
                    trainer = Trainer(
                        model=st.session_state.model,
                        args=training_args,
                        train_dataset=st.session_state.tokenized_datasets["train"],
                        eval_dataset=st.session_state.tokenized_datasets["test"],
                        compute_metrics=compute_metrics,
                    )
                    
                    # Train the model
                    status_text.text("Starting training...")
                    train_result = trainer.train()
                    
                    st.success("🎉 Model training completed!")
                    
                    # Display training results
                    st.subheader("📊 Training Results")
                    
                    # Format train output
                    train_output = {
                        "global_step": train_result.global_step,
                        "training_loss": float(train_result.training_loss),
                        "metrics": train_result.metrics
                    }
                    
                    st.code(f"TrainOutput({train_output})", language="python")
                    
                    # Evaluate the model
                    st.subheader("📈 Evaluation Results")
                    eval_result = trainer.evaluate()
                    
                    # Format evaluation results
                    eval_output = {
                        'eval_loss': eval_result['eval_loss'],
                        'eval_accuracy': eval_result['eval_accuracy'],
                        'eval_runtime': eval_result['eval_runtime'],
                        'eval_samples_per_second': eval_result['eval_samples_per_second'],
                        'eval_steps_per_second': eval_result['eval_steps_per_second'],
                        'epoch': eval_result['epoch']
                    }
                    
                    st.code(json.dumps(eval_output, indent=2), language="json")
                    
                    # Save the model
                    trainer.save_model(output_dir)
                    st.success(f"✅ Model saved to: {output_dir}")
                    
                    # Save to session state
                    st.session_state.trainer = trainer
                    st.session_state.model_trained = True
                    st.session_state.output_dir = output_dir
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Evaluation Loss", f"{eval_result['eval_loss']:.4f}")
                    with col2:
                        st.metric("Evaluation Accuracy", f"{eval_result['eval_accuracy']:.4f}")
                    with col3:
                        st.metric("Training Time", f"{train_result.metrics.get('train_runtime', 0):.2f}s")
                    
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("Training completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    import traceback
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())
        
        # Model Download Section
        if st.session_state.model_trained:
            st.subheader("📦 Model Download")
            
            # Create ZIP file function
            def create_zip_file(model_dir, zip_name):
                zip_path = f"{zip_name}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, start=model_dir)
                            zipf.write(file_path, arcname)
                return zip_path
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Package Model for Download"):
                    with st.spinner("Packaging model..."):
                        try:
                            zip_path = create_zip_file(
                                st.session_state.output_dir,
                                "fine_tuned_model"
                            )
                            st.session_state.zip_path = zip_path
                            st.success("✅ Model packaged successfully!")
                        except Exception as e:
                            st.error(f"Error packaging model: {e}")
            
            with col2:
                if 'zip_path' in st.session_state and os.path.exists(st.session_state.zip_path):
                    with open(st.session_state.zip_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Model (ZIP)",
                            data=f,
                            file_name="fine_tuned_model.zip",
                            mime="application/zip",
                            type="primary"
                        )

with tab3:
    st.header("Model Testing")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Model Training' tab")
    else:
        st.subheader("🔍 Test the Fine-tuned Model")
        
        # Test with sample text
        test_text = st.text_area(
            "Enter text for testing",
            value="dr. goldberg offers everything i look for in a general practitioner. he's nice and easy to talk to without being patronizing.",
            height=100,
            help="Enter any text you want to test the model with"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            predict_button = st.button("🔍 Make Prediction", type="primary", use_container_width=True)
        
        if predict_button and test_text:
            with st.spinner("Making prediction..."):
                try:
                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        st.session_state.output_dir,
                        num_labels=st.session_state.num_labels
                    )
                    
                    # Preprocess input
                    inputs = tokenizer(
                        test_text,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predictions_np = predictions.cpu().numpy()
                    
                    # Get predicted class
                    predicted_class = np.argmax(predictions_np[0])
                    
                    # Display results
                    st.success(f"**Predicted Class: {predicted_class}**")
                    
                    # Show probability distribution
                    st.subheader("📊 Class Probability Distribution")
                    
                    # Create probability dataframe
                    prob_df = pd.DataFrame({
                        "Class": range(st.session_state.num_labels),
                        "Probability": predictions_np[0]
                    })
                    
                    # Sort by probability
                    prob_df = prob_df.sort_values("Probability", ascending=False)
                    
                    # Display table
                    st.dataframe(
                        prob_df.style.format({"Probability": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Display bar chart
                    st.bar_chart(prob_df.set_index("Class")["Probability"], use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Footer
st.divider()
st.caption("""
Note: This application uses Hugging Face's Transformers and Datasets libraries.
Training time depends on dataset size, model complexity, and hardware configuration.
""")

# Add some styling
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .stButton > button {
        width: 100%;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
