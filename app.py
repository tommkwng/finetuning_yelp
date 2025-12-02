# Main workflow
st.header("📋 Workflow Summary")

# Display configuration summary
with st.expander("📋 View Configuration", expanded=True):
    config_table = {
        "Setting": ["Dataset", "Train Size", "Test Size", "Model", 
                   "Epochs", "Batch Size", "Learning Rate", "Output Directory"],
        "Value": [dataset_name, train_size, test_size, model_name,
                 epochs, batch_size, f"{learning_rate:.6f}", output_dir]
    }
    
    config_df = pd.DataFrame(config_table)
    # Ensure all values are strings for display
    for col in config_df.columns:
        config_df[col] = config_df[col].astype(str)
    
    st.table(config_df)

# Single start button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_button = st.button("🚀 Start Fine-tuning Process", type="primary", use_container_width=True)

if start_button:
    # Create progress containers
    progress_container = st.container()
    status_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.subheader("📊 Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Initialize error flag
    error_occurred = False
    
    # Step 1: Load Dataset
    with status_container:
        st.subheader("🔍 Step 1: Loading Dataset")
        status_text.text("Loading dataset...")
        
        try:
            with st.spinner("Loading dataset..."):
                # Load dataset
                train_dataset = load_dataset(
                    dataset_name, 
                    split=f"train[:{train_size}]"
                )
                test_dataset = load_dataset(
                    dataset_name, 
                    split=f"test[:{test_size}]"
                )
                
                # Create DatasetDict
                dataset = DatasetDict({
                    "train": train_dataset,
                    "test": test_dataset
                })
                
                st.success("✅ Dataset loaded successfully!")
                
                # Display dataset info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Samples", len(train_dataset))
                with col2:
                    st.metric("Test Samples", len(test_dataset))
                
                # Detect label column
                def get_label_info(dataset):
                    """Safely extract label information from dataset."""
                    possible_label_columns = ['label', 'labels', 'Label', 'Labels', 'class', 'Class', 
                                            'rating', 'Rating', 'sentiment', 'Sentiment', 'score', 'Score']
                    
                    for label_col in possible_label_columns:
                        if label_col in dataset.features:
                            try:
                                unique_labels = len(set(dataset[label_col]))
                                return label_col, unique_labels
                            except Exception:
                                continue
                    
                    # Try any column
                    for col in dataset.features:
                        try:
                            if isinstance(dataset[col][0], (int, float, str)):
                                unique_labels = len(set(dataset[col]))
                                return col, unique_labels
                        except Exception:
                            continue
                    
                    return None, None
                
                label_column, num_labels = get_label_info(train_dataset)
                
                if label_column and num_labels is not None:
                    st.metric("Number of Labels", num_labels)
                    
                    # Save to session state
                    st.session_state.dataset = dataset
                    st.session_state.num_labels = num_labels
                    st.session_state.label_column = label_column
                    
                    progress_bar.progress(20)
                    status_text.text("Dataset loaded. Initializing model...")
                    
                else:
                    st.error("No suitable label column found in the dataset")
                    st.info(f"Available columns: {list(train_dataset.features.keys())}")
                    error_occurred = True
                    
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            import traceback
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())
            error_occurred = True
    
    # If error occurred in step 1, stop execution
    if error_occurred:
        st.error("❌ Process stopped due to error in dataset loading.")
        st.stop()
    
    # Step 2: Initialize Model and Tokenizer
    st.subheader("🔧 Step 2: Initializing Model and Tokenizer")
    status_text.text("Loading model and tokenizer...")
    
    try:
        with st.spinner("Loading model and tokenizer..."):
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=st.session_state.num_labels
            )
            
            st.success("✅ Model and Tokenizer loaded successfully!")
            
            # Display model info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", model_name.split("/")[-1])
            with col2:
                st.metric("Tokenizer", tokenizer_name.split("/")[-1])
            
            # Save to session state
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            
            progress_bar.progress(40)
            status_text.text("Model initialized. Preprocessing data...")
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        with st.expander("View Error Details"):
            st.code(traceback.format_exc())
        st.error("❌ Process stopped due to error in model initialization.")
        st.stop()
    
    # Step 3: Data Preprocessing
    st.subheader("🔄 Step 3: Data Preprocessing")
    status_text.text("Preprocessing data...")
    
    try:
        with st.spinner("Preprocessing data..."):
            # Define tokenize function
            def tokenize_function(examples):
                # Get text column
                text_field = None
                possible_text_fields = ['text', 'review', 'content', 'sentence', 'comment', 
                                      'article', 'description', 'summary']
                
                for field in possible_text_fields:
                    if field in examples:
                        text_field = field
                        break
                
                # If no common text field found, use first non-label field
                if text_field is None:
                    available_fields = [k for k in examples.keys() 
                                      if k != st.session_state.label_column]
                    if available_fields:
                        text_field = available_fields[0]
                    else:
                        text_field = list(examples.keys())[0]
                
                return st.session_state.tokenizer(
                    examples[text_field],
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            
            # Get column names to remove (all except label column)
            columns_to_remove = [
                col for col in st.session_state.dataset["train"].column_names 
                if col != st.session_state.label_column
            ]
            
            # Apply tokenization
            tokenized_datasets = st.session_state.dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=columns_to_remove
            )
            
            # Rename label column to 'labels' for compatibility
            if st.session_state.label_column != 'labels':
                def rename_labels(example):
                    example['labels'] = example[st.session_state.label_column]
                    return example
                
                tokenized_datasets = tokenized_datasets.map(rename_labels)
            
            # Set format for PyTorch
            tokenized_datasets.set_format("torch")
            
            st.success("✅ Data preprocessing completed!")
            
            # Save to session state
            st.session_state.tokenized_datasets = tokenized_datasets
            
            progress_bar.progress(60)
            status_text.text("Data preprocessed. Starting training...")
            
    except Exception as e:
        st.error(f"Error during data preprocessing: {e}")
        import traceback
        with st.expander("View Error Details"):
            st.code(traceback.format_exc())
        st.error("❌ Process stopped due to error in data preprocessing.")
        st.stop()
    
    # Step 4: Model Training
    st.subheader("🏃‍♂️ Step 4: Model Training")
    status_text.text("Training model...")
    
    try:
        with st.spinner("Training model. This may take several minutes..."):
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Define compute metrics function
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                accuracy_metric = evaluate.load("accuracy")
                return accuracy_metric.compute(predictions=predictions, references=labels)
            
            # Set training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                eval_strategy="epoch",
                save_strategy="epoch",
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
            train_result = trainer.train()
            
            # Evaluate the model
            eval_result = trainer.evaluate()
            
            # Save the model
            trainer.save_model(output_dir)
            
            st.success("🎉 Model training completed!")
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("Training completed successfully!")
            
            # Save to session state
            st.session_state.trainer = trainer
            st.session_state.model_trained = True
            st.session_state.training_complete = True
            st.session_state.train_result = train_result
            st.session_state.eval_result = eval_result
            
    except Exception as e:
        st.error(f"Error during training: {e}")
        import traceback
        with st.expander("View Error Details"):
            st.code(traceback.format_exc())
        st.error("❌ Process stopped due to error in training.")
        st.stop()
