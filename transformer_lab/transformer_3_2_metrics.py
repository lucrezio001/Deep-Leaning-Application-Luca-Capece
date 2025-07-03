# Comparison between fine tuned and zero shot performance

from transformers import AutoProcessor, AutoModel
from datasets import load_dataset
from transformers import pipeline
import yaml
import torch
from utility import clip_lora_model, preprocess, get_training_args, get_trainer,compute_final_metrics, compute_final_metrics_from_logits
import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    config_path = "Config/default.yaml"

    with open(config_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # beans leaf images
    train_dataset = load_dataset("AI-Lab-Makerere/beans", split="train")
    valid_dataset = load_dataset("AI-Lab-Makerere/beans", split="validation")
    test_dataset  = load_dataset("AI-Lab-Makerere/beans", split="test")

    classes = {
        "angular_leaf_spot": 0,
        "bean_rust": 1,
        "healthy": 2,
    }
    # zero-shot prompt
    prompts = [f'A photo of a {cls}' for cls in classes]

    if config['mode'] == "text" or "vision" or "hybrid":
        
        lora_model = clip_lora_model(config['mode'])
        lora_model.to(device)

        train_dataset = train_dataset.map(preprocess, batched=True)
        validation_dataset = valid_dataset.map(preprocess, batched=True)
        test_dataset = test_dataset.map(preprocess, batched=True)

        trainer = get_trainer(lora_model, get_training_args(config['mode'], config), train_dataset, validation_dataset)

        save_conf = config['save_folders']
    
        if config['mode'] == "text":
            save_dir = save_conf['text_folder']
        elif config['mode'] == "vision":
            save_dir = save_conf['vision_folder']
        elif config['mode'] == "hybrid":
            save_dir = save_conf['text_vision_folder']

        save_dir = os.path.join(save_dir, f"best_{config['mode']}")

        trainer._load_from_checkpoint(save_dir)

        trainer.evaluate()
        
        prediction_train = trainer.predict(validation_dataset)
        metrics_train = compute_final_metrics_from_logits(prediction_train.predictions, validation_dataset['labels'])
        prediction_validation = trainer.predict(validation_dataset)
        metrics_validation = compute_final_metrics_from_logits(prediction_validation.predictions, validation_dataset['labels'])
        prediction_test = trainer.predict(test_dataset)
        metrics_test = compute_final_metrics_from_logits(prediction_test.predictions, test_dataset['labels'])
        
        print(f'Result for fine tuned, mode: {config['mode']}')
        print("Train Metrics:", metrics_train)
        print("Validation Metrics:", metrics_validation)
        print("Test Metrics:", metrics_test)

    # model and processor
    
    model_name = "openai/clip-vit-base-patch16"
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch16")
    model = model.to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)

    # zero-shot image classification pipeline
    clip = pipeline(
        task='zero-shot-image-classification',
        model = model_name,
        processor = processor,
        device = 0,
        seed = 42
    )

    def predictions(batch):
        scores = clip(batch['image'], candidate_labels=prompts)
        pred_labels = [s[0]['label'].replace('A photo of a ', '') for s in scores]
        batch['predicted_label'] = [classes[label] for label in pred_labels]
        return batch

    # batch to speed up a little (Extremely slow anyway)

    train_dataset = train_dataset.map(predictions, batched=True)
    valid_dataset = valid_dataset.map(predictions, batched=True)
    test_dataset  = test_dataset.map(predictions, batched=True)

    print(f'Result for zero shot:')
    # Results
    print("Train metrics:", compute_final_metrics(train_dataset['predicted_label'], train_dataset['labels']))
    print("Validation metrics:", compute_final_metrics(valid_dataset['predicted_label'], valid_dataset['labels']))
    print("Test metrics:", compute_final_metrics(test_dataset['predicted_label'], test_dataset['labels']))

    # Train metrics: {'accuracy': 0.3346228239845261, 'f1': 0.16963341774641652, 'precision': 0.1537524366471735, 'recall': 0.3342911877394636}
    # Validation metrics: {'accuracy': 0.3383458646616541, 'f1': 0.18115942028985507, 'precision': 0.4444444444444444, 'recall': 0.34074074074074073}
    # Test metrics: {'accuracy': 0.3359375, 'f1': 0.16764132553606237, 'precision': 0.11197916666666667, 'recall': 0.3333333333333333}