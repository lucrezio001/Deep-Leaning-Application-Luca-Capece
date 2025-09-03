from datasets import load_dataset
import torch
from utility import clip_lora_model, preprocess, get_training_args, get_trainer
import yaml
import warnings
import os
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    os.environ["WANDB_PROJECT"] = "Lab3_transformer_3.2_clip"
    os.environ["WANDB_LOG_MODEL"]="checkpoint"
    os.environ["WANDB_WATCH"]="false"
    
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
    
    # Setup lora on clip (not all layer are supported)
    lora_model = clip_lora_model(config['mode'])
    lora_model.to(device)

    train_dataset = train_dataset.map(preprocess, batched=True)
    validation_dataset = valid_dataset.map(preprocess, batched=True)
    #test_dataset = test_dataset.map(preprocess, batched=True)

    trainer = get_trainer(lora_model, get_training_args(config['mode'], config), train_dataset, validation_dataset)

    trainer.train()
