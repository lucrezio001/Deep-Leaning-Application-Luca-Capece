import torch
import torch.nn as nn
import yaml
from transformers import AutoProcessor, AutoModel
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, get_peft_model

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
data_collator = DataCollatorWithPadding(tokenizer=processor.tokenizer, return_tensors="pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = {
        "angular_leaf_spot": 0,
        "bean_rust": 1,
        "healthy": 2,
    }

config_path = "Config/default.yaml"

with open(config_path) as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

class CLIP_Adapter(nn.Module):
    def __init__(self, base_model, num_classes = 10, input_dim = 1024):
        super().__init__()
        self.clip = base_model
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes)
        )
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        embeddings = []
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        embeddings.append(image_features)
        text_features = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        embeddings.append(text_features)

        features = torch.cat(embeddings, dim=-1)

        logits = self.classifier(features)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

def get_clip_target_modules(model):
    vision_target_modules = []
    text_target_modules = []
    classifier_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "vision_model" in name:
                vision_target_modules.append(name)
            elif "text_model" in name:
                text_target_modules.append(name)
            elif "classifier" in name:
                classifier_modules.append(name)
    
    return vision_target_modules, text_target_modules, classifier_modules

def clip_lora_model(
    mode,
    device = "cuda" if torch.cuda.is_available() else "cpu",
    adapter = CLIP_Adapter,
    base_model_name = "openai/clip-vit-base-patch16",
    config = config
):
    base_model = AutoModel.from_pretrained(base_model_name)
    model = adapter(base_model=base_model).to(device)

    vision_modules, text_modules, classifier_modules = get_clip_target_modules(model)

    if mode == "text":
        target_parts = ["text", "classifier"]
    elif mode == "vision":
        target_parts = ["vision", "classifier"]
    elif mode == "hybrid":
        target_parts = ["text", "vision", "classifier"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    target_modules = []
    if "vision" in target_parts:
        target_modules += vision_modules
    if "text" in target_parts:
        target_modules += text_modules
    if "classifier" in target_parts:
        target_modules += classifier_modules

    lora_args = config["lora_args"]
    lora_config = LoraConfig(
        task_type = lora_args["task_type"],
        inference_mode = False,
        r = lora_args["r"],
        lora_alpha = lora_args["alpha"],
        lora_dropout = lora_args["dropout"],
        target_modules = target_modules
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    return lora_model

def preprocess(dataset):
    prompts = [f'A photo of a {label}' for label in dataset['labels']]
    processed = processor(
        text=prompts,
        images=dataset['image'],
        return_tensors="pt",
        padding = True,
        truncation=True)
    processed['labels'] = dataset['labels']
    return processed

def get_training_args(mode, config):
    
    save_conf = config['save_folders']
    
    if mode == "text":
        output_dir = save_conf['text_folder']
    elif mode == "vision":
        output_dir = save_conf['vision_folder']
    elif mode == "hybrid":
        output_dir = save_conf['text_vision_folder']
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ta = config['training_args']

    training_args = TrainingArguments(
        output_dir = output_dir,
        eval_strategy = ta['eval_strategy'],
        per_device_train_batch_size = ta['per_device_train_batch_size'],
        per_device_eval_batch_size = ta['per_device_eval_batch_size'],
        optim = ta['optim'],
        learning_rate = ta['lr'],
        num_train_epochs = ta['epochs'],
        save_strategy = ta['save_strategy'],
        seed = ta['seed'],
        metric_for_best_model = ta['metric_for_best_model'],
        label_names = ta['label_names'],
        dataloader_pin_memory = ta['dataloader_pin_memory'],
        tf32 = ta['tf32']
    )
    return training_args

def get_trainer(model, training_args, train_dataset, eval_dataset, 
                data_collator = data_collator):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    

def compute_final_metrics(predictions, labels):
    acc = accuracy_score(labels, predictions)
    # macro because we have aprox. 10% of dataset for each class (no imbalance)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division = 0.0)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
def compute_final_metrics_from_logits(logits, labels):
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division = 0.0)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }