# Transformer Lab

Exercise 1 & 2 can be found into the "transformer_1-2.ipynb" notebook

* Last picked exercise 3.2 is implemented into the two python file "transformer_3_2_Lora_fine_tuned.py" & "transformer_3_2_metrics.py" (both use function from utility.py)
* Dataset: https://huggingface.co/datasets/AI-Lab-Makerere/beans (I know is small but is useful to train all the part of CLIP separately)
* Run "transformer_3_2_Lora_fine_tuned.py" to fine-tune the CLIP model using Lora, to perform the fine-tune a classification head is added to CLIP and trained on the new dataset
* Run "transformer_3_2_metrics.py" to get the metric for fine-tuned model and zero shot prediction before training (Note that if no mode is provided only perform zero-shot metric)
* [This file load an existing fine-tuned checkpoint, rename the folder of the model you want to use to "best_{mode}" (eg. beans_text_vision\best_hybrid)]

* All configuration and hyperparameter are stored into Config/default.yaml edit this file to change mode.
  [In detail mode refer to how the model get fine tuned:
  
  text -> only CLIP text encoder get fine-tuned and the rest is freezed,
  
  vision -> only CLIP visual encoder get fine-tuned and the rest is freezed,
  
  hybrid -> both CLIP visual and text encoder get fine-tuned]
