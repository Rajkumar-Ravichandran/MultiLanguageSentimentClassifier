# MultiLanguageSentimentClassifier
This repository contains a multi-language sentiment classifier built using a fine-tuned causal language model with LoRA (Low-Rank Adaptation) and 8-bit quantization for efficiency. The model is capable of analyzing sentiment in various Indian languages.

**Features**
- Uses transformers for natural language processing
- LoRA fine-tuning for efficient adaptation
- 8-bit quantization to reduce memory usage
- Supports multiple Indian languages
- Instruction-based prompts for sentiment classification

**Installation**

Ensure you have Python 3.8+ installed. Then, install the required dependencies:

- pip install --upgrade pip
- pip install torch torchvision torchaudio
- pip install transformers accelerate bitsandbytes
- pip install git+https://github.com/huggingface/peft.git

**Dataset**

The model requires a labeled dataset in CSV format with the following structure in Indian languages (but technically any language is fine):

ID,sentence,label
1,"This is a great product!",Positive
2,"I hated the experience.",Negative

Update the file paths accordingly:

Training Data: /train.csv

Testing Data: /test.csv

**Training**
- Load and preprocess the dataset.
- Split data into train and validation sets.
- Tokenize inputs using AutoTokenizer.
- Apply LoRA fine-tuning on a pre-trained causal language model.
- Train using Hugging Face Trainer API.
- Run the script containing the training code:


**Inference**

To generate predictions on new text inputs:
- Load the trained model.
- Tokenize input sentences.
- Use model inference to classify sentiment.


**Model Checkpoints**

- After training, the fine-tuned model will be saved under lora_output/.
- The best model checkpoint is automatically loaded at the end of training.
- To manually load the trained model:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("lora_output/")
model.eval()


**Results & Performance**
- Evaluates validation loss to track performance.
- Uses temperature scaling (temperature=0.01) to control randomness in predictions.
- Extracts sentiment from generated text to ensure classification outputs "Positive" or "Negative".


**Contributions**

Contributions are welcome! Feel free to open issues or submit pull requests.
