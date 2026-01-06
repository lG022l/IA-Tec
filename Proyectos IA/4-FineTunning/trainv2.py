import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

MAX_LENGTH = 1024 

print("="*70)
print("ENTRENAMIENTO DE TUTOR DE ALGORITMOS")
print("="*70)

if not torch.cuda.is_available():
    print("ERROR: No se detectó GPU")
    exit()

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f" VRAM Total: {vram_gb:.2f} GB\n")

print("Cargando dataset...")
if not os.path.exists("tutor_dataset.jsonl"):
    print("ERROR: Falta tutor_dataset.jsonl")
    exit()

dataset = load_dataset('json', data_files='tutor_dataset.jsonl')

def format_chat_template(example):
    messages = example['messages']
    formatted_text = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system': formatted_text += f"<|system|>\n{content}\n"
        elif role == 'user': formatted_text += f"<|user|>\n{content}\n"
        elif role == 'assistant': formatted_text += f"<|assistant|>\n{content}\n<|end|>\n"
    return {"text": formatted_text}

dataset = dataset.map(format_chat_template, remove_columns=dataset["train"].column_names)

print("\nCargando modelo...")
model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, # float16 es nativo en Ampere (serie 30)
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Importante para evitar errores de padding

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# DESACTIVAR CACHÉ (Obligatorio para Gradient Checkpointing)
model.config.use_cache = False 
model.config.pretraining_tp = 1

# 3. CONFIGURAR LORA (Optimizado para VRAM)
print("Configurando LoRA Ligero...")

# Habilitar Gradient Checkpointing (EL SALVAVIDAS DE VRAM)
model.gradient_checkpointing_enable() 
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # AÑADIMOS DE NUEVO: k_proj y dense.
    target_modules=["q_proj", "k_proj", "v_proj", "dense"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros entrenables: {trainable_params:,}")

# 4. TOKENIZAR
print("Tokenizando...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 5. CONFIGURAR ENTRENAMIENTO (Estrategia Low VRAM)
print("Configurando Trainer...")

training_args = TrainingArguments(
    output_dir="./resultados_rtx3050_8gb",
    num_train_epochs=3, # Puedes subirlo a 5-10 si ves que aprende bien
    
    # AJUSTE PARA 8GB:
    per_device_train_batch_size=2,   # Intentamos procesar 2 a la vez
    gradient_accumulation_steps=4,   
    
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    
    # MANTENER ESTO ACTIVO (Vital para que quepa en 8GB con contexto de 1024)
    gradient_checkpointing=True, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# 6. ENTRENAR
print("----------ENTRENANDO---------------")
print("="*70)

try:
    trainer.train()
    
    print("--------------COMPLETADO--------------")
    output_dir = "./tutor-algoritmos-rtx3050"
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modelo guardado en: {output_dir}")
    
except Exception as e:
    print(f"ERROR CRÍTICO: {str(e)}")