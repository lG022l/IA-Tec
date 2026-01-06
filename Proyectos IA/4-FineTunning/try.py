# probar_tutor_fix.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import warnings
warnings.filterwarnings("ignore")

print("="*70)
print("TUTOR DE ALGORITMOS")
print("="*70)

# Verificar modelo
if not os.path.exists("./tutor-algoritmos-rtx3050"):
    print("No hay modelo")
    exit()

# Determinar dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsando: {device}")

print("Cargando modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
)

print("Cargando adaptador LoRA...")
model = PeftModel.from_pretrained(base_model, "./tutor-algoritmos-rtx3050")

model = model.to(device)
model.eval()

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./tutor-algoritmos-rtx3050", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Todo listo")
print("="*70)

def preguntar(pregunta):
    """Pregunta al tutor"""
    prompt = f"<|user|>\n{pregunta}\n<|assistant|>\n"
    
    # Tokenizar
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # CRÍTICO: Mover TODOS los inputs al mismo dispositivo que el modelo
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar
    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraer solo respuesta del asistente
    if "<|assistant|>" in respuesta_completa:
        respuesta = respuesta_completa.split("<|assistant|>")[-1]
        if "<|end|>" in respuesta:
            respuesta = respuesta.split("<|end|>")[0]
        respuesta = respuesta.strip()
    else:
        respuesta = respuesta_completa.strip()
    
    return respuesta

print("PRUEBAS AUTOMÁTICAS")

preguntas_prueba = [
    "¿Para qué sirve un ciclo for?"]

for i, pregunta in enumerate(preguntas_prueba, 1):
    print(f"Pregunta {i}/{len(preguntas_prueba)}: {pregunta}")
    print("-"*70)
    
    try:
        respuesta = preguntar(pregunta)
        print(respuesta)
        print("="*70)
    except Exception as e:
        print(f"Error: {e}")
        print("="*70)


print("\n\nPREGUNTA TU:")
print("Escribe 'salir' para terminar\n")

while True:
    try:
        pregunta_usuario = input("\nTu pregunta: ").strip()
        
        if not pregunta_usuario:
            continue
            
        if pregunta_usuario.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\nFin")
            break
        
        print("Tutor:")
        print("-"*70)
        respuesta = preguntar(pregunta_usuario)
        print(respuesta)
        print("-"*70)
        
    except KeyboardInterrupt:
        print("Fin")
        break
    except Exception as e:
        print(f"Error al procesar pregunta: {e}")

print("Sesión terminada")