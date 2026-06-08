# -*- coding: utf-8 -*-
"""
MSLG-SPA 2026 v8 - VERSIÓN CORREGIDA
=====================================
✅ CORREGIDO: Error "TokenizersBackend has no attribute get_lang_token"
✅ CORREGIDO: Predicciones vacías
✅ CORREGIDO: Formato de submission según especificaciones IberLEF 2026
✅ CORREGIDO: Preservación de convenciones de anotación MSL (-, +, #, dm-)
✅ SIMPLIFICADO: Código más robusto y fácil de usar

Convenciones de anotación MSL preservadas:
  - Guion (-)     : signo único compuesto, ej. YA-VEO, CON-PERMISO
  - Signo más (+) : signos compuestos/contracciones, ej. MAMÁ+PAPÁ
  - Signo # (#)   : préstamo lingüístico / deletreo manual, ej. #OK
  - Prefijo dm-   : deletreo manual de nombre propio, ej. dm-LUIS

Formato de submission:
  - Sin ID  : "Salida del sistema"\n
  - Con ID  : "Identificador"\t"Salida del sistema"\n
  - Saltos de línea en formato Linux (\n)

Soporta:
- HuggingFace (con o sin fine-tuning)
- Ollama local  (http://localhost:11434)
- Ollama remoto / web con API key (cualquier endpoint compatible OpenAI)
"""

import os
import argparse
import warnings
from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================

def load_data_file(filepath: str) -> List[Dict[str, str]]:
    """
    Carga archivos TSV (entrenamiento o test).

    Formatos soportados:
      - 3 columnas (train): ID \\t MSLG \\t SPA
      - 2 columnas (test):  ID \\t MSLG   o   ID \\t SPA

    Las convenciones de anotación MSL (-, +, #, dm-) se preservan tal cual.
    """
    rows = []
    if not os.path.exists(filepath):
        print(f"❌ Error: El archivo {filepath} no existe.")
        return rows

    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\n\r")
        header_tokens = [t.upper().strip() for t in first_line.split("\t")]
        is_header = any(h in header_tokens for h in ["ID", "MSLG", "SPA"])

        # Detectar qué columna es la fuente según la cabecera
        col_map: Dict[str, int] = {}
        if is_header:
            for idx, tok in enumerate(header_tokens):
                col_map[tok] = idx
        else:
            f.seek(0)

        for line in f:
            line = line.rstrip("\n\r")
            if not line or "\t" not in line:
                continue
            parts = line.split("\t")

            if len(parts) >= 3:
                # Archivo de entrenamiento: ID, MSLG, SPA
                rows.append({
                    "id":   parts[0].strip(),
                    "mslg": parts[1].strip(),
                    "spa":  parts[2].strip(),
                })
            elif len(parts) == 2:
                # Archivo de test: ID + contenido (glosa o español)
                # Determinar campo según cabecera detectada
                content = parts[1].strip()
                if "SPA" in col_map and col_map.get("SPA") == 1:
                    rows.append({"id": parts[0].strip(), "mslg": None, "spa": content})
                else:
                    # Por defecto la segunda columna es MSLG (test MSLG2SPA)
                    rows.append({"id": parts[0].strip(), "mslg": content, "spa": None})

    print(f"✓ {len(rows)} registros cargados desde {os.path.basename(filepath)}")
    return rows

def extract_pairs(rows: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    return [(r["mslg"], r["spa"]) for r in rows if r.get("mslg") and r.get("spa")]

# ============================================================================
# 2. PROVEEDORES LLM
# ============================================================================

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, text: str) -> str: ...

class OllamaLLM(BaseLLMProvider):
    """
    Proveedor usando Ollama (local o remoto vía ollama.com).

    Modos de operación:
      1. LOCAL  (por defecto): API nativa en http://localhost:11434, sin autenticación.
      2. CLOUD  (ollama.com):  misma API nativa en https://ollama.com/api con
         autenticación Bearer via API key.
         Crear API key en: https://ollama.com/settings/keys
      3. REMOTO CUSTOM: cualquier servidor Ollama expuesto en otra URL.

    Referencia: https://docs.ollama.com/api/authentication
    """

    def __init__(self, model_name: str = "llama3", direction: str = "mslg2spa",
                 temperature: float = 0.3, max_tokens: int = 256,
                 api_url: Optional[str] = None, api_key: Optional[str] = None):
        try:
            import requests  # noqa: F401
        except ImportError:
            raise ImportError("Instala requests: pip install requests")

        self.model_name = model_name
        self.direction = direction
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")

        # ── Determinar endpoint base ──────────────────────────────────
        if api_url:
            self.base_url = api_url.rstrip("/")
        elif self.api_key:
            # Si hay API key pero no URL, asumir Ollama Cloud
            self.base_url = "https://ollama.com/api"
        else:
            self.base_url = "http://localhost:11434"

        # Normalizar: si el usuario pasó la URL con /api al final, usarla tal cual;
        # si no, agregar /api para las rutas de la API nativa.
        if self.base_url.endswith("/api"):
            self.api_chat_url = f"{self.base_url}/chat"
            self.api_generate_url = f"{self.base_url}/generate"
            self.api_tags_url = f"{self.base_url}/tags"
        else:
            self.api_chat_url = f"{self.base_url}/api/chat"
            self.api_generate_url = f"{self.base_url}/api/generate"
            self.api_tags_url = f"{self.base_url}/api/tags"

        self.is_remote = self.base_url != "http://localhost:11434"

        mode_label = "CLOUD" if "ollama.com" in self.base_url else ("REMOTO" if self.is_remote else "LOCAL")
        print(f"  [Ollama {mode_label}] Modelo: {model_name} | Dirección: {direction}")
        print(f"  [Ollama {mode_label}] Base URL: {self.base_url}")
        if self.api_key:
            masked = ("*" * max(0, len(self.api_key) - 4)) + self.api_key[-4:]
            print(f"  [Ollama {mode_label}] API Key: {masked}")

        self._check_connection()

    # ── Helpers ────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        """Devuelve headers comunes; incluye Authorization si hay API key."""
        h: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _check_connection(self):
        """Verifica que el endpoint responda."""
        import requests
        try:
            resp = requests.get(self.api_tags_url, headers=self._headers(), timeout=10)
            if resp.status_code == 200:
                print(f"  ✓ Ollama conectado")
            elif resp.status_code in (401, 403):
                print(f"  ⚠️  Autenticación fallida ({resp.status_code}). Verifica tu API key.")
            else:
                print(f"  ⚠️  Endpoint respondió con status {resp.status_code}")
        except Exception:
            if self.is_remote:
                print(f"  ⚠️  No se pudo conectar al endpoint: {self.base_url}")
            else:
                print(f"  ⚠️  Ollama no responde. Ejecuta: ollama serve")

    def _build_prompt_text(self, text: str) -> str:
        """Construye el prompt de traducción."""
        if self.direction == "mslg2spa":
            return (
                "Traduce las siguientes glosas de Lengua de Señas Mexicana (LSM) al español natural.\n"
                "IMPORTANTE: Las glosas pueden contener convenciones de anotación LSM que debes respetar "
                "al interpretar el significado, pero NO las copies en la traducción al español:\n"
                "  - Guion (-): signo único compuesto, ej. YA-VEO = 'ya veo'\n"
                "  - Signo más (+): signo compuesto, ej. MAMÁ+PAPÁ = 'padres'\n"
                "  - Prefijo #: préstamo por deletreo manual, ej. #OK = 'OK'\n"
                "  - Prefijo dm-: nombre propio deletreado, ej. dm-LUIS = 'Luis'\n"
                f"Glosas LSM: {text}\n"
                "Traducción al español:"
            )
        else:
            return (
                "Traduce el siguiente texto en español a glosas de Lengua de Señas Mexicana (LSM).\n"
                "IMPORTANTE: Usa las convenciones de anotación LSM cuando corresponda:\n"
                "  - Guion (-): para signos únicos compuestos, ej. YA-VEO\n"
                "  - Signo más (+): para signos compuestos, ej. MAMÁ+PAPÁ\n"
                "  - Prefijo #: para préstamos por deletreo manual, ej. #OK\n"
                "  - Prefijo dm-: para nombres propios deletreados, ej. dm-LUIS\n"
                f"Español: {text}\n"
                "Glosas LSM:"
            )

    # ── Generación ────────────────────────────────────────────────────

    def generate(self, text: str) -> str:
        """
        Genera traducción usando la API nativa de Ollama (local o cloud).

        Usa /api/chat con formato de mensajes, que funciona tanto en
        localhost como en https://ollama.com/api.
        """
        import requests

        prompt = self._build_prompt_text(text)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            response = requests.post(
                self.api_chat_url,
                json=payload,
                headers=self._headers(),
                timeout=120,
            )

            if response.status_code == 200:
                data = response.json()
                # Formato /api/chat: message.content
                content = data.get("message", {}).get("content", "")
                return content.strip()
            else:
                error_msg = ""
                try:
                    error_msg = response.text[:300]
                except Exception:
                    pass
                print(f"  ❌ Error API ({response.status_code}): {error_msg}")
                return ""
        except Exception as e:
            print(f"  ❌ Error Ollama: {e}")
            return ""

class HuggingFaceLLM(BaseLLMProvider):
    """Proveedor usando HuggingFace (CORREGIDO)."""
    
    def __init__(self, model_name: str, use_qlora: bool = False, direction="mslg2spa", 
                 load_finetuned: Optional[str] = None):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print(f"  [HF] Cargando: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.direction = direction
        self.model_name = model_name
        
        # ✅ FIX: Configurar idiomas solo para modelos NLLB
        if "nllb" in model_name.lower():
            try:
                if direction == "mslg2spa":
                    self.tokenizer.src_lang = "eng_Latn"
                    self.tokenizer.tgt_lang = "spa_Latn"
                else:
                    self.tokenizer.src_lang = "spa_Latn"
                    self.tokenizer.tgt_lang = "eng_Latn"
                print(f"  [HF] Idiomas: {self.tokenizer.src_lang} → {self.tokenizer.tgt_lang}")
            except:
                print(f"  [HF] Modelo sin soporte multilingüe explícito")

        # Configurar cuantización
        bnb_cfg = None
        if use_qlora:
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            except:
                print("  ⚠️  QLoRA no disponible, usando modelo completo")
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            quantization_config=bnb_cfg, 
            device_map="auto"
        )
        
        # Cargar checkpoint si existe
        if load_finetuned and os.path.exists(load_finetuned):
            print(f"  [HF] Cargando checkpoint: {load_finetuned}")
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, load_finetuned)
                print(f"  ✓ Checkpoint cargado")
            except Exception as e:
                print(f"  ⚠️  Error cargando checkpoint: {e}")
        
        # Configurar tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if hasattr(self.model, 'config'):
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def apply_lora(self, r=16, alpha=32):
        """Aplica LoRA para fine-tuning."""
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, 
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], 
            lora_dropout=0.05, 
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.model = get_peft_model(self.model, cfg)
        self.model.print_trainable_parameters()

    def generate(self, text: str) -> str:
        """✅ CORREGIDO: Genera sin errores de tokenizer."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Parámetros de generación básicos
        gen_kwargs = {
            "max_new_tokens": 128,
            "num_beams": 4,
            "early_stopping": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # ✅ FIX: Solo configurar forced_bos_token_id para NLLB
        if "nllb" in self.model_name.lower():
            try:
                if hasattr(self.tokenizer, 'lang_code_to_id') and hasattr(self.tokenizer, 'tgt_lang'):
                    tgt_lang = self.tokenizer.tgt_lang
                    if tgt_lang in self.tokenizer.lang_code_to_id:
                        gen_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[tgt_lang]
            except:
                pass  # Continuar sin forced_bos_token_id
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

# ============================================================================
# 3. DATASET Y ENTRENAMIENTO
# ============================================================================

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer, direction="mslg2spa", max_len=128):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.direction = direction
        self.max_len = max_len

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx):
        gloss, spa = self.pairs[idx]
        src = gloss if self.direction == "mslg2spa" else spa
        tgt = spa if self.direction == "mslg2spa" else gloss
        
        model_inputs = self.tokenizer(src, max_length=self.max_len, truncation=True, padding=False)
        labels = self.tokenizer(text_target=tgt, max_length=self.max_len, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

def train_hf_model(llm, train_pairs, val_pairs, direction, output_dir, epochs, batch_size, lr):
    """Entrena modelo HuggingFace."""
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    import transformers
    
    train_ds = TranslationDataset(train_pairs, llm.tokenizer, direction)
    val_ds = TranslationDataset(val_pairs, llm.tokenizer, direction)
    collator = DataCollatorForSeq2Seq(llm.tokenizer, model=llm.model, label_pad_token_id=-100)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch", 
        save_strategy="epoch",
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="none"
    )

    # Compatibilidad con versiones de transformers
    tf_version = tuple(map(int, transformers.__version__.split(".")[:2]))
    trainer_kwargs = {
        "model": llm.model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
    }
    
    if tf_version >= (4, 46):
        trainer_kwargs["processing_class"] = llm.tokenizer
    else:
        trainer_kwargs["tokenizer"] = llm.tokenizer

    trainer = Trainer(**trainer_kwargs)
    
    print("  🏋️ Entrenando...")
    trainer.train()
    
    best_model_path = f"{output_dir}/best_model"
    trainer.save_model(best_model_path)
    print(f"  💾 Modelo guardado en: {best_model_path}")
    
    return best_model_path

# ============================================================================
# 4. GENERACIÓN Y ESCRITURA
# ============================================================================

def generate_predictions(llm: BaseLLMProvider, test_rows: List[Dict], direction: str) -> List[Tuple[str, str]]:
    """
    Genera predicciones para cada instancia del conjunto de test.

    Selecciona el campo fuente según la dirección:
      - mslg2spa: usa el campo 'mslg'
      - spa2mslg: usa el campo 'spa'

    Las convenciones de anotación MSL (-, +, #, dm-) presentes en el texto
    fuente se pasan al modelo sin modificación.
    """
    results = []
    src_field = "mslg" if direction == "mslg2spa" else "spa"

    print(f"  Generando {len(test_rows)} predicciones...")

    for i, row in enumerate(test_rows):
        src_text = row.get(src_field) or ""
        instance_id = row.get("id", str(i))

        if not src_text:
            results.append((instance_id, ""))
            continue

        try:
            pred = llm.generate(src_text)

            # Mostrar progreso
            if i < 3 or (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(test_rows)}] ID={instance_id} | Out: {pred[:60]}...")

        except Exception as e:
            print(f"  ❌ Error ID {instance_id}: {e}")
            pred = ""

        results.append((instance_id, pred))

    non_empty = sum(1 for _, p in results if p)
    print(f"✓ Completado: {non_empty}/{len(results)} predicciones no vacías")

    return results

def write_submission_file(predictions: List[Tuple[str, str]], direction: str,
                          team_name: str, solution_name: str, output_dir: str = ".",
                          include_ids: bool = True) -> str:
    """
    Escribe el archivo de submission según el formato oficial IberLEF 2026.

    Formato sin ID  (include_ids=False):
        "Salida del sistema"\\n

    Formato con ID  (include_ids=True, por defecto):
        "Identificador de instancia"\\t"Salida del sistema"\\n

    Notas:
      - Saltos de línea en formato Linux (\\n).
      - Las comillas dobles (") dentro de la predicción se eliminan para no
        romper el formato; las convenciones MSL (-, +, #, dm-) se preservan.
      - Sin encabezados ni comentarios adicionales.
    """
    suffix = "MSLG2SPA" if direction == "mslg2spa" else "SPA2MSLG"
    filename = f"{team_name}_{solution_name}_{suffix}.txt"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        for instance_id, prediction in predictions:
            # Eliminar comillas dobles internas para no romper el formato;
            # las convenciones MSL (-, +, #, dm-) NO se tocan.
            pred_clean = prediction.strip().replace('"', "")
            id_clean   = str(instance_id).strip().replace('"', "")

            if include_ids:
                f.write(f'"{id_clean}"\t"{pred_clean}"\n')
            else:
                f.write(f'"{pred_clean}"\n')

    print(f"✅ Archivo guardado: {filepath}")
    print(f"   Formato: {'con ID' if include_ids else 'sin ID'} | {len(predictions)} líneas")
    return filepath

# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MSLG-SPA 2026 v8 - CORREGIDO")
    
    parser.add_argument("--train_path", type=str, help="Archivo de entrenamiento")
    parser.add_argument("--test_path", type=str, required=True, help="Archivo de test")
    parser.add_argument("--direction", type=str, choices=["mslg2spa", "spa2mslg"], default="mslg2spa")
    
    parser.add_argument("--llm_backend", type=str, choices=["huggingface", "ollama"], default="ollama")
    parser.add_argument("--llm_model", type=str, default="llama3")
    parser.add_argument("--load_checkpoint", type=str, help="Checkpoint a cargar")
    
    parser.add_argument("--ollama_url", type=str, default=None,
                        help="URL base del servidor Ollama. Ejemplos: "
                             "https://ollama.com/api (cloud), "
                             "http://mi-servidor:11434 (remoto). "
                             "Si se omite y hay --api_key, usa https://ollama.com/api; "
                             "si no hay nada, usa localhost:11434")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key para Ollama Cloud (https://ollama.com/settings/keys). "
                             "También se puede definir con la variable de entorno OLLAMA_API_KEY")
    
    parser.add_argument("--train", action="store_true", help="Entrenar antes de predecir")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=256)
    
    parser.add_argument("--team_name", type=str, default="TeamName")
    parser.add_argument("--solution_name", type=str, default="Run1")
    parser.add_argument("--output_dir", type=str, default="./submissions")
    parser.add_argument("--include_ids", action="store_true", default=True,
                        help="Incluir identificador de instancia en el archivo de submission (por defecto: True)")
    parser.add_argument("--no_ids", dest="include_ids", action="store_false",
                        help="Omitir identificadores; genera solo 'Salida del sistema'\\n")
    
    args = parser.parse_args()

    print("=" * 70)
    print("MSLG-SPA 2026 v8 - VERSIÓN CORREGIDA")
    print("=" * 70)
    print(f"Backend: {args.llm_backend} | Modelo: {args.llm_model}")
    print(f"Dirección: {args.direction}")
    print("=" * 70)

    # Inicializar LLM
    if args.llm_backend == "ollama":
        llm = OllamaLLM(
            model_name=args.llm_model,
            direction=args.direction,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_url=args.ollama_url,
            api_key=args.api_key,
        )
        
    else:  # huggingface
        if args.train:
            if not args.train_path:
                print("❌ --train_path requerido con --train")
                return
            
            print("\n📂 Cargando datos...")
            rows = load_data_file(args.train_path)
            pairs = extract_pairs(rows)
            if not pairs:
                print("❌ Sin datos")
                return
            
            llm = HuggingFaceLLM(args.llm_model, use_qlora=True, direction=args.direction)
            llm.apply_lora()
            
            split = int(len(pairs) * 0.9)
            tr, va = pairs[:split], pairs[split:]
            
            best_path = train_hf_model(llm, tr, va, args.direction, 
                                      f"./results_{args.direction}", 
                                      args.epochs, args.batch_size, args.lr)
            
            llm = HuggingFaceLLM(args.llm_model, use_qlora=True, 
                                direction=args.direction, load_finetuned=best_path)
        else:
            llm = HuggingFaceLLM(args.llm_model, use_qlora=False, 
                                direction=args.direction, load_finetuned=args.load_checkpoint)

    # Generar predicciones
    print("\n📄 Generando predicciones...")
    test_rows = load_data_file(args.test_path)
    if not test_rows:
        print("❌ Sin datos de test")
        return
    
    predictions = generate_predictions(llm, test_rows, args.direction)
    
    write_submission_file(predictions, args.direction,
                         args.team_name, args.solution_name, args.output_dir,
                         include_ids=args.include_ids)
    
    print("\n✅ COMPLETADO")

if __name__ == "__main__":
    main()
