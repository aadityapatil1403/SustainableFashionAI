import torch
import json
from PIL import Image
from huggingface_hub import login as hf_login
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import openai  # pip install openai

# ——— 1) Authentication ————————————————————————————————————————————
def login_to_hf(hf_token: str):
    hf_login(token=hf_token)
    print("Logged into Hugging Face.")

def login_to_openai(openai_key: str):
    openai.api_key = openai_key
    print("OpenAI key set.")

# ——— 2) Robust JSON parser ————————————————————————————————————————
def parse_model_json(raw: str) -> dict:
    """
    Extracts the JSON object that appears after the 'ASSISTANT:' marker.
    Un-escapes any '\\_' so that keys like 'product_name' parse correctly.
    """
    marker = "ASSISTANT:"
    idx    = raw.find(marker)
    start  = raw.find("{", idx if idx != -1 else 0)
    end    = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not locate JSON in the model output:\n{raw!r}")
    json_str = raw[start : end+1].replace("\\_", "_")
    return json.loads(json_str)

# ——— 3) Phase 1: Extract garment JSON (no prompt echo) —————————————————————
def extract_garment_json(image_path: str, model_id: str, hf_token: str) -> dict:
    processor = LlavaProcessor.from_pretrained(
        model_id, trust_remote_code=True, use_auth_token=hf_token
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_auth_token=hf_token,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    prompt = """
SYSTEM: You are a garment-tech vision assistant. Output JSON
**only**—nothing else—using keys:
  product_name, category, color of hoodie, components like hood,
sleeve, pocket, cuff.
Do NOT repeat this prompt. Do not print design elements
USER: <image>
ASSISTANT:
"""
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=processor.tokenizer.eos_token_id
    )
    raw = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # isolate JSON payload
    json_payload = raw.split("ASSISTANT:")[-1].strip()
    return parse_model_json("ASSISTANT:" + json_payload)

# ——— 4) Phase 2: Tech-Pack Narration ——————————————————————————————————
def generate_techpack_paragraph(data: dict, openai_key: str) -> str:
    openai.api_key = openai_key
    system_msg = """
You are a product-tech writer. Given the garment JSON and lookup tables:
  – Brown cotton: 3.5 m @ $17.50 total
  – Blue screen-print: $5.00
  – Pantone 18-1142 TCX Bison
  – Pantone 15-3920 TCX Sky Blue
Produce ONE concise paragraph covering:
  1. Stylized product name & category
  2. Fabric composition, yardage & cost
  3. Pantone color references of the hoodie
  4. Key design elements & their locations
  5. List of components like hood , cuff , pocket, sleeves
Do NOT echo your instructions.
"""
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": json.dumps(data)}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# ——— 5) Main workflow —————————————————————————————————————————————
if __name__ == "__main__":
    HF_TOKEN       = "HF token”
    OPENAI_KEY     = "OpenAI API key”
    IMAGE_PATH     = “imagepath”
    MODEL_ID       = "llava-hf/llava-1.5-7b-hf"
    WORDS_PER_LINE = 15  # exactly 15 words per line

    # Auth
    login_to_hf(HF_TOKEN)
    login_to_openai(OPENAI_KEY)

    # Phase 1
    print("\n[…] Phase 1: extracting garment JSON …")
    garment_data = extract_garment_json(IMAGE_PATH, MODEL_ID, HF_TOKEN)
    print("Parsed JSON:\n", json.dumps(garment_data, indent=2))

    # Phase 2
    print("\n[…] Phase 2: generating tech-pack paragraph …")
    paragraph = generate_techpack_paragraph(garment_data, OPENAI_KEY)

    # Chunk into 15 words/line for screenshot
    words = paragraph.split()
    print("\nTech-Pack Summary :\n")
    for i in range(0, len(words), WORDS_PER_LINE):
        print(" ".join(words[i : i + WORDS_PER_LINE]))