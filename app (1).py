import json, re, contextlib
import numpy as np, torch, gradio as gr, spaces
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ASSETS = "hashss55/karamazov-rag-assets"
BASE = "Qwen/Qwen2.5-7B-Instruct"

root = snapshot_download(
    ASSETS,
    allow_patterns=["units_v2.json", "unit_embeddings_v2.npy", "adapter_*"],
)
units = json.load(open(f"{root}/units_v2.json", encoding="utf-8"))
unit_emb = np.load(f"{root}/unit_embeddings_v2.npy")
tok_ru = lambda t: re.findall(r"\w+", t.lower())
bm25 = BM25Okapi([tok_ru(u) for u in units])

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
# Adapter stays separate (not merged) so we can switch it off for translation
model = PeftModel.from_pretrained(
    model, root, torch_dtype=torch.float16, torch_device="cpu"
).to("cuda").eval()
e5 = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")

SYSTEM = (
    "Ты отвечаешь на вопросы по книге «Братья Карамазовы», используя только "
    "приведённые ниже отрывки текста. Отвечай кратко и точно, как можно короче, "
    "дословно как в тексте. Если в отрывках нет прямого ответа на вопрос, ответь только: «Не уверен — в найденных отрывках ответа нет». Не угадывай.\n\nОтрывки:\n{context}"
)


def run(msgs, max_new_tokens=80, use_adapter=True):
    ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to("cuda")
    ctx = contextlib.nullcontext() if use_adapter else model.disable_adapter()
    with torch.inference_mode(), ctx:
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def translate(text, src, dst, max_new_tokens=300):
    msgs = [
        {"role": "system", "content": (
            f"You are a professional translator. Translate the user's text from {src} "
            f"to {dst}. Translate literally and completely, without paraphrasing or changing any details. Output only the translation, using only {dst} words.")},
        {"role": "user", "content": text},
    ]
    return run(msgs, max_new_tokens, use_adapter=False)


def hybrid_retrieve(q, pool=30, k=60):
    qe = e5.encode([f"query: {q}"], normalize_embeddings=True)
    d = (qe @ unit_emb.T)[0]
    dr = {i: r for r, i in enumerate(d.argsort()[::-1][:pool])}
    b = bm25.get_scores(tok_ru(q))
    br = {i: r for r, i in enumerate(b.argsort()[::-1][:pool])}
    fused = {i: 1 / (k + dr.get(i, pool)) + 1 / (k + br.get(i, pool)) for i in set(dr) | set(br)}
    return sorted(fused, key=fused.get, reverse=True)


@spaces.GPU(duration=60)
def answer(message, history, show_en=True):
    message = message.strip()
    note = ""
    question = message
    if not re.search(r"[А-Яа-яЁё]", message):
        question = translate(message, "English", "Russian", 100)
        note = f"*Question understood as:* {question}\n\n"

    cand = hybrid_retrieve(question)[:15]
    scores = reranker.predict([[question, units[i]] for i in cand])
    top = [cand[i] for i in scores.argsort()[::-1][:5]]
    ctx = "\n\n".join(units[i] for i in top)
    reply = run(
        [{"role": "system", "content": SYSTEM.format(context=ctx)},
         {"role": "user", "content": question}],
        80, use_adapter=True,
    )
    excerpts = [f"{units[i][:300]}…" for i in top]
    sources_ru = "\n\n".join(f"> {e}" for e in excerpts)

    text = f"{note}{reply}"
    if show_en:
        
        reply_en = run(
            [{"role": "system", "content": "Translate the Russian text into English. Reply with ONLY the English translation: no notes, no comments, no explanations. Glossary: гривенник = ten-kopeck coin; копейка = kopeck; рубль = ruble; целковый = silver ruble."},
             {"role": "user", "content": reply}],
            60, use_adapter=False,
        )
        
        text += f"\n\n**English:** {reply_en}"
    text += f"\n\n**Источники / Sources:**\n{sources_ru}"
    if show_en:
        src_en = translate("\n\n".join(excerpts), "Russian", "English", 450)
        text += f"\n\n**Sources (English):**\n{src_en}"
    return text


gr.ChatInterface(
    answer,
    additional_inputs=[gr.Checkbox(value=False, label="Show English translation")],
    title="Братья Карамазовы — Q&A about the text (Russian / English)",
    description="Ask in Russian or English. Answers come from retrieved passages of the Russian original; check the sources, as answers can be wrong.",
).launch()