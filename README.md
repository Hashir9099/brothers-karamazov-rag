# Brothers Karamazov Q&A (Russian / English)

A retrieval-augmented Q&A app for Dostoevsky's *Братья Карамазовы*. Ask in Russian or English; answers are generated from retrieved passages of the Russian original, with the sources shown under each answer.

**Live demo:** https://huggingface.co/spaces/hashss55/karamazov-rag

## How it works
- Passages: the book split into ~850-character overlapping chunks
- Retrieval: multilingual-e5-large embeddings + BM25, fused, then reranked with bge-reranker-v2-m3
- Generation: Qwen2.5-7B-Instruct with a LoRA adapter, running on Hugging Face ZeroGPU
- English support: questions are translated to Russian for retrieval; answers and sources can be translated back

## Known limits
- Works best on questions answered by a single passage (about 70-77% on a small held-out test set)
- Plot-level questions, where the answer is spread across the book, can be wrong
- Always check the sources shown under each answer

The model adapter and retrieval data are hosted on the Hugging Face Hub, not in this repo.
