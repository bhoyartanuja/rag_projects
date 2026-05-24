# LLM Internals — Tokens, Temperature, RLHF 🧠

## How GPT Actually Works

### One line:
GPT is a function that takes a sequence of tokens and predicts the next most likely token.

### Training:
```
Take internet text
Cover last word of every sentence
Ask model to predict covered word
If wrong → adjust weights
Repeat billions of times
```
No magic. Learned everything by being wrong billions of times and correcting itself.

---

## Transformer Architecture

### Why transformers:
Before 2017 models read left to right — slow, forgot early context.
Transformers read entire sequence at once and use attention to decide what to focus on.

### Self Attention:
"The cat sat on the mat because it was tired"
When processing "it" → attention assigns weights to every other word:
```
cat    0.72  ← highest
mat    0.12
sat    0.08
```
Every token attends to every other token simultaneously.

### Transformer block:
```
Input tokens
     ↓
Embedding layer    — convert tokens to vectors
     ↓
Self Attention     — every token looks at every other token
     ↓
Feed Forward       — process each token independently
     ↓
Repeat N times     — GPT4 has 96 layers
     ↓
Output layer       — probability distribution over all tokens
```
Early layers learn syntax. Later layers learn concepts and reasoning.

---

## Tokens

### What is a token:
Not a word. Not a character. Something in between.
```
"Hello"          → 1 token
"Tokenization"   → 2 tokens  ["Token", "ization"]
" the"           → 1 token   (space included)
"🔥"             → 2 tokens
```

### Rule of thumb:
```
1 token ≈ 4 characters ≈ 0.75 words
1000 tokens ≈ 750 words ≈ 1.5 pages
```

### Why tokens matter in production:
- **Cost** — charged per token (input + output)
- **Context window** — max tokens model can see at once
- **Chunking** — chunk size should respect token limits

### Token budget in RAG:
```
System prompt        ~200 tokens
Retrieved chunks     ~1500 tokens  (3 chunks × 500 chars)
User question        ~50 tokens
Answer               ~300 tokens
─────────────────────────────────
Total per query      ~2050 tokens
```

---

## Tokenization — Model Specific

### How it works — BPE (Byte Pair Encoding):
```
Start: "tokenization" → [t][o][k][e][n][i][z][a][t][i][o][n]
Find most frequent pairs → merge
"t"+"o" → "to"
"to"+"k" → "tok"
"tok"+"en" → "token"
Repeat 50,000 times → vocabulary of 50,000 subword units
```

### Tokenization varies across models:
Each model trains its own tokenizer on different data with different vocabulary size.

```
"ChatGPT" → GPT-4:   ["Chat", "G", "PT"]    3 tokens
"ChatGPT" → Claude:  ["Chat", "GPT"]         2 tokens
```

Same word — different token count — different cost.

### Real inconsistencies:
```
Numbers:
"100"    → 1 token
"99999"  → 3 tokens ["999", "99"]
→ Why LLMs are bad at arithmetic

Non-English:
"hello"      → 1 token
"こんにちは"  → 5 tokens  (costs more, uses more context)

Spaces matter:
"token"   → 1 token
" token"  → 1 token  (different token)
"  token" → 2 tokens
```

### Token based chunking (more precise than character based):
```python
from tiktoken import encoding_for_model
enc = encoding_for_model("gpt-4")
tokens = enc.encode(text)
chunks = [tokens[i:i+500] for i in range(0, len(tokens), 450)]
```

### Tools:
- OpenAI: `pip install tiktoken`
- Visual: platform.openai.com/tokenizer
- Gemini: `model.count_tokens("text")`

### Special tokens:
```
<|endoftext|>   — end of document
<|im_start|>    — start of message
<|im_end|>      — end of message
```
How model understands conversation structure — where system prompt ends, user message begins.

---

## Temperature

### What it controls:
How random or deterministic the output is.

### How LLM picks next token:
Model produces probability distribution over entire vocabulary:
```
"The answer is" → next token:
"42"       0.40
"unknown"  0.25
"complex"  0.20
"yes"      0.10
"banana"   0.001
```

### Temperature effect:
```
Temperature = 0.0  → always pick highest ("42") — deterministic
Temperature = 0.5  → slightly random, still sensible
Temperature = 1.0  → sample proportionally
Temperature = 2.0  → very random, creative but often nonsense
```

### Practical settings:
| Use Case | Temperature |
|---|---|
| RAG / factual Q&A | 0.0 – 0.3 |
| Code generation | 0.0 – 0.2 |
| Summarization | 0.3 – 0.5 |
| Creative writing | 0.7 – 1.0 |
| Brainstorming | 0.8 – 1.2 |

**RAG systems → temperature near 0. Factual, consistent, auditable.**

### Top-K:
Only consider K most likely tokens at each step.
```
Top-K = 5 → only pick from top 5 tokens
```

### Top-P (nucleus sampling) — preferred in production:
Pick from tokens whose cumulative probability exceeds P.
```
Top-P = 0.9 → pick tokens until probabilities sum to 90%
```
More adaptive than Top-K — sometimes 3 tokens, sometimes 20.

---

## Context Window

### What it is:
Maximum tokens model can see at once — input + output combined.

```
GPT-4        — 128,000 tokens
Gemini 2.5   — 1,000,000 tokens
Claude        — 200,000 tokens
```

### Lost in the middle problem:
LLMs remember beginning and end of context well but forget the middle.
→ Why top-K = 3 in RAG, not 20. Quality over quantity.

### Context window formula:
```
Context = system prompt + chat history + retrieved chunks + question + answer
```

---

## RLHF — Reinforcement Learning from Human Feedback

### The problem:
Raw GPT trained on internet text → helpful, harmful, toxic, all mixed.
RLHF transforms raw text predictor into helpful, harmless assistant.

---

### Stage 1 — Supervised Fine Tuning (SFT):
Show model human written ideal conversations:
```
Human: Explain quantum computing
Assistant: [human written ideal response]
```
Model learns format and style. Limited by how many examples humans can write.

---

### Stage 2 — Train a Reward Model:
Instead of writing responses — rank them. Much faster and scalable.

```
Same prompt → 3 responses → human ranks A > B > C
```

Train separate neural network (Reward Model) to predict human preference score:
```
Response A → 0.92
Response B → 0.71
Response C → 0.08
```
Now you have automated human preference predictor.

---

### Stage 3 — PPO (Proximal Policy Optimization):
Use Reward Model as feedback signal:
```
Prompt → LLM generates → Reward Model scores → Update LLM weights
```
LLM learns to generate responses humans prefer.

**PPO constraint:** Small careful updates only — prevents reward hacking (gaming the score with nonsense).

**KL divergence penalty:**
```
Final reward = Reward Model score - KL penalty
```
KL penalty = how far model drifted from SFT baseline.
Keeps model grounded in sensible language.

---

### Full pipeline:
```
Base GPT (pretrained on internet)
     ↓
Stage 1: SFT on human conversations
     ↓
Stage 2: Human rankings → Reward Model
     ↓
Stage 3: PPO optimization
     ↓
ChatGPT / Claude / Gemini
```

### What RLHF changed:
| Before | After |
|---|---|
| Completes text | Follows instructions |
| No safety | Refuses harmful requests |
| No personality | Consistent helpful tone |
| Raw prediction | Aligned assistant |

---

## Constitutional AI — What Anthropic Does (Claude)

Instead of human rankers → set of principles (constitution):
```
"Be helpful"
"Be harmless"
"Be honest"
```
Model critiques its own responses against these principles and revises.
Then RLHF on top.

Scales better — model does much of its own feedback.

---

## Interview Answers

**On how GPT works:**
*"GPT is a transformer that predicts the next token using self-attention across the entire context. Each layer learns increasingly abstract representations — early layers handle syntax, later layers handle reasoning and concepts."*

**On tokens:**
*"Tokenization is model specific — each model trains its own BPE tokenizer on its corpus. Same text produces different token counts across GPT, Claude, and Gemini. In production RAG I chunk by tokens using the target model's tokenizer for precise context window management."*

**On temperature:**
*"Temperature controls sampling randomness from the output probability distribution. In production RAG I set it near zero for factual consistency. Top-P nucleus sampling is generally preferred over Top-K as it's more adaptive to the probability distribution shape."*

**On RLHF:**
*"RLHF has three stages — SFT on ideal conversations, training a reward model on human preference rankings, then PPO to optimize toward that reward signal. KL penalty prevents reward hacking. Anthropic extends this with Constitutional AI where the model self-critiques against a set of principles."*

---

## Next Topics
- [ ] Fine tuning vs RAG — when to use which
- [ ] How embeddings are trained (contrastive learning)
- [ ] Prompt engineering patterns
- [ ] Hallucination — causes and mitigation
