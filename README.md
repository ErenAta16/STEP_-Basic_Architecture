# STEP — runnable app

This folder holds **only what you need to run** the pipeline locally (code + templates). **Most pipeline logs and code comments are in English**; a few CLI/web hints stay **Turkish** for local users (`run.py --check`, `web_app.py` startup banner). The **Project report** subsection below keeps **three narrative languages** (English, Portuguese, Turkish) on purpose. Quick start is at the **bottom**.

---

## Project report

### English

**Project:** Automated mathematical problem-solving system  
**Environment:** Windows 10, Python 3.11.9, NVIDIA GeForce RTX 3050 Ti (4.3 GB VRAM), CUDA 12.4

---

#### Scope

The first version of the system was developed as a Google Colab notebook. That single file combined PDF handling, OCR, LLM calls, and verification. The first step was to refactor this monolith into a modular local Python project. Each stage became an independent Python module, a central configuration file was added, and a command-line interface was introduced. That refactor made it possible to test, tune, and replace each layer on its own.

---

#### Pipeline architecture

The pipeline has seven layers. **Layer 0 (PDF ingestion)** uses PyMuPDF to extract raw text, metadata, and high-resolution page images from PDFs. As a C-based library, PyMuPDF completes this step in about 0.02 s on average. It ran without failure on all 58 PDFs. Because mathematical symbols (integral signs, superscripts, Greek letters) are often garbled in extracted text, a quality check with seven checkpoints evaluates the text. Image rasterization started at 300 DPI and was raised to 400 DPI to improve OCR performance.

**Layer 1 (Profiling)** interprets PDF content with a custom heuristic regex classifier. It pulls relevant terms from a pool of mathematical keywords, assigns a **primary category** (surface-integral family: scalar/flux/divergence/Stokes, plus general-math types such as integrals, derivative, limit, series, ODE, linear algebra, etc.), optional **secondary** signals, and surface type when applicable (sphere, paraboloid, cylinder, cone, plane, hemisphere, torus). A typical PDF yields several keywords in milliseconds. The derived **domain** is either `surface_integral` or `general_math`, which selects the Layer 5 system prompt in `config.get_system_prompt`.

**Layer 2 (OCR)** uses Meta’s Nougat-OCR. Nougat combines a Swin Transformer and mBART in an encoder–decoder setup and maps pixels directly to LaTeX—a different approach from engines like Tesseract that understand structures such as `\frac{}{}`, `\int`, and `\sqrt{}`. Integrating Nougat locally was one of the hardest technical steps. It depended on Albumentations, which conflicted with the installed PyTorch/transformers stack; that was fixed with a fake-module shim `_setup_albumentations_bypass()`. `generate()` validation changes in newer transformers were also monkey-patched. Even so, Nougat succeeded on only 42 of 58 PDFs (72.4%): 17 PDFs produced empty output or hit a `[repetition]` loop. Average time was ~10.7 s per PDF (~620.9 s total). Those gaps motivated **Layer 3**.

**Layer 3 (VLM)** is one of the most important additions. Meta’s LLaMA 4 Scout 17B-16E-Instruct runs through Groq’s Vision API. It uses a Mixture-of-Experts layout: ~17B active parameters, 16 experts, and a 128k-token context window; on DocVQA it reaches 94.4 ANLS. Each page is sent base64 to the API and the model returns LaTeX read from the image. VLM succeeded on **all** 58 PDFs (100%) and alone recovered the 16 PDFs where Nougat failed. Average time was ~2.4 s per PDF—about **4.5× faster** than local GPU Nougat despite running via a cloud API. Total VLM cost for 58 PDFs was on the order of **~$0.01**.

Prompt design matters: the VLM is instructed to **read only, not solve**—an eight-rule system prompt blocks solution steps, forbids `\boxed{}` answers, and stresses reading every symbol carefully. **Dual-pass** VLM runs (when the first pass does not already hit the quality rubric maximum) pick the better output by score and length. `clean_output()` strips lines that look like reasoning (“we get”, “substituting”, “therefore”) while keeping the problem statement.

**Layer 4 (Synthesis)** merges prior outputs into a single LLM prompt. It chooses among four strategies: triple source (Nougat + VLM + raw text, 42 PDFs), VLM-primary (VLM + raw, 16 PDFs), Nougat-primary, or raw fallback. Layer-1 classification adds targeted hints (e.g. for flux problems, notes on **r_u × r_v** and **|r_u × r_v|** in the surface element). Phase 6 adds general-math support: surface problems use an eight-step procedure; other math uses a more flexible prompt. Average prompt length is ~3200 characters; synthesis time is negligible.

**Layer 5 (LLM solver)** is the mathematical reasoner. Four LLMs were tried. **Claude Sonnet 4** was dropped due to latency and unreliable quality. **Groq LLaMA 3.3 70B** gave ~96.5% accuracy on the first try and 100% with retries, with ~2.8 s average latency and free API access, but inconsistent errors on some problems (e.g. si28, si35) limited trust as the sole solver. **GPT-4o** reached ~88% in limited tests but full evaluation stopped for credit limits. **Gemini 2.5 Flash** (with “thinking”) achieved **100%** on the tested set but is slower (~15–35 s). The current setup uses **Gemini as primary** and **Groq as fallback**.

**Layer 6 (answer display)** parses the LLM reply to extract one **final line** for the UI (`\boxed{}`, `FINAL_ANSWER:`, tail heuristics). The class name still says “SymPyVerifier” for historical reasons; **reference-answer SymPy checks are not in this layer**. **Consensus / numeric agreement** between repeated LLM attempts uses `latex_parser.parse_latex_to_value` inside `run.STEPSolver._solve_with_consensus`. The LaTeX normalizer behind that path evolved across versions (fractions, roots, powers, implicit multiplication).

---

#### Issues encountered and fixes

**Nougat:** (1) Albumentations vs PyTorch → fake-module injection. (2) `generate()` validation on new transformers → monkey-patch. (3) `[repetition]` loops on 17 PDFs → partial output salvage plus VLM fallback.

**VLM:** The worst bug was `clean_output()` **deleting the problem statement**. Revising the kill-phrase list increased retained length ~570% and restored correct LLM answers.

**Answer extraction:** Gemini sometimes omits `\boxed{}` or answers in free-form LaTeX—addressed with a seven-stage extraction strategy and a follow-up API call.

---

#### Technology choices

**PyMuPDF** — speed and reliability (much faster than typical pure-Python PDF libs). **Nougat** — academic PDFs and direct LaTeX, but 72.4% alone was insufficient. **LLaMA 4 Scout 17B** — DocVQA score, MoE efficiency, Groq Vision access. **Gemini 2.5 Flash** — strong math reasoning (“thinking”). **Groq LLaMA 3.3 70B** — very fast fallback with free tier. **SymPy** — full Python CAS for verification. **Flask** — lightweight web UI with SSE. **MathJax 3** — high-quality LaTeX in the browser.

---

### Português (Portugal / Brasil)

**Projeto:** Sistema automático de resolução de problemas de matemática  
**Ambiente:** Windows 10, Python 3.11.9, NVIDIA GeForce RTX 3050 Ti (4.3 GB VRAM), CUDA 12.4

---

#### Âmbito

A primeira versão foi um notebook Google Colab monolítico com PDF, OCR, LLM e verificação no mesmo ficheiro. O primeiro passo foi modularizar num projeto Python local: cada etapa tornou-se um módulo, criou-se configuração central e uma CLI, permitindo testar e otimizar cada camada separadamente.

---

#### Arquitetura do pipeline

O pipeline tem **sete camadas**. **Camada 0** (PyMuPDF) extrai texto, metadados e imagens de página (~0,02 s em média; 58/58 PDFs sem falha). O texto passa por **sete verificações de qualidade**; a resolução das imagens subiu de 300 para **400 DPI**.

**Camada 1** usa um classificador heurístico com regex: 17 palavras-chave, quatro categorias principais (integral de superfície escalar, fluxo, teorema da divergência, Stokes) e tipo de superfície. Fase 6 adicionou dez categorias de matemática geral; o domínio passa a `surface_integral` ou `general_math`.

**Camada 2 (Nougat-OCR):** arquitetura Swin + mBART, LaTeX a partir de pixeis. Integração difícil: conflito com Albumentations → `_setup_albumentations_bypass()`; validação do `generate()` → monkey-patch. Sucesso em **42/58 PDFs (72,4%)**; 17 com saída vazia ou `[repetition]`; ~10,7 s/PDF. **Camada 3 (VLM)** — LLaMA 4 Scout 17B via **Groq Vision**, MoE, contexto longo, **94,4 ANLS** no DocVQA; **100%** nos 58 PDFs; ~2,4 s/PDF (~4,5× mais rápido que Nougat local neste hardware); custo total ~**0,01 USD**. Prompt “só ler, não resolver”; **dupla passagem** por página; `clean_output()` remove passos de solução.

**Camada 4** funde fontes (tripla, VLM+prioridade, Nougat+prioridade, só texto) e *hints* por classe de problema. **Camada 5:** testados Claude, Groq Llama 3.3 70B, GPT-4o, Gemini 2.5 Flash; **Gemini primário**, **Groq fallback**. **Camada 6 (SymPy):** numérico, simbólico e *string search*; parser LaTeX evoluiu até **58/58** parses.

---

#### Problemas e soluções

Nougat: módulo falso + *patch* do `generate` + VLM quando há `[repetition]`. VLM: `clean_output()` apagava o enunciado — lista de frases corrigida. Extração: Gemini sem `\boxed{}` consistente — extração em sete níveis + *follow-up*.

---

#### Escolhas tecnológicas

PyMuPDF pela velocidade; Nougat para PDFs académicos mas insuficiente sozinho; LLaMA 4 Scout na Groq para visão; Gemini para raciocínio matemático; Groq como *fallback* rápido; SymPy para CAS; Flask + SSE na UI; MathJax 3 no browser.

---

### Türkçe

**Proje:** Matematik problemi otomatik çözüm sistemi  
**Ortam:** Windows 10, Python 3.11.9, NVIDIA GeForce RTX 3050 Ti (4,3 GB VRAM), CUDA 12.4

---

#### Kapsam

Sistemin ilk versiyonu bir Google Colab notu olarak geliştirilmiştir; PDF işleme, OCR, LLM ve doğrulama tek dosyadaydı. İlk adım olarak bu monolitik yapı modüler bir yerel Python projesine dönüştürülmüştür. Her katman bağımsız modül, merkezi yapılandırma ve komut satırı arayüzü ile katmanların ayrı ayrı test ve iyileştirilmesi mümkün hale gelmiştir.

---

#### Pipeline mimarisi

**Katman 0 (PDF ingestion):** PyMuPDF ile ham metin, metadata ve yüksek çözünürlüklü görseller; ortalama ~0,02 s; 58 PDF’de hatasız çalışma. Metin 7 kontrol noktasından geçer; görüntü DPI başlangıçta 300, OCR için **400**’e çıkarılmıştır.

**Katman 1 (Profiling):** Heuristik regex sınıflandırıcı; 17 anahtar kelime havuzu, dört ana kategori (skaler yüzey integrali, akı integrali, diverjans teoremi, Stokes) ve yüzey tipi. Faz 6’da 10 ek genel matematik kategorisi; çözüm stratejisi `surface_integral` veya `general_math`.

**Katman 2 (Nougat-OCR):** Swin Transformer + mBART; pikselden LaTeX. Albumentations çakışması → `_setup_albumentations_bypass()`; transformers `generate()` doğrulaması → monkey-patch. **42/58 PDF (%72,4)** başarı; 17 PDF’de boş çıktı veya `[repetition]`; ortalama ~10,7 s/PDF. **Katman 3 (VLM):** Groq Vision üzerinden **LLaMA 4 Scout 17B**, MoE, DocVQA’da **94,4 ANLS**; **58/58 PDF (%100)**; Nougat’ın düştüğü 16 PDF’yi kurtarır; ortalama ~2,4 s/PDF (Nougat’a göre ~4,5× daha hızlı); toplam maliyet ~**0,01 USD**. Prompt: “sadece oku, çözme”; sayfa başına **çift geçiş**; `clean_output()` çözüm cümlelerini budar.

**Katman 4 (Synthesis):** Nougat + VLM + ham metin stratejileri; akı gibi problemlerde otomatik ipuçları; yüzey integrali için 8 adımlı prosedür, genel matematik için esnek prompt; ortalama ~3200 karakter.

**Katman 5 (LLM):** Claude (gecikme/kalite) → Groq Llama 3.3 70B (hızlı, retry ile yüksek doğruluk ama bazı problemlerde tutarsız) → GPT-4o (sınırlı test) → **Gemini 2.5 Flash** (%100 test seti, daha yavaş). **Birincil: Gemini, yedek: Groq.**

**Katman 6 (Verification):** SymPy ile sayısal (48 problem, 0,01 tolerans), sembolik (5 parametrik) ve string yedek. LaTeX parser v1’den v4’e: **58/58** parse.

---

#### Hatalar ve çözümler

Nougat: sahte modül + `generate` yaması + VLM yedek. VLM: `clean_output()` en başta problem metnini siliyordu — kill-phrase revizyonu; çıktı uzunluğu ~%570 arttı. Cevap çıkarma: Gemini `\boxed{}` tutarsızlığı — 7 kademeli strateji + takip çağrısı.

---

#### Teknoloji seçimleri

PyMuPDF (hız); Nougat (akademik LaTeX, tek başına yetersiz); LLaMA 4 Scout + Groq (VLM); Gemini (düşünme / matematik); Groq 70B (hızlı fallback); SymPy (CAS); Flask + SSE (web); MathJax 3 (tarayıcı).

---

## Run this project locally

### Setup

```bash
cd Step_Project
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
# or: cp .env.example .env
```

Put at least `GEMINI_API_KEY` and `GROQ_API_KEY` in `.env`.

### Run

**Web UI (upload a PDF):**

```bash
python web_app.py
```

Open `http://127.0.0.1:5000`.

**CLI:**

```bash
python run.py path/to/problem.pdf --no-nougat
python run.py --check
python run.py Surface_Integration/
```

Docs, reports, and extra tooling: **`Step_Project_Disari/`** in the repo root.

Editable narrative source (same content you maintain): **`../STEP_Pipeline_Rapor_Paragraf.md`** (one level above this folder).
