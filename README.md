# STEP — runnable app

This directory is the runnable slice of the project (application code and templates). Pipeline logs and comments are mostly **English**; a few CLI and web strings stay **Turkish** for local use (`run.py --check`, `web_app.py` banner). The **Project report** below is intentionally repeated in **English, Portuguese, and Turkish**. Quick start is at the end.

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

#### Runtime and reliability updates

**Layers 0 and 2 — raster reuse:** When `dpi` is not passed in, Layer 0 rasterizes page images at **`NOUGAT_DPI`** from config so output matches what Nougat expects. Each folder of `page_*.png` files carries a small **`.step_raster_meta`** file (DPI on the first line, SHA-256 of the PDF on the second). Layer 2 only reuses those PNGs when that sidecar, the on-disk page names, and the current PDF all agree; otherwise it renders again and refreshes the sidecar. **`page_*.png` paths are sorted numerically** so order stays correct past page nine.

**Layer 3 — VLM concurrency:** Per-page vision calls can run through a **bounded worker pool**. Set **`STEP_VLM_PAGE_WORKERS`** if you need fewer parallel requests (rate limits) or fully serial calls (`1`). Gemini remains available **via the API** in addition to the Groq vision path described above.

**Web UI:** Several solves at once are limited with a **semaphore**; the cap is **`STEP_WEB_MAX_CONCURRENT_SOLVES`** (defaults to 2). Uploading the same sanitized filename again while the first job is still running returns **HTTP 409**. Lower the cap when Nougat-heavy GPU work risks contention.

**Layer 6 — answer cleanup:** Extracting a short final string no longer splits on `=` blindly. The code tracks **brace depth** and only uses the last top-level equals, so groups such as `\text{...}` and similar LaTeX do not get chopped apart.

---

#### Technology choices

**PyMuPDF** — fast, reliable PDF text and raster output. **Nougat** — pixels to LaTeX for academic layouts; on its own it missed too many of our PDFs. **LLaMA 4 Scout 17B** — vision model on Groq, good DocVQA-style scores. **Gemini 2.5 Flash** — primary text solver (“thinking” mode). **Groq LLaMA 3.3 70B** — quick fallback when Gemini is busy or errors. **SymPy** (via `latex_parser`) — turns short LaTeX fragments into numbers so repeated solver attempts can agree. **Flask** — small web UI with SSE. **MathJax 3** — render LaTeX in the browser.

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

**Camada 1** usa um classificador heurístico com regex: palavras-chave, categoria **primária** (família de integrais de superfície ou tipos de matemática geral), sinais **secundários** opcionais e tipo de superfície quando aplicável. O **domínio** derivado (`surface_integral` ou `general_math`) escolhe o *system prompt* da Camada 5.

**Camada 2 (Nougat-OCR):** arquitetura Swin + mBART, LaTeX a partir de pixeis. Integração difícil: conflito com Albumentations → `_setup_albumentations_bypass()`; validação do `generate()` → monkey-patch. Sucesso em **42/58 PDFs (72,4%)**; 17 com saída vazia ou `[repetition]`; ~10,7 s/PDF. **Camada 3 (VLM)** — LLaMA 4 Scout 17B via **Groq Vision**, MoE, contexto longo, **94,4 ANLS** no DocVQA; **100%** nos 58 PDFs; ~2,4 s/PDF (~4,5× mais rápido que Nougat local neste hardware); custo total ~**0,01 USD**. Prompt “só ler, não resolver”; **duas passagens** quando a primeira não atinge o máximo da rubrica; `clean_output()` remove passos de solução.

**Camada 4** funde fontes (tripla, VLM+prioridade, Nougat+prioridade, só texto) e *hints* por classe de problema. **Camada 5:** testados Claude, Groq Llama 3.3 70B, GPT-4o, Gemini 2.5 Flash; **Gemini primário**, **Groq fallback**. **Camada 6** extrai uma linha final para a UI (`\boxed{}`, `FINAL_ANSWER:`, heurísticas no fim do texto); o nome da classe ainda remete a SymPy por histórico. A **comparação numérica entre tentativas** do solver usa `latex_parser` em `run.STEPSolver._solve_with_consensus`.

---

#### Problemas e soluções

Nougat: módulo falso + *patch* do `generate` + VLM quando há `[repetition]`. VLM: `clean_output()` apagava o enunciado — lista de frases corrigida. Extração: Gemini sem `\boxed{}` consistente — extração em sete níveis + *follow-up*.

---

#### Atualizações de execução e fiabilidade

**Camadas 0 e 2 — reutilização de raster:** Sem argumento `dpi`, a camada 0 rasteriza à **`NOUGAT_DPI`** definida em configuração, alinhando com o Nougat. Cada pasta de `page_*.png` inclui **`.step_raster_meta`** (linha 1: DPI; linha 2: SHA-256 do PDF). A camada 2 só reutiliza esses PNGs quando o *sidecar*, os nomes das páginas e o ficheiro PDF atual coincidem; caso contrário volta a rasterizar e atualiza o *sidecar*. **`page_*.png`** é ordenado **numericamente** para manter a ordem correta após a página 9.

**Camada 3 — paralelismo VLM:** Chamadas por página podem usar um **conjunto limitado de *workers***. **`STEP_VLM_PAGE_WORKERS`** controla o paralelismo (use `1` para chamadas totalmente sérias ou para limitar pedidos por minuto).

**Interface web:** Várias resoluções em simultâneo ficam limitadas por um **semáforo** (**`STEP_WEB_MAX_CONCURRENT_SOLVES`**, por omissão 2). Um segundo *upload* com o **mesmo nome sanitizado** enquanto a primeira tarefa corre devolve **HTTP 409**. Reduza o limite quando o Nougat e a GPU estiverem sob pressão.

**Camada 6 — limpeza da resposta:** A extração deixa de partir a cadeia em `=` de forma ingénua. O código respeita a **profundidade de chavetas** e só considera o último `=` ao nível superior, evitando cortar texto dentro de `\text{...}` e construções semelhantes.

---

#### Escolhas tecnológicas

PyMuPDF pela velocidade; Nougat para LaTeX a partir de imagens; LLaMA 4 Scout na Groq para visão; Gemini como solver principal; Groq como *fallback*; SymPy (via `latex_parser`) para alinhar respostas numéricas entre tentativas; Flask + SSE; MathJax 3 no browser.

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

**Katman 1 (Profiling):** Heuristik regex; anahtar kelimeler, **birincil** kategori (yüzey integrali ailesi veya genel matematik türleri), isteğe bağlı **ikincil** sinyaller ve uygunsa yüzey tipi. Türetilen **domain** (`surface_integral` / `general_math`) Katman 5 sistem promptunu seçer.

**Katman 2 (Nougat-OCR):** Swin Transformer + mBART; pikselden LaTeX. Albumentations çakışması → `_setup_albumentations_bypass()`; transformers `generate()` doğrulaması → monkey-patch. **42/58 PDF (%72,4)** başarı; 17 PDF’de boş çıktı veya `[repetition]`; ortalama ~10,7 s/PDF. **Katman 3 (VLM):** Groq Vision üzerinden **LLaMA 4 Scout 17B**, MoE, DocVQA’da **94,4 ANLS**; **58/58 PDF (%100)**; Nougat’ın düştüğü 16 PDF’yi kurtarır; ortalama ~2,4 s/PDF (Nougat’a göre ~4,5× daha hızlı); toplam maliyet ~**0,01 USD**. Prompt: “sadece oku, çözme”; birinci geçiş zaten tam puan değilse **ikinci VLM geçişi**; `clean_output()` çözüm cümlelerini budar.

**Katman 4 (Synthesis):** Nougat + VLM + ham metin stratejileri; akı gibi problemlerde otomatik ipuçları; yüzey integrali için 8 adımlı prosedür, genel matematik için esnek prompt; ortalama ~3200 karakter.

**Katman 5 (LLM):** Claude (gecikme/kalite) → Groq Llama 3.3 70B (hızlı, retry ile yüksek doğruluk ama bazı problemlerde tutarsız) → GPT-4o (sınırlı test) → **Gemini 2.5 Flash** (%100 test seti, daha yavaş). **Birincil: Gemini, yedek: Groq.**

**Katman 6:** LLM çıktısından arayüz için tek bir **final satır** çıkarımı (`\boxed{}`, `FINAL_ANSWER:`, kuyruk heuristikleri). Sınıf adı geçmişten “SymPyVerifier” olsa da bu katmanda referans cevap doğrulaması yok. **Yinelemeli denemelerde sayısal örtüşme** `run.STEPSolver._solve_with_consensus` içinde `latex_parser.parse_latex_to_value` ile yapılır.

---

#### Hatalar ve çözümler

Nougat: sahte modül + `generate` yaması + VLM yedek. VLM: `clean_output()` en başta problem metnini siliyordu — kill-phrase revizyonu; çıktı uzunluğu ~%570 arttı. Cevap çıkarma: Gemini `\boxed{}` tutarsızlığı — 7 kademeli strateji + takip çağrısı.

---

#### Çalışma zamanı ve güvenilirlik güncellemeleri

**Katman 0 / 2 — raster yeniden kullanım:** `dpi` verilmezse Katman 0, yapılandırmadaki **`NOUGAT_DPI`** ile raster üretir; Nougat ile uyum korunur. Her `page_*.png` klasöründe **`.step_raster_meta`** bulunur (1. satır: DPI, 2. satır: PDF SHA-256). Katman 2, bu dosya ve sayfa adları güncel PDF ile örtüşmedikçe PNG’leri **yeniden çizer**; örtüşürse **yeniden rasterize etmeden** kullanır. **`page_*.png`** listesi **sayısal** sıralanır; 9. sayfanın üzerindeki PDF’lerde sıra bozulmaz.

**Katman 3 — VLM eşzamanlılığı:** Sayfa başına API çağrıları **sınırlı iş parçacığı havuzu** ile yürütülebilir. Yoğun RPM veya tam seri çalışma için **`STEP_VLM_PAGE_WORKERS`** (ör. `1`) ayarlanabilir.

**Web arayüzü:** Eşzamanlı çözüm sayısı **`STEP_WEB_MAX_CONCURRENT_SOLVES`** ile sınırlanır (varsayılan 2). Bitmemiş bir görev varken **aynı güvenli dosya adıyla** ikinci yükleme **HTTP 409** döner. Nougat ve GPU yükü yüksekse limit düşürülebilir.

**Katman 6 — cevap temizliği:** Final metin çıkarımında düz `split('=')` kullanılmaz; **süslü parantez derinliği** ile yalnızca üst düzeydeki son `=` dikkate alınır; `\text{...}` gibi yapılar budanmaz.

---

#### Teknoloji seçimleri

PyMuPDF (hız); Nougat (görüntüden LaTeX); LLaMA 4 Scout + Groq (VLM); Gemini (ana metin çözücü); Groq 70B (yedek); SymPy tabanlı `latex_parser` (denemeler arası sayısal karşılaştırma); Flask + SSE; MathJax 3.

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
python run.py path/to/problem.pdf
python run.py path/to/problem.pdf --with-nougat
python run.py --check
python run.py Surface_Integration/
```

Docs, reports, and extra tooling: **`Step_Project_Disari/`** in the repo root.

Editable narrative source (same content you maintain): **`../STEP_Pipeline_Rapor_Paragraf.md`** (one level above this folder).
