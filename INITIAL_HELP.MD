## MODEL

**Amharic OmniVoice**

Start from:

```text
african-low-resource/omnivoice-amharic
```

rather than raw:

```text
k2-fsa/OmniVoice
```

because the former is already adapted to Amharic. ([Hugging Face][9])

## DATASET

```text
snapwre/amharic-speech
```

Use the existing:

```text
train
validation
test
```

splits.

## TRAINING APPROACH

Do:

```text
Amharic OmniVoice
       ↓
small 5–10 speaker experiment
       ↓
verify training/inference
       ↓
10–20 hour experiment
       ↓
full 42.1-hour train set
```

Don't immediately train everything.

## VOICE CLONING

Keep the zero-shot reference mechanism.

Test:

```text
seen speaker
```

and:

```text
unseen speaker
```

separately.

## EXPECTED RESULT

A useful Amharic TTS + zero-shot voice-cloning system is **technically realistic**.

In fact, your dataset is an excellent candidate because it provides 493 speakers and a speaker-disjoint test set.

The biggest uncertainty isn't whether the model can learn Amharic.

The bigger question is:

> **How much does your 51.5-hour corpus improve the existing Amharic OmniVoice model?**

That's exactly what the experiments should measure.

---

# ============================================================

# PHASE 2 — IMPLEMENTATION / GOOGLE COLAB

# ============================================================

Now we can actually build it.

The official OmniVoice training pipeline uses JSONL → audio token extraction → WebDataset shards → training. ([GitHub][14])

---

# 1. Create the Colab Notebook

Open Google Colab and select:

**Runtime → Change runtime type → GPU**

For the first experiment:

```text
T4 16 GB
```

is acceptable.

For serious training:

```text
L4 24 GB
```

is better.

A100 is ideal if available.

---

# 2. First Cell — Check GPU

```python
!nvidia-smi
```

You want something like:

```text
Tesla T4
```

or:

```text
NVIDIA L4
```

or:

```text
A100
```

Then:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:",
      round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
      "GB")
```

---

# 3. Mount Google Drive

You absolutely want checkpoints on Drive.

```python
from google.colab import drive

drive.mount("/content/drive")
```

Create the project:

```python
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/amharic_tts")

for directory in [
    "checkpoints",
    "datasets",
    "samples",
    "logs",
    "configs",
    "manifests",
    "audio",
    "tokens",
    "final_model",
]:
    (PROJECT / directory).mkdir(parents=True, exist_ok=True)

print(PROJECT)
```

You'll get:

```text
MyDrive/
└── amharic_tts/
    ├── checkpoints/
    ├── datasets/
    ├── samples/
    ├── logs/
    ├── configs/
    ├── manifests/
    ├── audio/
    ├── tokens/
    └── final_model/
```

---

# 4. Install OmniVoice

The current OmniVoice documentation uses PyTorch 2.8 and provides its own training package. ([Hugging Face][15])

For Colab, first inspect the installed environment:

```python
!python --version
!pip --version
```

Then:

```python
!pip install -q -U omnivoice datasets soundfile librosa accelerate
```

Clone the repository because we need its training scripts:

```python
!git clone https://github.com/k2-fsa/OmniVoice.git
```

Enter it:

```python
%cd /content/OmniVoice
```

Install the repository dependencies:

```python
!pip install -q -e .
```

Verify:

```python
import omnivoice

print("OmniVoice imported successfully")
```

---

# 5. Test the Existing Amharic Model FIRST

Before touching your dataset, make sure the base model works.

```python
!pip install -q soundfile
```

Then:

```python
import torch
import soundfile as sf

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

MODEL_ID = "african-low-resource/omnivoice-amharic"

model = OmniVoice.from_pretrained(
    MODEL_ID,
    device_map="cuda:0",
    dtype=torch.float16,
)

print("Model loaded")
```

Generate Amharic:

```python
text = "ሰላም፣ ይህ የአማርኛ የድምፅ ሙከራ ነው።"

audio = model.generate(
    text=text,
    language="Amharic",
    generation_config=OmniVoiceGenerationConfig(
        num_step=32,
        guidance_scale=2.0,
    ),
)

sf.write(
    "/content/drive/MyDrive/amharic_tts/samples/base_amharic.wav",
    audio[0],
    24000,
)
```

Play it:

```python
from IPython.display import Audio, display

display(
    Audio(
        "/content/drive/MyDrive/amharic_tts/samples/base_amharic.wav"
    )
)
```

**Do not proceed to training if this doesn't work.**

---

# 6. Test Zero-Shot Voice Cloning

Upload a reference recording.

```python
from google.colab import files

uploaded = files.upload()
```

Suppose it is:

```text
reference.wav
```

Then:

```python
REFERENCE = next(iter(uploaded.keys()))

prompt = model.create_voice_clone_prompt(
    ref_audio=REFERENCE,
    ref_text=None,
)
```

Generate:

```python
text = "ዛሬ በኢትዮጵያ በጣም ጥሩ ቀን ነው።"

audio = model.generate(
    text=text,
    language="Amharic",
    voice_clone_prompt=prompt,
)

sf.write(
    "/content/drive/MyDrive/amharic_tts/samples/base_clone.wav",
    audio[0],
    24000,
)
```

Listen:

```python
display(
    Audio(
        "/content/drive/MyDrive/amharic_tts/samples/base_clone.wav"
    )
)
```

This establishes your **baseline**.

---

# 7. Download Your Dataset

Now:

```python
from datasets import load_dataset

dataset = load_dataset(
    "snapwre/amharic-speech"
)

dataset
```

You should see approximately:

```text
train
validation
test
```

---

# 8. Inspect the Schema

Do **not** assume column names.

Run:

```python
for split in dataset:
    print("\nSPLIT:", split)
    print("Rows:", len(dataset[split]))
    print("Columns:")
    print(dataset[split].column_names)
```

You should find fields including:

```text
audio
sentence
speaker_id
language
duration_s
speech_s
sample_rate
...
```

The current dataset card confirms those fields. ([Hugging Face][1])

---

# 9. Inspect One Sample

```python
sample = dataset["train"][0]

for key, value in sample.items():
    if key == "audio":
        print(
            key,
            {
                "sampling_rate": value["sampling_rate"],
                "shape": value["array"].shape,
            }
        )
    else:
        print(key, value)
```

---

# 10. Calculate Dataset Statistics

```python
import pandas as pd

for split in dataset:
    ds = dataset[split]

    durations = ds["duration_s"]
    speech = ds["speech_s"]

    print(f"\n===== {split.upper()} =====")

    print("Samples:", len(ds))
    print("Speakers:", len(set(ds["speaker_id"])))

    print(
        "Hours:",
        sum(durations) / 3600
    )

    print(
        "Speech hours:",
        sum(speech) / 3600
    )

    print(
        "Duration mean:",
        sum(durations) / len(durations)
    )

    print(
        "Duration min:",
        min(durations)
    )

    print(
        "Duration max:",
        max(durations)
    )
```

---

# 11. Speaker Distribution

```python
from collections import Counter

for split in dataset:
    counts = Counter(dataset[split]["speaker_id"])

    print(
        f"\n{split}: "
        f"{len(counts)} speakers"
    )

    print(
        "Min clips:",
        min(counts.values())
    )

    print(
        "Max clips:",
        max(counts.values())
    )

    print(
        "Average clips:",
        sum(counts.values()) / len(counts)
    )
```

---

# 12. Check Duplicate Text

```python
for split in dataset:
    texts = dataset[split]["sentence"]

    duplicates = len(texts) - len(set(texts))

    print(
        split,
        "duplicate transcripts:",
        duplicates
    )
```

Because the official dataset says the splits are prompt-disjoint, you should expect no cross-split prompt leakage. ([Hugging Face][12])

---

# 13. Check Missing Data

```python
for split in dataset:
    ds = dataset[split]

    missing_text = sum(
        x is None or not str(x).strip()
        for x in ds["sentence"]
    )

    missing_speaker = sum(
        x is None or not str(x).strip()
        for x in ds["speaker_id"]
    )

    print(f"\n{split}")
    print("Missing text:", missing_text)
    print("Missing speaker:", missing_speaker)
```

---

# 14. Normalize Amharic Text

Do **not** transliterate.

Use Unicode normalization.

```python
import unicodedata
import re

def normalize_amharic(text):
    if text is None:
        return ""

    text = str(text)

    # Unicode canonical normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text
```

Test:

```python
examples = [
    "ሰላም     እንዴት ነህ?",
    "  ይህ  የአማርኛ   ሙከራ ነው። "
]

for x in examples:
    print(normalize_amharic(x))
```

---

# 15. Don't Aggressively Remove Punctuation

Keep useful punctuation:

```text
።
፣
፤
፦
?
!
```

because punctuation can affect:

* pauses
* sentence boundaries
* intonation

So don't do something like:

```python
text = re.sub(r"[^\w\s]", "", text)
```

for this project.

---

# 16. Filter Bad Audio

Your dataset already provides:

```text
duration_s
speech_s
lufs
```

so use them.

For example:

```python
MIN_DURATION = 1.5
MAX_DURATION = 30.0
MIN_SPEECH = 0.8
```

Create a filtering function:

```python
def valid_sample(row):
    text = normalize_amharic(row["sentence"])

    if not text:
        return False

    if row["duration_s"] < MIN_DURATION:
        return False

    if row["duration_s"] > MAX_DURATION:
        return False

    if row["speech_s"] < MIN_SPEECH:
        return False

    return True
```

Test:

```python
for split in dataset:
    valid = sum(
        valid_sample(dataset[split][i])
        for i in range(len(dataset[split]))
    )

    print(
        split,
        valid,
        "/",
        len(dataset[split])
    )
```

Don't throw away large amounts of data blindly.

---

# 17. Export Audio + JSONL

OmniVoice's training pipeline expects JSONL entries of the form:

```json
{
  "id": "...",
  "audio_path": "...",
  "text": "...",
  "language_id": "am"
}
```

The official data-preparation documentation specifies this exact manifest concept. ([GitHub][14])

Create:

```python
import json
import os
import soundfile as sf
from pathlib import Path

AUDIO_DIR = PROJECT / "audio"

for split in ["train", "validation", "test"]:
    (AUDIO_DIR / split).mkdir(
        parents=True,
        exist_ok=True
    )
```

---

# 18. Convert Dataset Audio

```python
def export_split(split_name):
    ds = dataset[split_name]

    manifest_path = (
        PROJECT
        / "manifests"
        / f"{split_name}.jsonl"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as manifest:

        for i in range(len(ds)):

            row = ds[i]

            if not valid_sample(row):
                continue

            text = normalize_amharic(
                row["sentence"]
            )

            clip_id = row["clip_id"]

            audio_path = (
                AUDIO_DIR
                / split_name
                / f"{clip_id}.wav"
            )

            audio = row["audio"]

            sf.write(
                audio_path,
                audio["array"],
                audio["sampling_rate"]
            )

            item = {
                "id": clip_id,
                "audio_path": str(
                    audio_path.resolve()
                ),
                "text": text,
                "language_id": "am",
            }

            manifest.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    return manifest_path
```

Then:

```python
train_manifest = export_split("train")
val_manifest = export_split("validation")
test_manifest = export_split("test")

print(train_manifest)
print(val_manifest)
print(test_manifest)
```

---

# 19. Important: Storage

This is one place where I recommend a change from your original plan.

**Do not keep repeatedly reading 16,000+ individual audio files from Google Drive during training.**

Drive is useful for persistence, but its I/O is much slower than local Colab storage.

Use:

```text
Google Drive
     ↓
persistent raw dataset
     ↓
/content/local_dataset
     ↓
training
```

and save:

```text
checkpoints
tokens
models
```

back to Drive.

---

# 20. Copy Dataset to Local Colab Storage

For example:

```python
!mkdir -p /content/amharic_tts
```

Then copy:

```python
!cp -r /content/drive/MyDrive/amharic_tts/audio /content/amharic_tts/
```

And manifests:

```python
!cp -r /content/drive/MyDrive/amharic_tts/manifests /content/amharic_tts/
```

For a full 51-hour dataset this can take time.

---

# 21. Proof-of-Concept Dataset

Do **not** start with 42 hours.

Create a tiny experiment.

Use:

```text
5–10 speakers
~1–3 hours
```

First determine speaker IDs:

```python
from collections import defaultdict

speaker_to_rows = defaultdict(list)

for i in range(len(dataset["train"])):
    row = dataset["train"][i]

    speaker_to_rows[
        row["speaker_id"]
    ].append(i)
```

Select 10 speakers:

```python
selected_speakers = list(
    speaker_to_rows.keys()
)[:10]

print(selected_speakers)
```

Then build a small JSONL containing those speakers.

---

# 22. Why This Experiment Matters

Your first experiment isn't trying to produce the final model.

It answers:

```text
Can OmniVoice fine-tuning
improve/maintain Amharic quality
on this corpus?
```

You should compare:

```text
BASE MODEL
     vs
FINE-TUNED MODEL
```

on identical sentences.

---

# 23. Prepare OmniVoice Token Shards

The official OmniVoice pipeline uses:

```text
JSONL
  ↓
extract_audio_tokens.py
  ↓
WebDataset .tar shards
  ↓
data.lst
  ↓
training
```

The audio tokenizer is:

```text
eustlb/higgs-audio-v2-tokenizer
```

and the official pipeline uses it to generate 8-layer discrete audio tokens. ([GitHub][14])

For your PoC:

```python
%cd /content/OmniVoice
```

Then:

```python
!mkdir -p /content/amharic_tts/tokens/train
!mkdir -p /content/amharic_tts/tokens/dev
```

Run:

```python
!CUDA_VISIBLE_DEVICES=0 python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl /content/amharic_tts/manifests/train_poc.jsonl \
    --tar_output_pattern "/content/amharic_tts/tokens/train/shard-%06d.tar" \
    --jsonl_output_pattern "/content/amharic_tts/tokens/train/shard-%06d.jsonl" \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 1 \
    --shuffle True
```

Do the same for validation:

```python
!CUDA_VISIBLE_DEVICES=0 python -m omnivoice.scripts.extract_audio_tokens \
    --input_jsonl /content/amharic_tts/manifests/validation_poc.jsonl \
    --tar_output_pattern "/content/amharic_tts/tokens/dev/shard-%06d.tar" \
    --jsonl_output_pattern "/content/amharic_tts/tokens/dev/shard-%06d.jsonl" \
    --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
    --nj_per_gpu 1 \
    --shuffle True
```

---

# 24. Create Training Data Config

Create:

```python
import json

data_config = {
    "train": [
        {
            "language_id": "am",
            "manifest_path": [
                "/content/amharic_tts/tokens/train/data.lst"
            ],
            "repeat": 1
        }
    ],
    "dev": [
        {
            "language_id": "am",
            "manifest_path": [
                "/content/amharic_tts/tokens/dev/data.lst"
            ],
            "repeat": 1
        }
    ]
}

with open(
    "/content/amharic_tts/data_config.json",
    "w"
) as f:
    json.dump(
        data_config,
        f,
        indent=2
    )
```

---

# 25. PoC Training Configuration

For your first experiment, don't use the full 12,000 steps.

Start around:

```text
500–1,000 steps
```

and inspect the result.

Example:

```json
{
  "llm_name_or_path": "Qwen/Qwen3-0.6B",
  "audio_vocab_size": 1025,
  "audio_mask_id": 1024,
  "num_audio_codebook": 8,
  "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
  "drop_cond_ratio": 0.1,
  "prompt_ratio_range": [0.0, 0.3],
  "mask_ratio_range": [0.0, 1.0],
  "language_ratio": 0.8,
  "use_pinyin_ratio": 0.0,
  "instruct_ratio": 0.0,
  "only_instruct_ratio": 0.0,
  "resume_from_checkpoint": null,
  "init_from_checkpoint": "african-low-resource/omnivoice-amharic",
  "learning_rate": 1e-5,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,
  "steps": 1000,
  "seed": 42,
  "warmup_type": "ratio",
  "warmup_ratio": 0.05,
  "warmup_steps": 0,
  "batch_tokens": 4096,
  "gradient_accumulation_steps": 1,
  "num_workers": 2,
  "mixed_precision": "fp16",
  "allow_tf32": true,
  "logging_steps": 25,
  "eval_steps": 250,
  "save_steps": 250,
  "keep_last_n_checkpoints": 3,
  "attn_implementation": "sdpa"
}
```

### Why `sdpa`?

OmniVoice's documentation says its default `flex_attention` requires PyTorch ≥2.5 and a compatible GPU, while SDPA provides broader GPU compatibility. ([GitHub][16])

For Colab, SDPA is safer for the initial experiment.

---

# 26. Start PoC Training

From the repository:

```python
%cd /content/OmniVoice
```

Then:

```python
!accelerate launch \
    --num_processes 1 \
    -m omnivoice.cli.train \
    --train_config /content/amharic_tts/train_config_poc.json \
    --data_config /content/amharic_tts/data_config.json \
    --output_dir /content/drive/MyDrive/amharic_tts/checkpoints/poc
```

This follows the official OmniVoice training mechanism. ([GitHub][17])

---

# 27. Checkpointing

Your checkpoints should be on:

```text
Google Drive
```

For example:

```text
MyDrive/
└── amharic_tts/
    └── checkpoints/
        └── poc/
            ├── checkpoint-250/
            ├── checkpoint-500/
            ├── checkpoint-750/
            └── checkpoint-1000/
```

The official training configuration supports:

```text
save_steps
resume_from_checkpoint
```

and initialization from a pretrained checkpoint. ([GitHub][17])

---

# 28. Resume After Colab Disconnect

Change:

```json
"resume_from_checkpoint":
"/content/drive/MyDrive/amharic_tts/checkpoints/poc/checkpoint-750"
```

Then rerun training.

The official OmniVoice training system supports checkpoint resumption. ([GitHub][17])

---

# 29. Full Training

Once the PoC works, prepare the complete dataset.

Your actual training data is approximately:

```text
42.126 hours
360 speakers
13,792 clips
```

with:

```text
4.699 hours
28 speakers
```

for validation, and:

```text
4.722 hours
105 speakers
```

for testing. ([Hugging Face][12])

That's an excellent setup.

---

# 30. Full Training Configuration

I would start with:

```text
Learning rate:
1e-5

Batch tokens:
4096 on T4
8192+ on L4/A100

Gradient accumulation:
2–8 depending on VRAM

Precision:
fp16 on T4
bf16 on L4/A100

Attention:
SDPA initially

Gradient checkpointing:
enabled if available

Steps:
start 5,000
evaluate
then potentially 8,000–12,000
```

The existing Amharic OmniVoice training used:

```text
2e-5
12,000 steps
28,672 batch tokens
bf16
```

but that was performed on a much larger 331-hour corpus and should **not** be blindly copied to your smaller 51.5-hour dataset. ([Hugging Face][10])

---

# 31. T4 / L4 / A100

## T4 — 16 GB

Use approximately:

```text
batch_tokens = 2048–4096
gradient_accumulation = 4–8
fp16 = true
SDPA
```

The T4 is good for:

* inference
* PoC
* small fine-tuning

but full training will be slow.

---

## L4 — 24 GB

Better:

```text
batch_tokens = 4096–8192
gradient_accumulation = 2–4
bf16 = true
SDPA
```

This is probably the **best Colab GPU for your project if available**.

---

## A100 — 40/80 GB

Use:

```text
batch_tokens = 8192–16384+
gradient_accumulation = 1–2
bf16 = true
```

You can substantially increase throughput.

---

# 32. Evaluation Dataset

Don't evaluate only on training speakers.

Your test set already has:

```text
105 speakers
```

who don't appear in training.

That's extremely valuable.

You should create:

```text
TEST-A
Seen speakers

TEST-B
Unseen speakers
```

The existing dataset's speaker-disjoint test split makes TEST-B easy.

---

# 33. Fixed Amharic Evaluation Sentences

Create something like:

```python
TEST_SENTENCES = [
    "ሰላም፣ እንዴት ነህ?",
    "ዛሬ በኢትዮጵያ በጣም ጥሩ ቀን ነው።",
    "ይህ የአማርኛ የድምፅ ሙከራ ነው።",
    "ነገ ጠዋት ወደ አዲስ አበባ እሄዳለሁ።",
    "የኢትዮጵያ ባህል እና ታሪክ በጣም ሀብታም ነው።",
]
```

Add harder examples containing:

```text
ጠ
ጰ
ጸ
ፀ
ቸ
ጨ
```

because these sounds are useful for testing Amharic pronunciation.

---

# 34. Compare Base vs Fine-Tuned

For every sentence:

```text
BASE MODEL
      ↓
base.wav

FINE-TUNED MODEL
      ↓
finetuned.wav
```

Then listen to both.

Don't judge only training loss.

---

# 35. Voice Cloning Evaluation

For an unseen test speaker:

```text
reference.wav
```

should come from the test speaker.

Then:

```text
Amharic text
+
reference.wav
```

→ generated speech.

Do **not** use a reference clip from the same utterance that you're testing against.

---

# 36. Voice Clone Code

The actual OmniVoice API is:

```python
prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text=None,
)

audio = model.generate(
    text=text,
    language="Amharic",
    voice_clone_prompt=prompt,
)
```

This is an actual API documented by the Amharic OmniVoice model, rather than an invented API. ([Hugging Face][11])

---

# 37. Reference Audio Experiments

Test:

```text
5 seconds
10 seconds
20 seconds
30 seconds
```

using the same speaker.

For example:

```text
reference_5s.wav
reference_10s.wav
reference_20s.wav
reference_30s.wav
```

Then generate the same sentence.

Compare:

```text
speaker similarity
pronunciation
prosody
noise
stability
```

The current Amharic model specifically documents a 10-second reference as its cloning example. ([Hugging Face][18])

So:

**10 seconds should be your first standard reference length.**

---

# 38. ASR Evaluation

Install an Amharic ASR model later and run:

```text
generated audio
      ↓
Amharic ASR
      ↓
generated transcript
      ↓
compare with target
```

Calculate:

```text
CER
WER
```

CER is especially useful for Amharic because you're dealing with Ethiopic-script character sequences.

---

# 39. Human Evaluation

Create a simple spreadsheet:

| Sample | Naturalness | Pronunciation | Speaker similarity | Prosody |
| ------ | ----------: | ------------: | -----------------: | ------: |
| 1      |         1–5 |           1–5 |                1–5 |     1–5 |
| 2      |         1–5 |           1–5 |                1–5 |     1–5 |

Use the same evaluation sentences for:

```text
Base
Fine-tuned
```

and preferably blind the evaluator to which is which.

---

# 40. Expected Results

Don't interpret dataset size as a guarantee.

### 1–3 hours

Useful only for:

```text
pipeline verification
```

You should not expect a production-quality model.

### 10 hours

Enough to determine whether your training setup is behaving correctly.

### 25 hours

Potentially meaningful adaptation.

### 51.5 hours

This is where your dataset becomes genuinely interesting.

But remember:

**the existing Amharic OmniVoice model already has ~331 hours of Amharic training data.**

So your 51.5 hours should be thought of as:

```text
domain / speaker / pronunciation / corpus adaptation
```

rather than:

```text
teach OmniVoice Amharic from zero
```

The existing model's 331-hour training corpus is documented here. ([Hugging Face][10])

---

# 41. Most Important Experiment

I would actually run **three models**.

### Model A

```text
Existing Amharic OmniVoice
```

No training.

### Model B

```text
Amharic OmniVoice
+
10-speaker / 1–3h fine-tuning
```

### Model C

```text
Amharic OmniVoice
+
full 42.1h fine-tuning
```

Then compare:

```text
                 Base       PoC       Full
------------------------------------------------
Pronunciation
Naturalness
Speaker similarity
Unseen speaker
Prosody
CER
WER
```

This will tell you whether your dataset actually improves the model.

---

# 42. One Important Change to Your Original Plan

I would **not** do this:

```text
Chatterbox V3
      ↓
modify tokenizer
      ↓
invent [am]
      ↓
fine-tune
```

for the first project.

Instead:

```text
Existing Amharic OmniVoice
          ↓
snapwre/amharic-speech
          ↓
small PoC
          ↓
full fine-tuning
          ↓
zero-shot voice cloning
```

This is substantially lower risk.

---

# 43. Final Architecture

Your finished system should look like:

```text
                  ┌──────────────────────┐
                  │ Amharic Text         │
                  │ ሰላም፣ ይህ ሙከራ ነው። │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Amharic OmniVoice    │
                  │ Fine-tuned model     │
                  └──────────┬───────────┘
                             ▲
                             │
                  ┌──────────┴───────────┐
                  │ Reference WAV        │
                  │ 5–30 seconds         │
                  │ unseen speaker       │
                  └──────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Generated Amharic    │
                  │ speech               │
                  └──────────────────────┘
```

---

# 44. The Project Structure I Recommend

```text
/content/drive/MyDrive/amharic_tts/

├── audio/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── manifests/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
│
├── tokens/
│   ├── train/
│   ├── dev/
│   └── test/
│
├── checkpoints/
│   ├── poc/
│   └── full/
│
├── samples/
│   ├── base/
│   ├── poc/
│   └── full/
│
├── configs/
│   ├── poc.json
│   └── full.json
│
├── logs/
│
└── final_model/
```

---

# 45. What I Would Do First

Don't launch the 51-hour training yet.

Follow this exact order:

```text
1. Create Colab GPU runtime
        ↓
2. Install OmniVoice
        ↓
3. Load existing Amharic OmniVoice
        ↓
4. Generate normal Amharic
        ↓
5. Test 10-second voice cloning
        ↓
6. Download snapwre/amharic-speech
        ↓
7. Inspect dataset
        ↓
8. Export 5–10 speaker subset
        ↓
9. Tokenize subset
        ↓
10. Train 500–1,000 steps
        ↓
11. Generate fixed evaluation sentences
        ↓
12. Compare base vs fine-tuned
        ↓
13. If successful → tokenize all 42h
        ↓
14. Full fine-tuning
        ↓
15. Evaluate 105 unseen speakers
        ↓
16. Test 5/10/20/30-second references
        ↓
17. Select best checkpoint
        ↓
18. Build final inference notebook/API
```

specification asked for research before implementation; 
the research changes the original model choice, so **the implementation above deliberately follows OmniVoice rather than Chatterbox**. 

[1]: https://huggingface.co/datasets/snapwre/amharic-speech "snapwre/amharic-speech · Datasets at Hugging Face"
[2]: https://huggingface.co/ResembleAI/chatterbox?utm_source=chatgpt.com "ResembleAI/chatterbox · Hugging Face"
[3]: https://huggingface.co/mlx-community/chatterbox-multilingual-v3/blob/main/config.json?utm_source=chatgpt.com "config.json · mlx-community/chatterbox-multilingual-v3 at main"
[4]: https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/models/tokenizers/tokenizer.py?utm_source=chatgpt.com "chatterbox/src/chatterbox/models/tokenizers/tokenizer.py at master · resemble-ai/chatterbox · GitHub"
[5]: https://huggingface.co/gabar-tech/chatterbox-afaan-oromo?utm_source=chatgpt.com "gabar-tech/chatterbox-afaan-oromo · Hugging Face"
[6]: https://github.com/gokhaneraslan/chatterbox-finetuning/blob/main/train.py?utm_source=chatgpt.com "chatterbox-finetuning/train.py at main · gokhaneraslan/chatterbox-finetuning · GitHub"
[7]: https://github.com/Ahmed-Ezzat20/chatterbox-finetuning-multilingual?utm_source=chatgpt.com "GitHub - Ahmed-Ezzat20/chatterbox-finetuning-multilingual: SoTA open-source TTS · GitHub"
[8]: https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md?utm_source=chatgpt.com "OmniVoice/docs/languages.md at master · k2-fsa/OmniVoice · GitHub"
[9]: https://huggingface.co/african-low-resource/omnivoice-amharic/tree/main "african-low-resource/omnivoice-amharic at main"
[10]: https://huggingface.co/african-low-resource/omnivoice-amharic/commit/e1f04839ad2a9724e84984fae325d59deb5e9158?utm_source=chatgpt.com "Complete README with Colab inference code and model comparison · african-low-resource/omnivoice-amharic at e1f0483"
[11]: https://huggingface.co/gheero-Leyu/amharic-omnivoice-tts?utm_source=chatgpt.com "gheero-Leyu/amharic-omnivoice-tts · Hugging Face"
[12]: https://huggingface.co/datasets/snapwre/amharic-speech?utm_source=chatgpt.com "snapwre/amharic-speech · Datasets at Hugging Face"
[13]: https://github.com/k2-fsa/OmniVoice/blob/master/examples/README.md "OmniVoice/examples/README.md at master · k2-fsa/OmniVoice · GitHub"
[14]: https://github.com/k2-fsa/OmniVoice/blob/master/docs/data_preparation.md?utm_source=chatgpt.com "OmniVoice/docs/data_preparation.md at master · k2-fsa/OmniVoice · GitHub"
[15]: https://huggingface.co/k2-fsa/OmniVoice?utm_source=chatgpt.com "k2-fsa/OmniVoice · Hugging Face"
[16]: https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md "OmniVoice/docs/training.md at master · k2-fsa/OmniVoice · GitHub"
[17]: https://github.com/k2-fsa/OmniVoice/blob/master/docs/training.md?utm_source=chatgpt.com "OmniVoice/docs/training.md at master · k2-fsa/OmniVoice · GitHub"
[18]: https://huggingface.co/african-low-resource/omnivoice-amharic/commit/9e6a17ec03d993fe32d5327844361a2356041c70?utm_source=chatgpt.com "Update README.md · african-low-resource/omnivoice-amharic at 9e6a17e"
