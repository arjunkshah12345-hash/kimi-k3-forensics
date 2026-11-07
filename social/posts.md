# Launch posts: Kimi K3 Weight Forensics

Drafts only. Nothing here has been posted. Every number traces to `results/v1`, `results/v2` or `site/src/data/findings.json`.

Links:
- Site: https://arjunkshah12345-hash.github.io/kimi-k3-forensics/
- Repo: https://github.com/arjunkshah12345-hash/kimi-k3-forensics
- Video: `launch/videos/k3-forensics/renders/k3-forensics-launch.mp4` (40 s, 1920×1080)

Each tweet below is under 280 characters.

---

## 1. Main thread (attach the video to 1/)

**1/**
Kimi K3 is 2.8 trillion parameters. 1.56 TB. No free GPU can run it.

So I didn't run it. I read it.

546 GB of tensors pulled straight out of the checkpoint with HTTP range requests, analyzed on one free Kaggle T4.

Here's what the weights say 🧵

**2/**
The trick: a safetensors file starts with a JSON header listing every tensor's byte offsets. Hugging Face serves byte ranges.

So you can fetch one tensor out of a 1.56 TB model:

Range: bytes=2348810824-4697621063

That's K3's whole embedding matrix. Nothing else.

**3/**
Biggest finding: late experts become spelling experts.

I read out what each of K3's 82,432 routed experts writes. In layers 1–47, ~1% of experts write one word family. In layers 82–91 it's 29%. Layer 89: 48%.

**4/**
What a "spelling expert" looks like.

Layer 56, expert 743 writes: ` java` `-java` `java` `(java` `_java` `Java` ` JAVA`

Layer 84, expert 757: ` kg` `KG` `Kg` `公斤` ` pound`

One word. Every casing, leading space and glued punctuation.

**5/**
Only 16 of 163,584 tokens were never trained.

13 are bytes that can't appear in valid UTF-8. One is a broken Cyrillic pair. And two are an adjacent pair, `\tBmori` and `\tPnapi`, which look like genome species codes that made it into the tokenizer but not the training data.

**6/**
Also in the weights:
• linear-attention memory half-life grows from 4 tokens (layer 0) to 117 (layer 86)
• no two experts are duplicates (most similar pair of 2,688: cosine 0.24)
• the load balancer pushes popular experts down to −0.76 and never boosts above 0.15

**7/**
All code, raw outputs and the site are open (MIT). Both runs together: 93 minutes on a free T4.

Site: https://arjunkshah12345-hash.github.io/kimi-k3-forensics/
Code + data: https://github.com/arjunkshah12345-hash/kimi-k3-forensics

---

## 2. Single post: the spelling experts (attach a screenshot of the site's spelling chart)

Half the experts in Kimi K3's layer 89 are spelling experts.

Each one writes a single word in every surface form: ` java` `-java` `_java` `Java` ` JAVA`.

1% of experts do this in the first half of the network. 48% by layer 89.

https://arjunkshah12345-hash.github.io/kimi-k3-forensics/

---

## 3. Single post: the 16 untrained tokens

Kimi K3 has 163,584 tokens. Only 16 never got trained.

13 are bytes UTF-8 forbids. 1 is a broken Cyrillic pair. The last 2 sit side by side: `\tBmori` and `\tPnapi`, likely silkworm and butterfly species codes from genome files.

The tokenizer saw them. The model never did.

---

## 4. Thread: how to study a 1.56 TB model on a free GPU (method, for builders)

**1/**
You don't need 1.56 TB of disk or an H100 cluster to study a frontier open-weight model.

How I analyzed Kimi K3 (2.8T params) on a free Kaggle T4 in 93 minutes 🧵

**2/**
Safetensors = 8-byte header length + JSON header + raw bytes. The header maps every tensor name to [start, end] offsets.

Read 8 bytes, read the header, then request only the byte range you need. HF's CDN supports Range, so a 50 MB router is a 50 MB download.

**3/**
K3's experts are MXFP4: two 4-bit E2M1 values per byte and one power-of-two scale per 32 weights. Decoding is a 16-entry lookup times 2^(scale−127).

The nibble order checked itself: low-nibble-first gives structured responses, the other order gives noise.

**4/**
Coalesce adjacent tensors into ~96 MB chunks, fetch on 12–24 threads, and run the GPU math while the next chunk downloads. Kaggle sustained ~100 MB/s.

Run 2 read the down-projection of all 82,432 experts: 493 GB in 81 minutes.

**5/**
Everything is reproducible: two Kaggle scripts, the raw results and the analysis.

https://github.com/arjunkshah12345-hash/kimi-k3-forensics

---

## 5. Single post: the short hook (attach the video)

Kimi K3: 1.56 TB.
A free Kaggle T4: 14.5 GB.

So I read the weights instead of running them. 546 GB of range requests later: late experts become spelling experts, and only 16 tokens were never trained.

https://arjunkshah12345-hash.github.io/kimi-k3-forensics/
