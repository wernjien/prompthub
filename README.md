# PromptHub

Turn a rough idea — or a photo, or a video clip — into a properly formatted
prompt for **Krea 2** (images) or **MiniMax H3** (video), entirely on your
own machine. No cloud APIs, no content filtering, nothing leaves the
computer at runtime.

You get two entries in the chat model picker:

| Entry | Gives you |
|---|---|
| **Krea2** | A Krea 2 image prompt — one flowing descriptive paragraph |
| **MiniMax H3** | A MiniMax H3 video prompt in its required three-field format, with the mode (T2VA/I2VA/FL2VA/L2VA) worked out for you |

Each accepts **typed text, an attachment, or both together**.

---

## 1. Install

**Prerequisites:** [Homebrew](https://brew.sh) on macOS, or a package
manager plus an NVIDIA/AMD driver on Linux, or
[winget](https://learn.microsoft.com/windows/package-manager/) on Windows.
At least 16GB RAM/VRAM.

```bash
git clone git@github.com:wernjien/prompthub.git
cd prompthub

./setup/install_macos.sh          # macOS
./setup/install_linux.sh          # Linux
.\setup\install_windows.ps1       # Windows (elevated PowerShell)
```

One command does everything — there are no steps to click through
afterwards. It installs Ollama, ffmpeg and a dedicated Python 3.11 venv,
pulls the two models (`dolphin3:8b` for writing, `llava:13b` for looking at
your attachments, ~13GB total), installs Open WebUI, starts it, creates
your admin account, and deploys both Functions.

**First run takes a while** — model downloads plus a one-off embedding
model fetch. Later starts take seconds.

**Don't open http://localhost:8080 until the script says it's done.** Open
WebUI makes whoever signs up first the admin, so filling in that signup form
by hand takes the slot the installer needs — it can then no longer sign in to
deploy the two Functions, and you end up with an empty model picker. Wait for
the login it prints below.

When it finishes it prints your login:

```
Created Open WebUI admin account: admin@prompthub.local
Credentials saved to /Users/you/PromptHub/.admin_credentials.json
```

Open **http://localhost:8080**, log in with those, and you're ready. Want
your own password instead of a generated one? Set it before installing:

```bash
export PROMPTHUB_ADMIN_PASSWORD='your-password'
```

Check everything landed correctly at any time:

```bash
./setup/verify.sh                 # macOS / Linux
.\setup\verify.ps1                # Windows
```

---

## 2. Use it

Click **+ New Chat**, pick **Krea2** or **MiniMax H3** from the model name
at the top, and send. The reply *is* the prompt — copy it straight into
Krea 2 or MiniMax.

### Three ways to send input

| You send | What happens |
|---|---|
| **Text only** | Your words are the whole brief. |
| **An attachment only** | `llava:13b` looks at the image/clip and describes it; that becomes the brief. |
| **Text + attachment** | Your text is the **intent**; the attachment supplies **concrete detail for the elements it shows**. Merged into one prompt. |

That third one is the useful combination: describe the scene you want, and
attach a reference for a specific element in it.

### Example 1 — text → a Krea 2 image prompt

Pick **Krea2**, send:

> a woman drinking coffee by a rainy window

Real output:

> A woman sits cozily by a window, her hands cradling a warm cup of coffee.
> Raindrops patter gently against the glass, casting a rhythmic symphony as
> the woman finds solace in the quiet hum of the storm outside. The room is
> bathed in the soft, diffused light of a rainy day… The camera, equipped
> with a 50mm lens, captures this intimate moment with a moderate depth of
> field… rendered in a photorealistic style.

Subject → setting → lighting → lens → style, in the flowing prose Krea 2
wants. No comma-tags, no `(word:1.3)` weighting, no negative prompt —
Krea 2 doesn't use those.

### Example 2 — text → a MiniMax H3 video prompt

Pick **MiniMax H3**, send:

> a street cat leaps onto a windowsill and meows, 5 seconds

Real output (abridged):

```
integrated_multimodal_description:
[Shot 1] The scene is set in a dimly lit urban alleyway, with graffiti-covered
walls… A stray street cat with sleek black fur and piercing green eyes… It leaps
gracefully, with agility and purpose, landing softly on the windowsill…

overall_soundscape: The ambient soundscape consists of distant traffic, the
murmur of people from nearby buildings… The cat's meows are loud and demanding…

non_diegetic_music: There is no non-diegetic music playing in the video…

---
Wrote a **T2VA** prompt — nothing was attached, so the prompt has to describe
everything itself.
To force a different one, just say so in the message (I2VA / FL2VA / L2VA).
```

Those three field names are MiniMax's own required structure. The footer
after `---` is PromptHub telling you which mode it picked — see below.

### Example 3 — text + image together

Pick **Krea2**, type your scene *and* attach a reference image (📎):

> a lone fox crossing a frozen lake at dusk

with a photo attached of the specific fox/coat you have in mind. Your text
decides the scene; the image fills in the concrete details of the elements
it shows. Neither overrides the other — your text wins on intent.

### Example 4 — a video clip → a prompt describing it

Pick **MiniMax H3**, attach a clip, optionally add text like "make this
cinematic". The clip's frames are sampled, described, and turned into a
standalone prompt you can use to generate something similar.

---

## 3. The MiniMax mode is worked out for you

H3 needs a different prompt format depending on which reference images
you'll hand the generator. That's a consequence of what you have, not a
preference — so you don't pick it:

| What you supply | Mode | What MiniMax gets alongside the prompt |
|---|---|---|
| Nothing attached | **T2VA** | Prompt only — it invents all visuals |
| One image | **I2VA** | That image as the **first** frame |
| Two images | **FL2VA** | **First + last** frames; it morphs between them |
| One image + you say *"ends on…"* | **L2VA** | That image as the **last** frame |
| A clip | **T2VA** | Nothing — the clip is reference material only |
| A clip + *"start from…"* / *"ends on…"* / *"between…"* | **I2VA** / **L2VA** / **FL2VA** | The matching frames, pulled from the clip |

Every reply says which mode it used and why. **To force one, just say so** —
"use L2VA" beats the inference.

### Reference frames get saved for you

For I2VA / FL2VA / L2VA, MiniMax needs the actual image(s) as well as the
prompt. PromptHub saves them to `~/PromptHub/output/` and prints the paths:

```
Reference frame (Picture 1, t=0) saved to: /Users/you/PromptHub/output/picture1_1789795684.jpg
Supply these to your MiniMax generation alongside the prompt above.
```

If you attached an image, that image is saved. If you attached a clip, the
relevant frames are pulled from it. If you attached nothing, it tells you
so rather than leaving you to discover it later.

### Prompts never refer back to your attachment

The finished prompt gets pasted somewhere that can't see what you uploaded,
so it always reads as a standalone description — never "the attached
image", "as depicted in the photo" or remarks about resolution. Three
things enforce that: the vision model is told to describe the scene rather
than the file, its caption is stripped of medium-talk before the writer
sees it, and video prompts say "frame" rather than "image".

The one exception is MiniMax's own `<Picture 1>` / `Picture 2` notation in
the keyframe modes — those *are* supplied to MiniMax at generation time, so
the format requires naming them.

---

## 4. Everyday commands

```bash
./setup/start.sh     # start Open WebUI (also re-deploys the Functions)
./setup/stop.sh      # stop it
./setup/verify.sh    # health check + manual test checklist
```

(Windows: the matching `.ps1` files.)

Nothing runs at boot — Open WebUI only runs while you've started it. Ollama
runs as a background service from its own installer.

**Edited a prompt in `models/*.py`?** Run `./setup/start.sh` to push the
change into the running instance. It's create-or-update, so re-running is
always safe.

---

## 5. Configuration

Each Function reads settings from Open WebUI's **Valves** panel (Admin
Panel → Functions → ⚙️), or environment variables of the same name set
before `open-webui serve`:

| Valve | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is reachable |
| `TEXT_MODEL` | `dolphin3:8b` | Writes the final prompt |
| `VISION_MODEL` | `llava:13b` | Describes your attachments |
| `FRAME_COUNT` | `4` | Frames sampled per clip. Plus the final frame that's 5 images — read the vision-budget limit below before raising it |
| `MAX_VISION_IMAGES` | `5` | Hard cap on images sent to the vision model; extras are subsampled rather than erroring |
| `VISION_NUM_CTX` | `8192` | Context for the vision call. Improves its reasoning, but does **not** raise the image budget |
| `REFERENCE_FRAME_DIR` | `~/PromptHub/output` | Where reference frames are saved |
| `REQUEST_TIMEOUT_SECONDS` | `300` | Per-call timeout to Ollama |

---

## 6. Troubleshooting

**First reply after a while is slow.** Normal — Ollama loads the model into
memory on first use, then it's fast until it idles out. A clip is slower
still: frame extraction plus a vision pass before the writing pass.

**"You attached a file, but it couldn't be read."** The attachment didn't
resolve to something readable. Re-encode to mp4, or just describe it in
text. It deliberately refuses rather than quietly answering from your text
alone and pretending the file was used.

**Wrong MiniMax mode.** Say the mode in your message ("use I2VA") — an
explicit mention always wins.

**The alignment line looks malformed** (wrong duration, missing, stray
placeholder text). `dolphin3:8b` is an 8B model hitting an exacting
template; regenerate. If it's persistent, a larger `TEXT_MODEL` helps.

**The description doesn't match my clip.** `llava:13b` is a single-image
model, so it reads scenes well but is weak on motion across frames. See
limitations.

**Can't select `dolphin3:8b` / `llava:13b` in the picker.** Deliberate —
the Ollama connection is switched off inside Open WebUI so the picker holds
only PromptHub's entries. Ollama itself is untouched. Re-enable it in Admin
Panel → Settings → Connections if you want it back (`setup/start.sh` will
switch it off again).

**Forgot the admin password.** It's in
`~/PromptHub/.admin_credentials.json`.

**The picker is empty — no Krea2 or MiniMax H3.** Seeding never got that
far, usually because an admin account already existed (see the warning in
§1) so the installer couldn't sign in to deploy them. Either point it at the
account you already have:

```powershell
$env:PROMPTHUB_ADMIN_EMAIL = "you@example.com"
$env:PROMPTHUB_ADMIN_PASSWORD = "your-password"
.\setup\start.ps1
```

or start clean — stop it, delete the data, re-run the installer, and this
time don't open the browser until it finishes:

```powershell
.\setup\stop.ps1
Remove-Item -Recurse -Force "$HOME\PromptHub"
.\setup\install_windows.ps1
```

(macOS/Linux: `./setup/stop.sh`, `rm -rf ~/PromptHub`, then
`./setup/install_<os>.sh`. On Windows, if the delete reports the folder is
in use, close any shell or Explorer window sitting inside it and retry.)

---

## 7. How it works

```
You ──▶ Open WebUI (chat UI, :8080)
          └─ Pipe Function ──▶ ffmpeg      (frame extraction, if a clip)
                           ──▶ llava:13b   (describe the attachment)
                           ──▶ dolphin3:8b (write the final prompt)
```

Everything is local; the Functions talk to Ollama over `localhost:11434`.

```
models/      the 2 Pipe Functions — each self-contained, prompt embedded inline
setup/       per-OS install + start/stop/seed/verify scripts
```

`setup/seed.py` deploys everything through Open WebUI's REST API on every
start: both Functions, removal of entries from earlier layouts, the Ollama
connection switched off, and alphabetical picker order.

**Why Pipe Functions and not Open WebUI "Model presets"?** A preset resolves
its base model through Open WebUI's connection layer, so `dolphin3:8b` had
to stay visible in the picker for the presets to work — three separate
attempts to hide it broke every preset instead. A Pipe Function POSTs to
Ollama directly, so nothing depends on the connection being listed and it
can be switched off entirely.

**Deviations from `requirements.md`** (the original spec): no Docker and no
separate Pipelines container (Open WebUI's docs now mark that project
legacy, and an in-process Function gets a direct on-disk path to uploads);
setup is scripted through the API rather than clicked through the UI; and
the four MiniMax modes are inferred rather than being separate entries.

---

## 8. Known limitations

- **Motion description is weak.** `llava:13b` is trained on single images,
  so across sampled frames it describes the scene well but often misses the
  movement. A video-capable VLM (e.g. `qwen2.5vl`) is the upgrade path —
  swap `VISION_MODEL`.
- **The vision model can only see ~5 images.** `llava:13b` allows ~4096
  vision tokens at ~576 per image, so 9 images fails outright with
  `exceed_context_size_error`. (The original spec's `FRAME_COUNT = 8` could
  never have worked.) Raising `VISION_NUM_CTX` does **not** lift this —
  only sending fewer images does.
- **Text-model choice is deliberately open.** `requirements.md` §5 suggests
  comparing Stheno/Magnum/EVA-Qwen fine-tunes against Dolphin for writing
  quality. Swap `TEXT_MODEL` to try one.
- **One outbound network call at first launch.** Open WebUI eagerly fetches
  its default RAG embedding model (`all-MiniLM-L6-v2`) from Hugging Face,
  once, then caches it. This project never uses RAG; it's just an eager
  default. Everything after that is local.
- **Attachments must be on local storage.** Reference frames are read from
  Open WebUI's own uploads directory; an S3/remote storage backend isn't
  supported.
- **The admin account uses a generated password**, stored in
  `~/PromptHub/.admin_credentials.json` (chmod 600, never in the repo).
  Treat it like any local secret; set `PROMPTHUB_ADMIN_PASSWORD` yourself
  if the machine is shared.
- **Setup uses Open WebUI's internal API**, which carries no stability
  guarantee. `setup/seed.py` fails loudly with the exact HTTP status if a
  future release changes it — the fallback is pasting `models/*.py` into
  Admin Panel → Functions by hand.
- **Prompt conventions may drift.** The MiniMax prompts follow MiniMax's
  own [H3 prompt-writing guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt)
  (field names, alignment lines, camera vocabulary, speaker tags). The
  Krea 2 conventions came from `requirements.md` and weren't independently
  re-verified — worth spot-checking against current Krea docs.
- `requirements.md` names `dolphin3.0-llama3.1:8b`, which doesn't exist on
  Ollama's registry. The real tag for that model is `dolphin3:8b`.

## 9. Non-goals

No automatic hand-off of reference frames into a generation tool (you take
them from the printed path), no frame-sampling UI (edit `FRAME_COUNT`), no
fixed "best" model choice, no code outside the two Function scripts, and no
targets beyond Krea 2 and MiniMax H3.

---

## 10. License

MIT — see [LICENSE](LICENSE). The models this drives (`dolphin3:8b`,
`llava:13b`) and Open WebUI carry their own separate licenses.
