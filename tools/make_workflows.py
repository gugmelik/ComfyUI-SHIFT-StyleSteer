"""Generates the ComfyUI (UI-format) workflows in ../workflows. Run: python tools/make_workflows.py"""

import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_workflows")

UNET = "flux-2-klein-4b.safetensors"
TE = "qwen_3_4b.safetensors"
VAE = "flux2-vae.safetensors"
REF = "monet_water_lilies.jpg"
PROMPTS = """a red fox in the snow
a cat sitting on a windowsill
a horse in a meadow
an owl on a branch
a sea turtle swimming
a lighthouse on a rocky coast
a bowl of fruit on a wooden table
a city street at dusk
a mountain lake at sunrise
a steam train crossing a bridge
an old man reading a newspaper
a sailboat on a calm sea
a castle on a hill
a dog running on a beach
a vase of sunflowers
a market square with people"""
TEST_PROMPT = "a tiger in the jungle"

# name -> (inputs [(name, type)], outputs [(name, type)])
SPECS = {
    "UNETLoader": ([], [("MODEL", "MODEL")]),
    "CLIPLoader": ([], [("CLIP", "CLIP")]),
    "VAELoader": ([], [("VAE", "VAE")]),
    "LoadImage": ([], [("IMAGE", "IMAGE"), ("MASK", "MASK")]),
    "CLIPTextEncode": ([("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")]),
    "ConditioningZeroOut": ([("conditioning", "CONDITIONING")], [("CONDITIONING", "CONDITIONING")]),
    "CFGGuider": ([("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")], [("GUIDER", "GUIDER")]),
    "KSamplerSelect": ([], [("SAMPLER", "SAMPLER")]),
    "Flux2Scheduler": ([], [("SIGMAS", "SIGMAS")]),
    "RandomNoise": ([], [("NOISE", "NOISE")]),
    "EmptyFlux2LatentImage": ([], [("LATENT", "LATENT")]),
    "SamplerCustomAdvanced": ([("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
                               ("latent_image", "LATENT")], [("output", "LATENT"), ("denoised_output", "LATENT")]),
    "VAEDecode": ([("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")]),
    "SaveImage": ([("images", "IMAGE")], []),
    "PreviewImage": ([("images", "IMAGE")], []),
    "MarkdownNote": ([], []),
    "SHIFTExtractStyleVector": ([("model", "MODEL"), ("clip", "CLIP"), ("vae", "VAE"), ("reference_image", "IMAGE")],
                                [("shift_vector", "SHIFT_VECTOR"), ("teacher_i2i", "IMAGE"), ("student_t2i", "IMAGE")]),
    "SHIFTApplyStyleVector": ([("model", "MODEL"), ("shift_vector", "SHIFT_VECTOR")], [("MODEL", "MODEL")]),
    "SHIFTSaveStyleVector": ([("shift_vector", "SHIFT_VECTOR")], []),
    "SHIFTLoadStyleVector": ([], [("shift_vector", "SHIFT_VECTOR")]),
}


class Graph:
    def __init__(self):
        self.nodes, self.links = [], []

    def add(self, type_, pos, widgets=None, size=(320, 120), title=None, color=None):
        ins, outs = SPECS[type_]
        node = {
            "id": len(self.nodes) + 1, "type": type_, "pos": list(pos), "size": list(size), "flags": {},
            "order": len(self.nodes), "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in ins],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i} for i, (n, t) in enumerate(outs)],
            "properties": {"Node name for S&R": type_},
            "widgets_values": widgets or [],
        }
        if title:
            node["title"] = title
        if color:
            node["color"], node["bgcolor"] = color
        self.nodes.append(node)
        return node

    def link(self, src, out_name, dst, in_name):
        o = next(i for i, x in enumerate(src["outputs"]) if x["name"] == out_name)
        i = next(i for i, x in enumerate(dst["inputs"]) if x["name"] == in_name)
        lid = len(self.links) + 1
        typ = src["outputs"][o]["type"]
        self.links.append([lid, src["id"], o, dst["id"], i, typ])
        src["outputs"][o]["links"].append(lid)
        dst["inputs"][i]["link"] = lid

    def dump(self, path, title):
        data = {"last_node_id": len(self.nodes), "last_link_id": len(self.links), "nodes": self.nodes,
                "links": self.links, "groups": [], "config": {}, "extra": {"title": title}, "version": 0.4}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("wrote", path)


PURPLE = ("#323", "#535")


MODEL_URLS = {
    UNET: ("diffusion_models", "https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors"),
    TE: ("text_encoders", "https://huggingface.co/Comfy-Org/flux2-klein/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"),
    VAE: ("vae", "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"),
}


def with_model(node, name):
    directory, url = MODEL_URLS[name]
    node["properties"]["models"] = [{"name": name, "url": url, "directory": directory}]
    return node


def loaders(g, y=0):
    unet = with_model(g.add("UNETLoader", (0, y), [UNET, "default"], (330, 90)), UNET)
    clip = with_model(g.add("CLIPLoader", (0, y + 130), [TE, "flux2", "default"], (330, 110)), TE)
    vae = with_model(g.add("VAELoader", (0, y + 280), [VAE], (330, 60)), VAE)
    return unet, clip, vae


def sampling_branch(g, x, y, model, clip, vae, prompt, seed, prefix, title):
    pos = g.add("CLIPTextEncode", (x, y), [prompt], (360, 110), title=f"Prompt ({title})")
    neg = g.add("ConditioningZeroOut", (x, y + 150), [], (240, 30))
    guider = g.add("CFGGuider", (x + 400, y), [1.0], (240, 100))
    sampler = g.add("KSamplerSelect", (x + 400, y + 140), ["euler"], (240, 60))
    sigmas = g.add("Flux2Scheduler", (x + 400, y + 240), [4, 1024, 1024], (240, 110))
    noise = g.add("RandomNoise", (x + 400, y + 390), [seed, "fixed"], (240, 85))
    latent = g.add("EmptyFlux2LatentImage", (x + 400, y + 510), [1024, 1024, 1], (240, 110))
    sca = g.add("SamplerCustomAdvanced", (x + 680, y), [], (260, 110))
    dec = g.add("VAEDecode", (x + 680, y + 150), [], (200, 50))
    save = g.add("SaveImage", (x + 980, y), [prefix], (420, 460), title=title)
    g.link(clip, "CLIP", pos, "clip")
    g.link(pos, "CONDITIONING", neg, "conditioning")
    g.link(model, "MODEL", guider, "model")
    g.link(pos, "CONDITIONING", guider, "positive")
    g.link(neg, "CONDITIONING", guider, "negative")
    for n_, out_, in_ in ((noise, "NOISE", "noise"), (guider, "GUIDER", "guider"), (sampler, "SAMPLER", "sampler"),
                          (sigmas, "SIGMAS", "sigmas"), (latent, "LATENT", "latent_image")):
        g.link(n_, out_, sca, in_)
    g.link(sca, "output", dec, "samples")
    g.link(vae, "VAE", dec, "vae")
    g.link(dec, "IMAGE", save, "images")


NOTE_1 = f"""## SHIFT reference-image style steering - FLUX.2 Klein 4B

**1. Extract** (top): every prompt is run twice from the same noise, once with the reference image
(Klein I2I = teacher) and once without (T2I = student). The double-stream block outputs are recorded
at every step, reference tokens are cropped, and the mean difference is the steering field.

**2. Generate** (bottom): plain text-to-image, **no reference image**. The apply node patches the
double blocks: `h <- |h| (h + a v) / |h + a v|` on the image stream.
Left = unsteered, right = steered, same seed.

Tips
- `strength_img` 4-8 is a good start; sweep it. Large values copy the reference *content*.
- Prompts for extraction: content only, no style words.
- The vector is tied to the resolution and step grid used for extraction (1024x1024, 4 steps).
  Other resolutions fall back to the token-pooled vector.
- The vector is saved to `ComfyUI/models/shift_vectors/` - reuse it with workflow 02.

Models: `{UNET}` (diffusion_models), `{TE}` (text_encoders), `{VAE}` (vae).
"""


def workflow_extract_and_generate():
    g = Graph()
    g.add("MarkdownNote", (-460, 0), [NOTE_1], (420, 620))
    unet, clip, vae = loaders(g)
    ref = g.add("LoadImage", (0, 380), [REF, "image"], (330, 360), title="Style reference")
    ext = g.add("SHIFTExtractStyleVector", (380, 0),
                [PROMPTS, 1024, 1024, 4, 1.0, 42000, "fixed", 16, 1.0, True], (420, 560), color=PURPLE)
    save = g.add("SHIFTSaveStyleVector", (840, 0), ["klein4b_monet"], (300, 60), color=PURPLE)
    pv_t = g.add("PreviewImage", (840, 110), [], (380, 380), title="Teacher (prompt + reference)")
    pv_s = g.add("PreviewImage", (1240, 110), [], (380, 380), title="Student (prompt only)")
    for src, out_, in_ in ((unet, "MODEL", "model"), (clip, "CLIP", "clip"), (vae, "VAE", "vae"), (ref, "IMAGE", "reference_image")):
        g.link(src, out_, ext, in_)
    g.link(ext, "shift_vector", save, "shift_vector")
    g.link(ext, "teacher_i2i", pv_t, "images")
    g.link(ext, "student_t2i", pv_s, "images")

    app = g.add("SHIFTApplyStyleVector", (0, 820),
                ["renorm_add", 6.0, 10.0, 0.0, "all", "all", "per_token", "cond_and_uncond"], (330, 260), color=PURPLE)
    g.link(unet, "MODEL", app, "model")
    g.link(ext, "shift_vector", app, "shift_vector")
    sampling_branch(g, 380, 820, unet, clip, vae, TEST_PROMPT, 7, "shift_klein/unsteered", "Unsteered")
    sampling_branch(g, 380, 1500, app, clip, vae, TEST_PROMPT, 7, "shift_klein/steered", "Steered (SHIFT)")
    g.dump(os.path.join(OUT, "01_shift_klein4b_extract_and_generate.json"), "SHIFT Klein 4B - extract + generate")


def workflow_generate_from_saved():
    g = Graph()
    unet, clip, vae = loaders(g)
    load = g.add("SHIFTLoadStyleVector", (0, 380), ["klein4b_monet_00001.safetensors"], (330, 60), color=PURPLE)
    app = g.add("SHIFTApplyStyleVector", (0, 480),
                ["renorm_add", 6.0, 10.0, 0.0, "all", "all", "per_token", "cond_and_uncond"], (330, 260), color=PURPLE)
    g.link(unet, "MODEL", app, "model")
    g.link(load, "shift_vector", app, "shift_vector")
    sampling_branch(g, 380, 0, unet, clip, vae, TEST_PROMPT, 7, "shift_klein/unsteered", "Unsteered")
    sampling_branch(g, 380, 680, app, clip, vae, TEST_PROMPT, 7, "shift_klein/steered", "Steered (SHIFT)")
    g.dump(os.path.join(OUT, "02_shift_klein4b_generate_from_saved_vector.json"), "SHIFT Klein 4B - generate from saved vector")


if __name__ == "__main__":
    workflow_extract_and_generate()
    workflow_generate_from_saved()
