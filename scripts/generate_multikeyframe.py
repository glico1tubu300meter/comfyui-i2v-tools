r"""
ComfyUI + LTX-Video で、複数の画像を「動画内の特定フレーム位置」に誘導点として紐づけ、
1回の連続した拡散生成で長め(数秒〜十数秒)の動画を作る。

前後で独立に短い動画を作ってffmpegで繋ぐ方式より、モデルが全体の文脈を1回で見ながら
ノイズ除去するため、継ぎ目の破綻が少なくなりやすい。

ただし誘導画像そのものの品質が動画の品質上限を決める。誘導に使う画像が崩れていると、
その崩れがそのまま動画に焼き付けられるので、誘導画像は事前に必ず目視確認すること。

事前にComfyUIサーバーが起動している必要がある。

使い方:
    python generate_multikeyframe.py \
        --start-image path/to/start.png \
        --guide "path/to/mid.png:64:1.0" \
        --guide "path/to/climax.png:128:1.0" \
        --guide "path/to/end.png:-1:1.0" \
        --prompt "long descriptive text covering the whole story arc..." \
        --length 193 --width 576 --height 768 \
        --out-dir ./out

--guide は "画像パス:フレーム位置:強度" の形式で複数回指定できる。
  - フレーム位置: 単一フレーム画像の場合は任意の整数でよいが、慣例的に8の倍数を推奨。
    -1 は動画の最終フレームを意味する。
  - 強度: 0.0〜1.0。1.0でその時点をほぼ完全にその画像に固定する。
"""
import argparse
import json
import os
import shutil
import time
import urllib.request

DEFAULT_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, "
    "jpeg artifacts, static image, no motion, morphing face, extra fingers, abrupt cut, flicker"
)


def ensure_input_file(comfyui_root: str, path: str) -> str:
    input_dir = os.path.join(comfyui_root, "input")
    if os.path.dirname(path) == "":
        return path
    fname = os.path.basename(path)
    dst = os.path.join(input_dir, fname)
    if os.path.abspath(path) != os.path.abspath(dst):
        shutil.copy2(path, dst)
    return fname


def parse_guide(s: str):
    path, frame_idx, strength = s.rsplit(":", 2)
    return path, int(frame_idx), float(strength)


def build_workflow(args, start_image_filename, guides, seed):
    wf = {
        "44": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": args.checkpoint}},
        "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.text_encoder, "type": "ltxv"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": args.prompt, "clip": ["38", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": args.negative, "clip": ["38", 0]}},
        "78": {"class_type": "LoadImage", "inputs": {"image": start_image_filename}},
        "77": {
            "class_type": "LTXVImgToVideo",
            "inputs": {
                "positive": ["6", 0], "negative": ["7", 0], "vae": ["44", 2], "image": ["78", 0],
                "width": args.width, "height": args.height, "length": args.length,
                "batch_size": 1, "strength": 1.0,
            },
        },
    }

    prev_pos, prev_neg, prev_latent = ["77", 0], ["77", 1], ["77", 2]
    for i, (path, frame_idx, strength) in enumerate(guides):
        filename = ensure_input_file(args.comfyui_root, path)
        load_id = f"guide_load_{i}"
        pre_id = f"guide_pre_{i}"
        add_id = f"guide_add_{i}"
        wf[load_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
        wf[pre_id] = {"class_type": "LTXVPreprocess", "inputs": {"image": [load_id, 0], "img_compression": 18}}
        wf[add_id] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": prev_pos, "negative": prev_neg, "vae": ["44", 2], "latent": prev_latent,
                "image": [pre_id, 0], "frame_idx": frame_idx, "strength": strength,
            },
        }
        prev_pos, prev_neg, prev_latent = [add_id, 0], [add_id, 1], [add_id, 2]

    wf["crop"] = {
        "class_type": "LTXVCropGuides",
        "inputs": {"positive": prev_pos, "negative": prev_neg, "latent": prev_latent},
    }
    wf["69"] = {
        "class_type": "LTXVConditioning",
        "inputs": {"positive": ["crop", 0], "negative": ["crop", 1], "frame_rate": args.fps},
    }
    wf["71"] = {
        "class_type": "LTXVScheduler",
        "inputs": {
            "steps": args.steps, "max_shift": 2.05, "base_shift": 0.95,
            "stretch": True, "terminal": 0.1, "latent": ["crop", 2],
        },
    }
    wf["73"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    wf["72"] = {
        "class_type": "SamplerCustom",
        "inputs": {
            "model": ["44", 0], "add_noise": True, "noise_seed": seed, "cfg": args.cfg,
            "positive": ["69", 0], "negative": ["69", 1], "sampler": ["73", 0],
            "sigmas": ["71", 0], "latent_image": ["crop", 2],
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["72", 0], "vae": ["44", 2]}}
    wf["41"] = {
        "class_type": "SaveAnimatedWEBP",
        "inputs": {
            "images": ["8", 0], "filename_prefix": args.filename_prefix,
            "fps": float(args.fps), "lossless": False, "quality": 90, "method": "default",
        },
    }
    return wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-image", required=True, help="フレーム0の開始画像")
    ap.add_argument("--guide", action="append", default=[], help="'画像パス:フレーム位置:強度' の形式。複数回指定可")
    ap.add_argument("--prompt", required=True, help="動画全体のストーリーを説明する長めのテキスト")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--length", type=int, default=193, help="総フレーム数(8n+1推奨)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--checkpoint", default="ltx-video-2b-v0.9.5.safetensors")
    ap.add_argument("--text-encoder", default="t5xxl_fp8_e4m3fn_scaled.safetensors")
    ap.add_argument("--filename-prefix", default="ltxv_multikeyframe")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--comfyui-root", default=r"H:\HeavyData\ComfyUI")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(6), "big")
    start_image_filename = ensure_input_file(args.comfyui_root, args.start_image)
    guides = [parse_guide(g) for g in args.guide]

    workflow = build_workflow(args, start_image_filename, guides, seed)

    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(f"{args.server}/prompt", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        prompt_id = json.loads(resp.read())["prompt_id"]
    print("queued:", prompt_id, "seed:", seed)

    while True:
        with urllib.request.urlopen(f"{args.server}/history/{prompt_id}") as resp:
            hist = json.loads(resp.read())
        if prompt_id in hist:
            entry = hist[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                os.makedirs(args.out_dir, exist_ok=True)
                for out in entry.get("outputs", {}).values():
                    for img in out.get("images", []):
                        src = os.path.join(args.comfyui_root, img.get("type", "output"), img.get("subfolder", ""), img["filename"])
                        dst = os.path.join(args.out_dir, img["filename"])
                        shutil.copy2(src, dst)
                        print("copied to:", dst)
                return
            if status.get("status_str") == "error":
                print("ERROR:", json.dumps(entry, indent=2, ensure_ascii=False))
                return
        time.sleep(3)


if __name__ == "__main__":
    main()
