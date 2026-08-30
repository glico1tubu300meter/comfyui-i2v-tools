r"""
ComfyUI経由で、画像の一部(マスクで黒く塗った領域)を保護しつつ、それ以外(白い領域)だけを
テキストプロンプトに従ってimg2imgで変化させる。

被写体は一切変えず、背景・エフェクトだけを差し替えた「誘導画像」のバリエーションを作る用途を想定。
(例: image-to-videoのマルチキーフレーム誘導で使う中間状態の画像を安全に増やす)

保護は二重に行う:
  1. VAEEncodeForInpaintのnoise_maskで、拡散過程そのものをマスク領域にはあまり効かせない
  2. 生成後、ImageCompositeMaskedで元画像のピクセルをマスク領域にそのまま上書き(最終的な安全策)

事前にComfyUIサーバーが起動している必要がある。マスクは make_protect_mask.py で作成できる。

使い方:
    python generate_masked_variant.py \
        --image path/to/source.png \
        --mask path/to/protect_mask.png \
        --prompt "more magical sparkles swirling brighter, intense golden glow" \
        --checkpoint Realistic_Vision_V5.1_fp16-no-ema.safetensors \
        --denoise 0.8 \
        --out-dir ./out
"""
import argparse
import json
import os
import shutil
import time
import urllib.request

DEFAULT_NEGATIVE = "low quality, worst quality, blurry, deformed, extra limbs, watermark, text, jpeg artifacts"


def ensure_input_file(comfyui_root: str, path: str) -> str:
    input_dir = os.path.join(comfyui_root, "input")
    if os.path.dirname(path) == "":
        return path
    fname = os.path.basename(path)
    dst = os.path.join(input_dir, fname)
    if os.path.abspath(path) != os.path.abspath(dst):
        shutil.copy2(path, dst)
    return fname


def build_workflow(image_filename, mask_filename, prompt, negative, checkpoint, denoise, steps, cfg, seed, filename_prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "3": {"class_type": "LoadImageMask", "inputs": {"image": mask_filename, "channel": "red"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "6": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {"pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 0], "grow_mask_by": 6},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "dpmpp_2m",
                "scheduler": "karras", "denoise": denoise,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {
            "class_type": "ImageCompositeMasked",
            "inputs": {"destination": ["2", 0], "source": ["8", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["3", 0]},
        },
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": filename_prefix}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="入力画像。絶対パス、またはComfyUI/input内のファイル名")
    ap.add_argument("--mask", required=True, help="保護マスク画像(白=変化させる, 黒=保護)。make_protect_mask.pyで作成可")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--checkpoint", required=True, help="ComfyUI/models/checkpoints内のSD系チェックポイント名")
    ap.add_argument("--denoise", type=float, default=0.8, help="値が大きいほど元画像から大きく変化する(0〜1)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--filename-prefix", default="masked_variant")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--comfyui-root", default=r"H:\HeavyData\ComfyUI")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    args = ap.parse_args()

    seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(6), "big")
    image_filename = ensure_input_file(args.comfyui_root, args.image)
    mask_filename = ensure_input_file(args.comfyui_root, args.mask)

    workflow = build_workflow(
        image_filename, mask_filename, args.prompt, args.negative, args.checkpoint,
        args.denoise, args.steps, args.cfg, seed, args.filename_prefix,
    )

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
        time.sleep(2)


if __name__ == "__main__":
    main()
