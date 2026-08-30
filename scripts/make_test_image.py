"""
実在の人物を含まない、完全に合成のテスト画像を生成する。
image-to-video系スクリプトの動作確認用サンプルとして使う。

使い方:
    python make_test_image.py --out ../data/test_image.png
"""
import argparse
from PIL import Image, ImageDraw


def make_image(width: int, height: int) -> Image.Image:
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # sky-to-ground gradient background
    for y in range(height):
        t = y / height
        r = int(135 * (1 - t) + 30 * t)
        g = int(206 * (1 - t) + 140 * t)
        b = int(235 * (1 - t) + 60 * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # sun
    sun_r = width * 0.06
    draw.ellipse((width * 0.8, height * 0.08, width * 0.8 + sun_r, height * 0.08 + sun_r), fill=(255, 220, 90))

    # simple boat-like shape (SVD-style models tend to animate ripples/waves well)
    bx, by = width * 0.4, height * 0.55
    draw.polygon(
        [(bx, by + 60), (bx + 220, by + 60), (bx + 160, by + 110), (bx + 60, by + 110)],
        fill=(120, 70, 40),
    )
    draw.line([(bx + 80, by - 20), (bx + 80, by + 60)], fill=(80, 60, 40), width=4)

    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/test_image.png")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    args = ap.parse_args()

    img = make_image(args.width, args.height)
    img.save(args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
