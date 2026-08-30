"""
img2imgで「特定領域だけを保護し、それ以外を変化させる」ためのマスク画像を作る。
白(255) = 変化させる領域、黒(0) = 保護する領域(元の内容が維持される)。
境界はガウシアンブラーでぼかし、合成時の継ぎ目を目立たなくする。

使い方(保護したい領域を楕円で複数指定できる):
    python make_protect_mask.py --width 896 --height 1195 \
        --protect-ellipse 150,190,700,800 \
        --protect-ellipse 180,500,760,820 \
        --out ../data/protect_mask.png

--protect-ellipse は "x1,y1,x2,y2"(バウンディングボックス)を複数回指定できる。
座標は対象画像を見ながら手動で調整するのが手っ取り早い(自動セグメンテーションは行わない)。
"""
import argparse
from PIL import Image, ImageDraw, ImageFilter


def parse_box(s: str) -> tuple[int, int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected format: x1,y1,x2,y2")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument(
        "--protect-ellipse", type=parse_box, action="append", default=[],
        help="保護したい楕円領域のバウンディングボックス x1,y1,x2,y2。複数回指定可",
    )
    ap.add_argument(
        "--protect-polygon", type=str, action="append", default=[],
        help="保護したい多角形領域。'x1,y1;x2,y2;x3,y3;...' の形式。複数回指定可",
    )
    ap.add_argument("--feather", type=int, default=25, help="境界のぼかし半径(px)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mask = Image.new("L", (args.width, args.height), 255)
    draw = ImageDraw.Draw(mask)

    for box in args.protect_ellipse:
        draw.ellipse(box, fill=0)

    for poly_str in args.protect_polygon:
        points = [tuple(map(int, p.split(","))) for p in poly_str.split(";")]
        draw.polygon(points, fill=0)

    if args.feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=args.feather))

    mask.save(args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
