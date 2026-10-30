import os
import glob
from pathlib import Path
import rawpy
from PIL import Image
import exifread

INPUT_DIR = Path("./raw_photos")
OUTPUT_IMG_DIR = Path("./public/photos")
CONTENT_DIR = Path("./src/content/photos")

OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
CONTENT_DIR.mkdir(parents=True, exist_ok=True)

def parse_exif(file_path):
    with open(file_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    shutter = str(tags.get('EXIF ExposureTime', '1/250s'))
    aperture = f"ƒ/{float(tags.get('EXIF FNumber', 2.8).values[0]):.1f}" if 'EXIF FNumber' in tags else "ƒ/2.8"
    iso = f"ISO {tags.get('EXIF ISOSpeedRatings', '100')}"
    focal = f"{tags.get('EXIF FocalLength', '35')}mm"
    lens = str(tags.get('EXIF LensModel', 'Sony E-Mount Lens')).strip()

    return {
        "shutter": shutter,
        "aperture": aperture,
        "iso": iso,
        "focal": focal,
        "lens": lens,
    }

def convert_arw():
    arw_files = list(INPUT_DIR.glob("*.ARW")) + list(INPUT_DIR.glob("*.arw"))
    if not arw_files:
        print(f"No .ARW files found in {INPUT_DIR}. Create the folder and drop some files there!")
        return

    print(f"Found {len(arw_files)} ARW file(s). Processing...")

    for arw_path in arw_files:
        stem = arw_path.stem
        out_img_path = OUTPUT_IMG_DIR / f"{stem}.webp"
        md_path = CONTENT_DIR / f"{stem}.md"

        print(f"→ Processing: {arw_path.name}")

        # 1. Extract real EXIF metadata from the RAW file
        meta = parse_exif(arw_path)

        # 2. Develop the RAW sensor image into web-ready WebP
        if not out_img_path.exists():
            with rawpy.imread(str(arw_path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=8
                )
            img = Image.fromarray(rgb)
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            img.save(out_img_path, format="WEBP", quality=88)
            print(f"   ✓ Developed image: {out_img_path}")

        # 3. Create or update Markdown entry for Astro
        if not md_path.exists():
            clean_title = stem.replace("-", " ").replace("_", " ").title()
            frontmatter = f"""---
title: "{clean_title}"
caption: "Optical capture with Sony ZV-E10."
image: "/photos/{stem}.webp"
date: 2026-04-20
camera: "Sony ZV-E10"
lens: "{meta['lens']}"
focal_length: "{meta['focal']}"
aperture: "{meta['aperture']}"
shutter: "{meta['shutter']}"
iso: "{meta['iso']}"
---
"""
            with open(md_path, "w") as f:
                f.write(frontmatter)
            print(f"   ✓ Created markdown: {md_path}")

    print("\nAll ARW files processed! Check your photography page.")

if __name__ == "__main__":
    convert_arw()