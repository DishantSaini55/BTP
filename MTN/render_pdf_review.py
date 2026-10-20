"""Render report review sheets; used only for final visual QA."""
from pathlib import Path
from PIL import Image

folder = Path(r"D:\IDD-PeD-Evaluation\outputs\_final_pdf_review")
pages = sorted(folder.glob("page_*.png"))
for start in range(0, len(pages), 4):
    images = [Image.open(path).convert("RGB") for path in pages[start:start + 4]]
    canvas = Image.new("RGB", (images[0].width * 2, images[0].height * 2), "white")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 2) * image.width, (index // 2) * image.height))
    canvas.save(folder / f"contact_{start // 4 + 1}.png")
print(f"contacts={len(list(folder.glob('contact_*.png')))}")
