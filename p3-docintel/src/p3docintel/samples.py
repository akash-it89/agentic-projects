"""Invented sample documents with gold labels, for testing before you have real ones.

Every sample is clearly marked SPECIMEN and uses made-up organisations, people and numbers.
  utility_bill.pdf        born-digital (has a text layer: no OCR needed)
  utility_bill_scan.png   the same bill as a noisy scan (exercises the OCR path)
  lease_agreement.pdf     two pages of running text (exercises summary and chat)
  licence_card.png        a licence-style card photo (OCR on a small, dense layout)

Gold boxes are computed from where the text was actually drawn, normalized to 0..1 like the pipeline's.
Tamil and Arabic samples aren't generated: Pillow can't shape those scripts without extra libraries, so use
real documents for them (see the README).
"""

import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SPECIMEN = "SPECIMEN - invented data for testing, not a real document"


def _gold_box(page: pymupdf.Page, value: str) -> list[float]:
    hits = page.search_for(value)
    if not hits:
        raise ValueError(f"{value!r} not found on page")
    r = hits[0]
    w, h = page.rect.width, page.rect.height
    return [round(r.x0 / w, 5), round(r.y0 / h, 5), round(r.x1 / w, 5), round(r.y1 / h, 5)]


BILL_FIELDS = [
    ("account_number", "ACC-4471-0092"), ("customer_name", "Meera Raghavan"),
    ("service_address", "14 Lake View Road, Riverton 600042"), ("bill_date", "05/10/2026"),
    ("due_date", "25/10/2026"), ("meter_number", "MTR-889120"), ("electricity_used", "412 kWh"),
    ("amount_due", "USD 87.40"),
]


def make_bill(path: Path) -> dict:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 in points
    page.insert_text((50, 62), "SAMPLE POWER & WATER CO.", fontsize=18, fontname="hebo")
    page.insert_text((50, 80), SPECIMEN, fontsize=8, fontname="helv")
    page.insert_text((50, 120), "Monthly statement", fontsize=13, fontname="hebo")
    y = 150
    labels = {"account_number": "Account number", "customer_name": "Customer name",
              "service_address": "Service address", "bill_date": "Bill date", "due_date": "Due date",
              "meter_number": "Meter number", "electricity_used": "Electricity used", "amount_due": "Amount due"}
    for name, value in BILL_FIELDS:
        page.insert_text((50, y), f"{labels[name]}:", fontsize=11, fontname="helv")
        page.insert_text((190, y), value, fontsize=11, fontname="hebo" if name == "amount_due" else "helv")
        y += 22
    page.insert_textbox(pymupdf.Rect(50, y + 20, 545, y + 120),
                        "Please pay by the due date to avoid a late fee of USD 5.00. You can pay online through "
                        "the customer portal or in person at any branch office. Questions about this bill: call "
                        "the customer line on 1800 555 0199.", fontsize=10, fontname="helv")
    gold = [{"name": n, "value": v, "page": 1, "bbox": _gold_box(page, v)} for n, v in BILL_FIELDS]
    doc.save(path)
    doc.close()
    return {"file": path.name, "doc_type": "utility_bill", "language": "en", "fields": gold,
            "questions": [{"question": "When is the payment due?", "expected": "25/10/2026"},
                          {"question": "How much is the late fee?", "expected": "5.00"},
                          {"question": "What is the customer's date of birth?", "expected": None}]}


def make_scan(pdf: Path, out: Path, dpi: int = 150, seed: int = 3) -> None:
    """The bill as a slightly blurred, grainy greyscale scan. No rotation, so the gold boxes still apply."""
    with pymupdf.open(pdf) as doc:
        pix = doc[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    arr = np.asarray(img, dtype=np.float32) * 0.92 + np.random.default_rng(seed).normal(8, 10, (img.height, img.width))
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(out)


LEASE_FIELDS_P1 = [("landlord_name", "Harbor Homes Ltd"), ("tenant_name", "Daniel Okafor"),
                   ("property_address", "Flat 3B, 22 Mill Street, Riverton 600017"), ("monthly_rent", "USD 1,250"),
                   ("security_deposit", "USD 2,500"), ("start_date", "1 November 2026"), ("lease_term", "12 months")]
LEASE_FIELDS_P2 = [("notice_period", "60 days")]


def make_lease(path: Path) -> dict:
    doc = pymupdf.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 62), "RESIDENTIAL LEASE AGREEMENT", fontsize=17, fontname="hebo")
    p1.insert_text((50, 80), SPECIMEN, fontsize=8, fontname="helv")
    y = 120
    labels = {"landlord_name": "Landlord", "tenant_name": "Tenant", "property_address": "Property",
              "monthly_rent": "Monthly rent", "security_deposit": "Security deposit", "start_date": "Start date",
              "lease_term": "Term"}
    for name, value in LEASE_FIELDS_P1:
        p1.insert_text((50, y), f"{labels[name]}:", fontsize=11, fontname="helv")
        p1.insert_text((170, y), value, fontsize=11, fontname="helv")
        y += 22
    p1.insert_textbox(pymupdf.Rect(50, y + 20, 545, 800),
                      "1. Rent. The Tenant pays the monthly rent in advance on the first day of each month by bank "
                      "transfer to the account named by the Landlord.\n\n"
                      "2. Deposit. The security deposit is held for the length of the lease and returned within 30 "
                      "days of the end of the lease, less the cost of repairing any damage beyond normal wear.\n\n"
                      "3. Use. The property is to be used only as a private home for the Tenant and immediate "
                      "family. Subletting is not allowed without the Landlord's written consent.",
                      fontsize=10.5, fontname="helv")
    # Read page 1's positions now: PyMuPDF invalidates earlier Page objects once a new page is added.
    gold = [{"name": n, "value": v, "page": 1, "bbox": _gold_box(p1, v)} for n, v in LEASE_FIELDS_P1]
    p2 = doc.new_page(width=595, height=842)
    p2.insert_textbox(pymupdf.Rect(50, 60, 545, 800),
                      "4. Maintenance. The Landlord repairs the roof, plumbing, wiring and heating. The Tenant keeps "
                      "the property clean and reports any damage within 7 days.\n\n"
                      "5. Pets. No pets are allowed in the property.\n\n"
                      "6. Ending the lease. Either party may end this lease after the first six months by giving "
                      "60 days written notice.\n\n"
                      "7. Signatures. Signed by both parties on 20 October 2026.",
                      fontsize=10.5, fontname="helv")
    gold += [{"name": n, "value": v, "page": 2, "bbox": _gold_box(p2, v)} for n, v in LEASE_FIELDS_P2]
    doc.save(path)
    doc.close()
    return {"file": path.name, "doc_type": "lease_agreement", "language": "en", "fields": gold,
            "questions": [{"question": "Are pets allowed?", "expected": "No pets"},
                          {"question": "How much notice is needed to end the lease?", "expected": "60 days"},
                          {"question": "Who pays for electricity?", "expected": None}]}


LICENCE_FIELDS = [("licence_number", "SS-2026-118734"), ("full_name", "ARJUN MEHTA"), ("date_of_birth", "14/03/1994"),
                  ("issue_date", "02/01/2024"), ("expiry_date", "01/01/2034"), ("vehicle_class", "B"),
                  ("address", "7 Hill Street, Lakeside")]


def make_licence(path: Path) -> dict:
    w, h = 1012, 638  # a bank-card shape
    img = Image.new("RGB", (w, h), (226, 236, 244))
    d = ImageDraw.Draw(img)
    big, mid, small = (ImageFont.load_default(size=s) for s in (40, 28, 20))
    d.rectangle([0, 0, w, 90], fill=(30, 70, 120))
    d.text((30, 22), "SAMPLE STATE  DRIVING LICENCE", font=big, fill=(255, 255, 255))
    d.rectangle([30, 120, 250, 400], outline=(90, 110, 130), width=3)
    d.text((80, 250), "PHOTO", font=mid, fill=(120, 135, 150))
    labels = {"licence_number": "DL No", "full_name": "Name", "date_of_birth": "DOB", "issue_date": "Issued",
              "expiry_date": "Expires", "vehicle_class": "Class", "address": "Address"}
    gold, y = [], 125
    for name, value in LICENCE_FIELDS:
        d.text((290, y), f"{labels[name]}", font=small, fill=(80, 95, 110))
        x0, y0, x1, y1 = d.textbbox((450, y - 4), value, font=mid)
        d.text((450, y - 4), value, font=mid, fill=(15, 20, 30))
        gold.append({"name": name, "value": value, "page": 1,
                     "bbox": [round(x0 / w, 5), round(y0 / h, 5), round(x1 / w, 5), round(y1 / h, 5)]})
        y += 52
    d.rectangle([0, h - 60, w, h], fill=(170, 40, 40))
    d.text((30, h - 46), SPECIMEN.upper(), font=small, fill=(255, 255, 255))
    img.save(path)
    return {"file": path.name, "doc_type": "driving_licence", "language": "en", "fields": gold,
            "questions": [{"question": "When does the licence expire?", "expected": "01/01/2034"}]}


def generate(out_dir: Path) -> list[Path]:
    docs, gold_dir = out_dir / "docs", out_dir / "gold"
    docs.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    bill = make_bill(docs / "utility_bill.pdf")
    make_scan(docs / "utility_bill.pdf", docs / "utility_bill_scan.png")
    scan = {**bill, "file": "utility_bill_scan.png"}
    written = []
    for g in (bill, scan, make_lease(docs / "lease_agreement.pdf"), make_licence(docs / "licence_card.png")):
        p = gold_dir / f"{Path(g['file']).stem}.json"
        p.write_text(json.dumps(g, indent=1, ensure_ascii=False), encoding="utf-8")
        written.append(docs / g["file"])
    return written
