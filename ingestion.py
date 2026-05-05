"""
src/ingestion.py
----------------
Parses the BIS SP 21 dataset PDF and extracts one structured chunk per
standard entry. Saves output to data/chunks.json.

Chunking strategy:
  - Hard delimiter : "SUMMARY OF\n" (appears 570 times in the corpus)
  - IS number      : regex-extracted from the header line of each block
  - Title          : remainder of header line(s) after the IS number
  - Scope          : text after "1. Scope —" up to the next numbered section
  - Full text      : entire block (used for embedding)
  - Section        : parent section name carried forward as metadata
"""

import re
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
PDF_PATH    = REPO_ROOT / "data" / "dataset.pdf"
OUTPUT_PATH = REPO_ROOT / "data" / "chunks.json"

# ---------------------------------------------------------------------------
# Section header mapping  (line-prefix → clean section name)
# ---------------------------------------------------------------------------
SECTION_NAMES: dict[int, str] = {
    1:  "Cement and Concrete",
    2:  "Building Limes",
    3:  "Stones",
    4:  "Wood Products for Building",
    5:  "Gypsum Building Materials",
    6:  "Timber",
    7:  "Bitumen and Tar Products",
    8:  "Floor, Wall, Roof Coverings and Finishes",
    9:  "Water Proofing and Damp Proofing Materials",
    10: "Sanitary Appliances and Water Fittings",
    11: "Builder's Hardware",
    12: "Wood Products",
    13: "Doors, Windows and Shutters",
    14: "Concrete Reinforcement",
    15: "Structural Steels",
    16: "Light Metals and Their Alloys",
    17: "Structural Shapes",
    18: "Welding Electrodes and Wires",
    19: "Threaded Fasteners and Rivets",
    20: "Wire Ropes and Wire Products",
    21: "Glass",
    22: "Fillers, Stoppers and Putties",
    23: "Thermal Insulation Materials",
    24: "Plastics",
    25: "Conductors and Cables",
    26: "Wiring Accessories",
    27: "General",
}

# Regex to detect "SECTION N" boundary in the raw text
_SECTION_RE = re.compile(r"^SECTION\s+(\d{1,2})\s*$", re.MULTILINE)

# Regex to extract IS number from header line.
_IS_NUM_RE = re.compile(
    r"^(IS\s+\d+(?:\s*\([^)]+\))?\s*:?\s*\d{4})",
    re.IGNORECASE,
)

# Regex patterns to extract scope — ordered most to least specific
_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(r"1\.\s*Scope\s*[—\-–]+\s*(.+?)(?=\n\s*\n|\n\s*[2-9]\.|\n\s*\d+\s+[A-Z]|$)", re.DOTALL),
    re.compile(r"1\s+Scope\s*\n+\s*1\.1\s+(.+?)(?=\n\s*\n|\n\s*[2-9]\.|\n\s*1\.2|$)", re.DOTALL),
    re.compile(r"Scope\s*[—\-–:]+\s*(.+?)(?=\n\s*\n|\n\s*[2-9]\.|$)", re.DOTALL),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Uses pdftotext (poppler) to extract raw text from the PDF.
    Falls back gracefully to pypdf if poppler is unavailable.
    """
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftotext failed: {result.stderr[:200]}")
        return result.stdout
    except FileNotFoundError:
        print("[ingestion] pdftotext not found — falling back to pypdf (Windows mode)")
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

def normalise_is_number(raw: str) -> str:
    """Normalise an IS number to canonical form."""
    s = re.sub(r"\s*:\s*", ": ", raw.strip())
    s = re.sub(r"\(PART\s*(\d+)\)", lambda m: f"(Part {m.group(1)})", s, flags=re.IGNORECASE)
    s = re.sub(r" {2,}", " ", s)
    return s

def extract_scope(block_text: str) -> str:
    """Extract the scope sentence using multiple patterns to handle pypdf variants."""
    for pattern in _SCOPE_PATTERNS:
        match = pattern.search(block_text)
        if match:
            raw = match.group(1)
            raw = re.sub(r" {4,}.*", "", raw, flags=re.MULTILINE)
            scope = re.sub(r"\s+", " ", raw).strip()
            if len(scope) > 20:
                return scope
    return ""

def extract_title_from_header(header_lines: str) -> str:
    """Extract the human-readable title after the IS number."""
    lines = [l.strip() for l in header_lines.strip().splitlines() if l.strip()]
    if not lines:
        return ""

    first_line = lines[0]
    title_part = _IS_NUM_RE.sub("", first_line).strip()

    continuation = []
    for line in lines[1:]:
        if re.match(r"^\d+\.", line):
            break
        if len(line) > 120:
            break
        continuation.append(line)

    full_title = " ".join(filter(None, [title_part] + continuation))
    full_title = re.sub(r"\b\d+\.\d+\b", "", full_title).strip()
    return full_title

def assign_section(block_position: int, section_boundaries: list[tuple[int, int]]) -> str:
    """Return the parent section name based on character offset."""
    current_section = 1
    for offset, sec_num in section_boundaries:
        if block_position >= offset:
            current_section = sec_num
        else:
            break
    return SECTION_NAMES.get(current_section, "Unknown")

# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------

def parse_chunks(full_text: str) -> list[dict]:
    """Split the corpus on 'SUMMARY OF' and parse each block."""
    section_boundaries: list[tuple[int, int]] = []
    for m in _SECTION_RE.finditer(full_text):
        sec_num = int(m.group(1))
        section_boundaries.append((m.start(), sec_num))
    section_boundaries.sort(key=lambda x: x[0])

    raw_blocks = re.split(r"SUMMARY OF\s*\n", full_text)[1:]

    chunks: list[dict] = []
    skipped = 0

    for idx, block in enumerate(raw_blocks):
        block = block.strip()
        if not block:
            skipped += 1
            continue

        first_line = block.splitlines()[0].strip() if block.splitlines() else ""
        is_match = _IS_NUM_RE.match(first_line)
        if not is_match:
            skipped += 1
            continue

        raw_is_num = is_match.group(1)
        canonical_is_num = normalise_is_number(raw_is_num)
        title = extract_title_from_header(block)
        scope = extract_scope(block)

        block_offset = full_text.find(block[:60])
        section_name = assign_section(block_offset, section_boundaries)

        clean_lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            digit_ratio = sum(c.isdigit() or c in "-.%" for c in stripped) / max(len(stripped), 1)
            if digit_ratio > 0.5:
                continue
            if re.match(r"^\d+\.\d+$", stripped):
                continue
            clean_lines.append(stripped)

        clean_body = " ".join(clean_lines)

        embed_text = f"{canonical_is_num} {title}. {scope} {clean_body[:800]}"
        embed_text = re.sub(r"\s+", " ", embed_text).strip()

        chunks.append({
            "is_number":   canonical_is_num,
            "title":       title,
            "section":     section_name,
            "scope":       scope,
            "embed_text":  embed_text,
            "full_text":   block,   # <-- FIX: Removed [:2000] truncation
        })

    print(f"  Parsed : {len(chunks)} chunks")
    print(f"  Skipped: {skipped} blocks (no IS number found)")
    return chunks

# ---------------------------------------------------------------------------
# Entry point & CLI
# ---------------------------------------------------------------------------

def run_ingestion(pdf_path: Optional[Path] = None, output_path: Optional[Path] = None) -> list[dict]:
    pdf_path    = pdf_path    or PDF_PATH
    output_path = output_path or OUTPUT_PATH

    if not pdf_path.exists():
        raise FileNotFoundError(f"Dataset PDF not found at: {pdf_path}")

    print(f"[ingestion] Extracting text from: {pdf_path.name}")
    full_text = extract_text_from_pdf(pdf_path)
    
    print("[ingestion] Parsing chunks...")
    chunks = parse_chunks(full_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[ingestion] Saved {len(chunks)} chunks → {output_path}")
    return chunks

if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception as e:
        print(f"[ingestion] FATAL: {e}", file=sys.stderr)
        sys.exit(1)