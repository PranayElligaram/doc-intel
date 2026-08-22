"""Regenerates the two PDF fixtures.

The fixtures are committed, so this script is not run by the test suite. It
exists so the bytes in tests/fixtures/*.pdf are auditable rather than magic:
run it and the files should come back byte-identical.

Both PDFs are written by hand rather than by a PDF library on purpose. A
library would produce a realistic file with a compressed xref stream and object
streams, which is fine for a reader but useless as a fixture - you could not
read it to see why a text-layer measurement came out the way it did. These are
small enough to read end to end.

text_layer.pdf has three pages carrying real Tj text operators, with
deliberately unequal amounts of text so that the median and the mean of
chars-per-page differ. A fixture with equal pages would pass whether
analyse_pdf computed the median or the mean.

scanned.pdf has one page whose only content is an image XObject and no text
operator anywhere. That is the shape of a scanned filing: pypdf opens it
happily and extracts nothing.
"""

from pathlib import Path

# Page three is much longer than the other two. Median is ~len(page two);
# the mean is dragged well above it. See the module docstring.
PAGE_TEXTS = [
    "Annual report and financial statements",
    "Registered number 00000000 - Statement of financial position",
    (
        "Turnover for the year ended 31 December 2024 amounted to 4,200 thousand pounds "
        "compared with 3,850 thousand pounds in the preceding period. Profit before "
        "taxation was 512 thousand pounds. The average monthly number of employees "
        "during the year was 74. The auditor is a limited liability partnership "
        "appointed under section 485 of the Companies Act 2006."
    ),
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _assemble(objects: list[bytes], root: int) -> bytes:
    """Concatenate numbered objects and append a correct xref table.

    pypdf can recover from a wrong xref, but it warns when it does, and the
    suite runs with filterwarnings = error. So the offsets have to be right,
    which is the one part of a PDF that is genuinely tedious to do by hand.
    """
    out = bytearray(b"%PDF-1.7\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number
        out += body
        out += b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\n" % (len(objects) + 1, root)
    out += b"startxref\n%d\n%%%%EOF\n" % xref_at
    return bytes(out)


def _stream(body: bytes, extra: str = "") -> bytes:
    return b"<< /Length %d %s>>\nstream\n%s\nendstream" % (
        len(body),
        extra.encode("ascii"),
        body,
    )


def build_text_layer_pdf() -> bytes:
    page_count = len(PAGE_TEXTS)
    # Object numbering: 1 catalog, 2 pages tree, 3 font, then for each page a
    # page object followed by its content stream.
    first_page_obj = 4
    kids = " ".join(f"{first_page_obj + 2 * i} 0 R" for i in range(page_count))

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids.encode("ascii"), page_count),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(PAGE_TEXTS):
        contents_obj = first_page_obj + 2 * i + 1
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % contents_obj
        )
        drawn = f"BT /F1 11 Tf 72 720 Td ({_escape(text)}) Tj ET\n"
        objects.append(_stream(drawn.encode("ascii")))

    return _assemble(objects, root=1)


def build_scanned_pdf() -> bytes:
    # A 2x2 8-bit greyscale image, stored uncompressed so the bytes below are
    # the actual pixels rather than a zlib blob.
    pixels = bytes([0x20, 0x40, 0x60, 0x80])
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
        _stream(b"q 612 0 0 792 0 0 cm /Im0 Do Q\n"),
        _stream(
            pixels,
            "/Type /XObject /Subtype /Image /Width 2 /Height 2 "
            "/ColorSpace /DeviceGray /BitsPerComponent 8 ",
        ),
    ]
    return _assemble(objects, root=1)


if __name__ == "__main__":
    here = Path(__file__).parent
    (here / "text_layer.pdf").write_bytes(build_text_layer_pdf())
    (here / "scanned.pdf").write_bytes(build_scanned_pdf())
    print(f"wrote {here / 'text_layer.pdf'} and {here / 'scanned.pdf'}")
