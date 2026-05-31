from pathlib import Path

from gllm.cli import _sniff_mime


def test_png_magic_bytes():
    assert _sniff_mime(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) == "image/png"


def test_jpeg_magic_bytes():
    assert _sniff_mime(b"\xff\xd8\xff\xe0" + b"\x00" * 100) == "image/jpeg"


def test_gif_magic_bytes():
    assert _sniff_mime(b"GIF89a" + b"\x00" * 100) == "image/gif"
    assert _sniff_mime(b"GIF87a" + b"\x00" * 100) == "image/gif"


def test_webp_magic_bytes():
    data = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 100
    assert _sniff_mime(data) == "image/webp"


def test_pdf_magic_bytes():
    assert _sniff_mime(b"%PDF-1.7\n" + b"\x00" * 100) == "application/pdf"


def test_unknown_returns_none_without_path_hint():
    assert _sniff_mime(b"random bytes here, nothing recognizable") is None


def test_extension_fallback_when_bytes_unknown():
    # No magic bytes, but a .png path hint -> mimetypes resolves.
    assert _sniff_mime(b"junk", Path("foo.png")) == "image/png"


def test_magic_bytes_override_extension():
    # File is named .png but the bytes are JPEG — sniffing wins.
    assert _sniff_mime(
        b"\xff\xd8\xff\xe0" + b"\x00" * 100, Path("misnamed.png")
    ) == "image/jpeg"


def test_fixture_image_if_present():
    # The horse infographic, if the fixture exists.
    p = Path(__file__).parent / "test-img.png"
    if not p.exists():
        return
    mime = _sniff_mime(p.read_bytes(), p)
    assert mime in {"image/png", "image/jpeg"}
