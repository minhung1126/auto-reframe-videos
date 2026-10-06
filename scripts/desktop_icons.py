"""Generate original application icons without adding artwork dependencies."""
import struct
import zlib
from pathlib import Path


def png(size):
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            # Three layers: caption, central video with play triangle, caption.
            inside = size // 5 <= x < size * 4 // 5 and size // 8 <= y < size * 7 // 8
            video = inside and size * 3 // 8 <= y < size * 5 // 8
            play = video and size * 7 // 16 <= x < size * 10 // 16 and abs(y - size // 2) < (size * 10 // 16 - x) // 2
            color = (245, 248, 255, 255) if play else (38, 166, 190, 255) if video else (22, 33, 52, 255) if inside else (10, 15, 25, 255)
            rows.extend(color)
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')


def write_icons(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = png(256)
    (directory / 'app.ico').write_bytes(struct.pack('<HHH', 0, 1, 1) + struct.pack('<BBBBHHII', 0, 0, 0, 0, 1, 32, len(payload), 22) + payload)
    chunks = b''.join(kind + struct.pack('>I', len(data) + 8) + data for kind, data in [(b'ic08', payload), (b'ic09', png(512))])
    (directory / 'app.icns').write_bytes(b'icns' + struct.pack('>I', len(chunks) + 8) + chunks)
