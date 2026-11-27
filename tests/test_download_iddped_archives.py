import io
import threading
import tempfile
import tarfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from MTN.download_iddped_archives import (
    CHUNK_BYTES,
    download_archive,
    parse_content_range,
    require_d_drive,
    safe_member_name,
    verify_tar_readable,
)


class DownloadArchiveUtilityTests(unittest.TestCase):
    def test_parse_content_range(self):
        self.assertEqual(parse_content_range("bytes 12-20/100"), (12, 20, 100))
        self.assertIsNone(parse_content_range("bytes */100"))
        self.assertIsNone(parse_content_range(None))

    def test_rejects_unsafe_member_paths(self):
        for name in ("../escape.mp4", "/absolute.mp4", "C:\\outside.mp4", "a\\..\\b.mp4"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_member_name(name)

    def test_reads_tar_members_without_extracting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"synthetic-test-payload")
            archive_path = root / "sample.tar"
            with tarfile.open(archive_path, "w") as archive:
                archive.add(source, arcname="gp_set_0001/sample.mp4")

            members = verify_tar_readable(archive_path)

            self.assertEqual(
                members,
                [{"name": "gp_set_0001/sample.mp4", "size": len(b"synthetic-test-payload"), "type": "file"}],
            )
            self.assertFalse((root / "gp_set_0001").exists())

    def test_rejects_truncated_tar_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"x" * 1024)
            archive_path = root / "sample.tar"
            with tarfile.open(archive_path, "w") as archive:
                archive.add(source, arcname="gp_set_0001/sample.mp4")
            archive_path.write_bytes(archive_path.read_bytes()[:1024])

            with self.assertRaises(tarfile.TarError):
                verify_tar_readable(archive_path)

    def test_requires_d_drive_for_data_paths(self):
        with self.assertRaises(ValueError):
            require_d_drive(Path(r"C:\dataset\archive.tar"))

    def test_download_resumes_after_partial_response(self):
        payload_stream = io.BytesIO()
        with tarfile.open(fileobj=payload_stream, mode="w") as archive:
            content = b"x" * (2 * CHUNK_BYTES)
            member = tarfile.TarInfo("gp_set_0001/sample.mp4")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        payload = payload_stream.getvalue()

        class Handler(BaseHTTPRequestHandler):
            requests = []
            dropped = False

            def log_message(self, _format, *_args):
                del _format, _args
                return

            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", '"fixture-etag"')
                self.end_headers()

            def do_GET(self):
                range_header = self.headers.get("Range", "")
                start = int(range_header.removeprefix("bytes=").split("-", 1)[0])
                if range_header == "bytes=0-0":
                    end = 0
                else:
                    end = len(payload) - 1
                    self.__class__.requests.append(start)
                body = payload[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(body) if range_header == "bytes=0-0" else len(payload) - start))
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", '"fixture-etag"')
                self.end_headers()
                if range_header != "bytes=0-0" and start == 0 and not self.__class__.dropped:
                    self.__class__.dropped = True
                    self.wfile.write(payload[:CHUNK_BYTES])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        with tempfile.TemporaryDirectory(dir=r"D:\IDD-PeD-Evaluation") as temporary:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/gp_set_0001.tar"
                result = download_archive(
                    "0001",
                    Path(temporary),
                    max_bytes=len(payload),
                    allow_unknown_size=False,
                    execute=True,
                    timeout=10,
                    url=url,
                )
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

            self.assertEqual(result["status"], "completed")
            self.assertEqual(Handler.requests[:2], [0, CHUNK_BYTES])
            self.assertEqual(
                (Path(temporary) / "gp_set_0001.tar").stat().st_size,
                len(payload),
            )


if __name__ == "__main__":
    unittest.main()
