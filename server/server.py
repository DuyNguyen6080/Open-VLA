"""TCP server for OpenVLA numpy-array requests.

Protocol used by this server:
- The client sends exactly one NumPy .npy payload produced by np.save(...).
- The server parses the .npy header first, so it knows the exact payload size.
- The server sends exactly one NumPy .npy payload back.

No image-size guessing, no timeout-based receive loop, and no pickle.
"""

from __future__ import annotations

import ast
import io
import socket
import struct
from typing import Optional, Tuple

import numpy as np


_MAGIC = b"\x93NUMPY"
_HEADER_LEN_SIZE = {
    (1, 0): 2,
    (2, 0): 4,
    (3, 0): 4,
}
_MAX_HEADER_BYTES = 1_000_000
_MAX_ARRAY_BYTES = 2_000_000_000  # safety guard: 2 GB


class OpenVLAServer:
    """Small blocking TCP server for OpenVLA image -> action inference.

    The public method names intentionally match your existing main.py:
        server.accept()
        image = server.recvImage(...)
        server.sendImage(action)

    recvImage accepts any valid .npy array. The height/width/color arguments are
    only optional validation hints; they are not used to decide how many bytes to
    read from the socket.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9999, backlog: int = 1):
        self.host = host
        self.port = int(port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(backlog)

        self.conn: Optional[socket.socket] = None
        self.addr: Optional[Tuple[str, int]] = None

        print(f"OpenVLA TCP server listening on {self.host}:{self.port}", flush=True)

    def accept(self):
        """Accept one client connection and store it as the active connection."""
        self.close_client()
        print("Waiting for OpenVLA client...", flush=True)
        self.conn, self.addr = self.server_socket.accept()
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Connected by {self.addr}", flush=True)
        return self.conn, self.addr

    def recv(self, bufsize: int = 65536) -> bytes:
        """Receive raw bytes from the active connection."""
        if self.conn is None:
            raise RuntimeError("No client connected")
        return self.conn.recv(bufsize)

    def send(self, data: bytes) -> None:
        """Send raw bytes to the active connection."""
        if self.conn is None:
            raise RuntimeError("No client connected")
        self.conn.sendall(data)

    def recvImage(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        color: bool = True,
    ) -> Optional[np.ndarray]:
        """Receive one image as a real .npy file from the active connection.

        height, width, and color are kept for compatibility with your main.py.
        They are validation hints only. The exact read size comes from the .npy
        header, so arrays of different dimensions can still be received cleanly.
        """
        try:
            arr = self.recv_numpy()
        except EOFError:
            return None
        except Exception as exc:
            print(f"Error receiving numpy image: {exc}", flush=True)
            return None

        self._warn_if_unexpected_image_shape(arr, height, width, color)
        return arr

    def sendImage(self, img_array: np.ndarray) -> None:
        """Send one numpy array response as .npy to the active connection."""
        self.send_numpy(img_array)

    def recv_numpy(self) -> np.ndarray:
        """Receive exactly one NumPy .npy array from the active connection."""
        if self.conn is None:
            raise RuntimeError("No client connected")

        payload = self._recv_exact_npy_payload(self.conn)
        return np.load(io.BytesIO(payload), allow_pickle=False)

    def send_numpy(self, arr: np.ndarray) -> None:
        """Serialize and send exactly one NumPy .npy array."""
        if self.conn is None:
            raise RuntimeError("No client connected")

        buffer = io.BytesIO()
        np.save(buffer, np.asarray(arr), allow_pickle=False)
        self.conn.sendall(buffer.getvalue())

    def close_client(self) -> None:
        """Close only the active client connection, not the listening socket."""
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None
                self.addr = None

    def close(self) -> None:
        """Close the active client connection and the listening socket."""
        self.close_client()
        self.server_socket.close()

    @staticmethod
    def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
        """Read exactly nbytes from a blocking socket."""
        chunks = bytearray()
        while len(chunks) < nbytes:
            chunk = sock.recv(nbytes - len(chunks))
            if not chunk:
                raise EOFError(
                    f"Socket closed while reading {nbytes} bytes "
                    f"({len(chunks)} bytes received)"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    @classmethod
    def _recv_exact_npy_payload(cls, sock: socket.socket) -> bytes:
        """Read one complete .npy payload by parsing the .npy header."""
        magic = cls._recv_exact(sock, len(_MAGIC))
        if magic != _MAGIC:
            raise ValueError(f"Invalid .npy magic bytes: {magic!r}")

        version_bytes = cls._recv_exact(sock, 2)
        version = (version_bytes[0], version_bytes[1])
        if version not in _HEADER_LEN_SIZE:
            raise ValueError(f"Unsupported .npy version: {version}")

        header_len_size = _HEADER_LEN_SIZE[version]
        header_len_bytes = cls._recv_exact(sock, header_len_size)
        header_len_fmt = "<H" if header_len_size == 2 else "<I"
        header_len = struct.unpack(header_len_fmt, header_len_bytes)[0]
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
            raise ValueError(f"Invalid .npy header length: {header_len}")

        header = cls._recv_exact(sock, header_len)
        metadata = cls._parse_npy_header(header, version)
        data_nbytes = cls._npy_data_nbytes(metadata)
        if data_nbytes > _MAX_ARRAY_BYTES:
            raise ValueError(f"Refusing to receive huge array: {data_nbytes} bytes")

        data = cls._recv_exact(sock, data_nbytes)
        return magic + version_bytes + header_len_bytes + header + data

    @staticmethod
    def _parse_npy_header(header: bytes, version: Tuple[int, int]) -> dict:
        encoding = "latin1" if version in {(1, 0), (2, 0)} else "utf-8"
        text = header.decode(encoding).strip()
        metadata = ast.literal_eval(text)

        required = {"descr", "fortran_order", "shape"}
        missing = required.difference(metadata)
        if missing:
            raise ValueError(f"Invalid .npy header; missing keys: {sorted(missing)}")

        if metadata["fortran_order"]:
            raise ValueError("Fortran-order .npy arrays are not supported by this server")

        return metadata

    @staticmethod
    def _npy_data_nbytes(metadata: dict) -> int:
        dtype = np.dtype(metadata["descr"])
        if dtype.hasobject:
            raise ValueError("Object arrays are not accepted; pickle is disabled")

        shape = metadata["shape"]
        if shape == ():
            count = 1
        else:
            count = int(np.prod(shape, dtype=np.int64))

        return count * dtype.itemsize

    @staticmethod
    def _warn_if_unexpected_image_shape(
        arr: np.ndarray,
        height: Optional[int],
        width: Optional[int],
        color: bool,
    ) -> None:
        if height is None or width is None:
            return

        expected_shape = (height, width, 3) if color else (height, width)
        if tuple(arr.shape) != expected_shape:
            print(
                f"Warning: received array shape {arr.shape}, expected {expected_shape}",
                flush=True,
            )
