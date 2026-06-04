import ast
import io
import socket
import struct
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


_NPY_MAGIC = b"\x93NUMPY"
_HEADER_LEN_FORMAT = {
    (1, 0): (2, "<H"),
    (2, 0): (4, "<I"),
    (3, 0): (4, "<I"),
}
_MAX_HEADER_BYTES = 1_000_000
_MAX_ACTION_BYTES = 100_000_000


class OpenVLAClient:
    """TCP client for one OpenVLA image -> action request.

    request_action(frame) is the main API:
      1. opens a fresh socket connection
      2. sends the camera frame as a NumPy .npy payload
      3. reads exactly one NumPy .npy action payload back
      4. closes the socket

    The response shape is read from the .npy header, so the client does not need
    to guess whether OpenVLA returned (7,), (1, 7), or another numeric shape.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        timeout: float = 30.0,
        image_width: Optional[int] = 224,
        image_height: Optional[int] = 224,
        convert_bgr_to_rgb: bool = True,
        keep_alive: bool = True,
    ):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.image_width = image_width
        self.image_height = image_height
        self.convert_bgr_to_rgb = convert_bgr_to_rgb
        self.keep_alive = keep_alive
        self.sock: Optional[socket.socket] = None

    def connect(self) -> socket.socket:
        """Open a connection to the OpenVLA server."""
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(self.timeout)
        return self.sock

    def close(self) -> None:
        """Close the active socket, if one exists."""
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def request_action(
        self,
        frame_bgr: np.ndarray,
        action_shape: Optional[Sequence[int]] = None,
        action_dtype=np.float32,
    ) -> np.ndarray:
        """Send one camera frame and return the OpenVLA action as a flat array."""
        image = self.prepare_frame(frame_bgr)
        expected_shape = tuple(action_shape) if action_shape is not None else None

        try:
            sock = self.sock if self.sock is not None else self.connect()
            self._send_numpy(sock, image)
            action = self._recv_numpy(sock, expected_shape=expected_shape, dtype=action_dtype)
            return np.asarray(action, dtype=action_dtype).reshape(-1)
        except (EOFError, OSError):
            self.close()
            sock = self.connect()
            self._send_numpy(sock, image)
            action = self._recv_numpy(sock, expected_shape=expected_shape, dtype=action_dtype)
            return np.asarray(action, dtype=action_dtype).reshape(-1)
        finally:
            if not self.keep_alive:
                self.close()

    def prepare_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Resize a cv2 frame and convert BGR/BGRA to RGB/RGBA for OpenVLA."""
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        image = np.asarray(frame_bgr)

        if self.image_width is not None and self.image_height is not None:
            image = cv2.resize(image, (self.image_width, self.image_height))

        if self.convert_bgr_to_rgb and image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        return np.ascontiguousarray(image)

    @staticmethod
    def _send_numpy(sock: socket.socket, arr: np.ndarray) -> None:
        """Send one NumPy array as a .npy payload."""
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(arr), allow_pickle=False)
        sock.sendall(buffer.getvalue())

    def _recv_numpy(
        self,
        sock: socket.socket,
        expected_shape: Optional[Tuple[int, ...]] = None,
        dtype=None,
    ) -> np.ndarray:
        """Receive one complete .npy payload and return the OpenVLA action array."""
        payload = self._recv_exact_npy_payload(sock)
        action = np.load(io.BytesIO(payload), allow_pickle=False)
        return self._validate_action(action, expected_shape=expected_shape, dtype=dtype)

    @staticmethod
    def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
        """Read exactly nbytes from a socket."""
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
        """Read exactly one NumPy .npy payload by parsing its header."""
        magic = cls._recv_exact(sock, len(_NPY_MAGIC))
        if magic != _NPY_MAGIC:
            raise ValueError(f"Invalid .npy response from OpenVLA server: {magic!r}")

        version_bytes = cls._recv_exact(sock, 2)
        version = (version_bytes[0], version_bytes[1])
        if version not in _HEADER_LEN_FORMAT:
            raise ValueError(f"Unsupported .npy version from OpenVLA server: {version}")

        header_len_size, header_len_fmt = _HEADER_LEN_FORMAT[version]
        header_len_bytes = cls._recv_exact(sock, header_len_size)
        header_len = struct.unpack(header_len_fmt, header_len_bytes)[0]
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
            raise ValueError(f"Invalid .npy header length from server: {header_len}")

        header = cls._recv_exact(sock, header_len)
        metadata = cls._parse_npy_header(header, version)
        data_nbytes = cls._npy_data_nbytes(metadata)
        if data_nbytes > _MAX_ACTION_BYTES:
            raise ValueError(f"OpenVLA action payload is too large: {data_nbytes} bytes")

        data = cls._recv_exact(sock, data_nbytes)
        return magic + version_bytes + header_len_bytes + header + data

    @staticmethod
    def _parse_npy_header(header: bytes, version: Tuple[int, int]) -> dict:
        encoding = "latin1" if version in {(1, 0), (2, 0)} else "utf-8"
        metadata = ast.literal_eval(header.decode(encoding).strip())

        required = {"descr", "fortran_order", "shape"}
        missing = required.difference(metadata)
        if missing:
            raise ValueError(f"Invalid .npy header from server; missing keys: {sorted(missing)}")

        return metadata

    @staticmethod
    def _npy_data_nbytes(metadata: dict) -> int:
        dtype = np.dtype(metadata["descr"])
        if dtype.hasobject:
            raise ValueError("Object arrays are not valid OpenVLA actions")

        shape = metadata["shape"]
        count = 1 if shape == () else int(np.prod(shape, dtype=np.int64))
        return count * dtype.itemsize

    @staticmethod
    def _validate_action(
        action: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]],
        dtype,
    ) -> np.ndarray:
        if expected_shape is not None and tuple(action.shape) != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {action.shape}")

        if not np.issubdtype(action.dtype, np.number):
            raise ValueError(f"OpenVLA action must be numeric, got dtype {action.dtype}")

        if dtype is not None:
            action = action.astype(dtype, copy=False)

        return action
