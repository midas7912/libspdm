#!/usr/bin/env python3
# /**
#  *  Copyright Notice:
#  *  Copyright 2021-2025 DMTF. All rights reserved.
#  *  License: BSD 3-Clause License. For full text see link:
#  *  https://github.com/DMTF/libspdm/blob/main/LICENSE.md
#  **/

"""
SPDM Key Exchange Implementation in Python

This module implements the SPDM (Security Protocol and Data Model) key exchange
mechanism as defined in DMTF DSP0274 specification.

The key exchange follows the SPDM 1.1+ protocol flow:
1. Requester generates DHE key pair and sends KEY_EXCHANGE request
2. Responder generates DHE key pair, computes shared secret, and sends
   KEY_EXCHANGE_RSP with signature and HMAC
3. Both parties derive session keys from the shared secret

Supported algorithms:
- DHE: ECDHE P-256, P-384, P-521, FFDHE 2048/3072/4096
- Hash: SHA-256, SHA-384, SHA-512
- AEAD: AES-128-GCM, AES-256-GCM, ChaCha20-Poly1305
"""

import os
import struct
import hashlib
import hmac
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple
from abc import ABC, abstractmethod

# Try to import cryptography library for actual crypto operations
try:
    from cryptography.hazmat.primitives.asymmetric import ec, dh
    from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


# =============================================================================
# SPDM Constants (from DSP0274)
# =============================================================================

class SpdmVersion(IntEnum):
    """SPDM protocol versions."""
    SPDM_1_0 = 0x10
    SPDM_1_1 = 0x11
    SPDM_1_2 = 0x12
    SPDM_1_3 = 0x13


class SpdmRequestCode(IntEnum):
    """SPDM request message codes."""
    GET_VERSION = 0x84
    GET_CAPABILITIES = 0xE1
    NEGOTIATE_ALGORITHMS = 0xE3
    GET_DIGESTS = 0x81
    GET_CERTIFICATE = 0x82
    CHALLENGE = 0x83
    GET_MEASUREMENTS = 0xE0
    KEY_EXCHANGE = 0xE4
    FINISH = 0xE5
    PSK_EXCHANGE = 0xE6
    PSK_FINISH = 0xE7
    HEARTBEAT = 0xE8
    KEY_UPDATE = 0xE9
    END_SESSION = 0xEC


class SpdmResponseCode(IntEnum):
    """SPDM response message codes."""
    VERSION = 0x04
    CAPABILITIES = 0x61
    ALGORITHMS = 0x63
    DIGESTS = 0x01
    CERTIFICATE = 0x02
    CHALLENGE_AUTH = 0x03
    MEASUREMENTS = 0x60
    KEY_EXCHANGE_RSP = 0x64
    FINISH_RSP = 0x65
    PSK_EXCHANGE_RSP = 0x66
    PSK_FINISH_RSP = 0x67
    HEARTBEAT_ACK = 0x68
    KEY_UPDATE_ACK = 0x69
    END_SESSION_ACK = 0x6C
    ERROR = 0x7F


class SpdmDheNamedGroup(IntEnum):
    """SPDM DHE named groups."""
    FFDHE_2048 = 0x01
    FFDHE_3072 = 0x02
    FFDHE_4096 = 0x03
    SECP_256_R1 = 0x04  # P-256
    SECP_384_R1 = 0x05  # P-384
    SECP_521_R1 = 0x06  # P-521


class SpdmHashAlgorithm(IntEnum):
    """SPDM hash algorithms."""
    SHA_256 = 0x01
    SHA_384 = 0x02
    SHA_512 = 0x03
    SHA3_256 = 0x04
    SHA3_384 = 0x05
    SHA3_512 = 0x06


class SpdmAeadCipherSuite(IntEnum):
    """SPDM AEAD cipher suites."""
    AES_128_GCM = 0x01
    AES_256_GCM = 0x02
    CHACHA20_POLY1305 = 0x03


class SpdmMeasurementHashType(IntEnum):
    """SPDM measurement summary hash type for KEY_EXCHANGE."""
    NO_MEASUREMENT_SUMMARY_HASH = 0x00
    TCB_COMPONENT_MEASUREMENT_HASH = 0x01
    ALL_MEASUREMENTS_HASH = 0xFF


class SpdmMutAuthRequested(IntEnum):
    """Mutual authentication requested flags."""
    NO_MUT_AUTH = 0x00
    MUT_AUTH_REQUESTED = 0x01
    MUT_AUTH_WITH_ENCAP_REQUEST = 0x02
    MUT_AUTH_WITH_GET_DIGESTS = 0x04


# SPDM protocol constants
SPDM_NONCE_SIZE = 32
SPDM_RANDOM_DATA_SIZE = 32
SPDM_MAX_SLOT_COUNT = 8
SPDM_MAX_OPAQUE_DATA_SIZE = 1024

# Key exchange signing contexts
KEY_EXCHANGE_RESPONSE_SIGN_CONTEXT = b"responder-key_exchange_rsp signing"
KEY_EXCHANGE_REQUESTER_CONTEXT_V12 = b"Requester-KEP-dmtf-spdm-v1.2"
KEY_EXCHANGE_RESPONDER_CONTEXT_V12 = b"Responder-KEP-dmtf-spdm-v1.2"


# =============================================================================
# Helper Functions
# =============================================================================

def get_hash_size(algo: SpdmHashAlgorithm) -> int:
    """Get hash output size in bytes for given algorithm."""
    sizes = {
        SpdmHashAlgorithm.SHA_256: 32,
        SpdmHashAlgorithm.SHA_384: 48,
        SpdmHashAlgorithm.SHA_512: 64,
        SpdmHashAlgorithm.SHA3_256: 32,
        SpdmHashAlgorithm.SHA3_384: 48,
        SpdmHashAlgorithm.SHA3_512: 64,
    }
    return sizes.get(algo, 32)


def get_dhe_key_size(dhe_group: SpdmDheNamedGroup) -> int:
    """Get DHE public key size in bytes for given named group."""
    sizes = {
        SpdmDheNamedGroup.FFDHE_2048: 256,
        SpdmDheNamedGroup.FFDHE_3072: 384,
        SpdmDheNamedGroup.FFDHE_4096: 512,
        SpdmDheNamedGroup.SECP_256_R1: 64,   # 32 bytes x, 32 bytes y
        SpdmDheNamedGroup.SECP_384_R1: 96,   # 48 bytes x, 48 bytes y
        SpdmDheNamedGroup.SECP_521_R1: 132,  # 66 bytes x, 66 bytes y
    }
    return sizes.get(dhe_group, 64)


def get_hashlib_name(algo: SpdmHashAlgorithm) -> str:
    """Get hashlib algorithm name for given SPDM hash algorithm."""
    names = {
        SpdmHashAlgorithm.SHA_256: 'sha256',
        SpdmHashAlgorithm.SHA_384: 'sha384',
        SpdmHashAlgorithm.SHA_512: 'sha512',
        SpdmHashAlgorithm.SHA3_256: 'sha3_256',
        SpdmHashAlgorithm.SHA3_384: 'sha3_384',
        SpdmHashAlgorithm.SHA3_512: 'sha3_512',
    }
    return names.get(algo, 'sha256')


def compute_hash(algo: SpdmHashAlgorithm, data: bytes) -> bytes:
    """Compute hash of data using specified algorithm."""
    return hashlib.new(get_hashlib_name(algo), data).digest()


def compute_hmac(algo: SpdmHashAlgorithm, key: bytes, data: bytes) -> bytes:
    """Compute HMAC of data using specified algorithm."""
    return hmac.new(key, data, get_hashlib_name(algo)).digest()


# =============================================================================
# Message Structures
# =============================================================================

@dataclass
class SpdmMessageHeader:
    """SPDM message header structure."""
    spdm_version: int
    request_response_code: int
    param1: int = 0
    param2: int = 0

    def pack(self) -> bytes:
        """Pack header into bytes."""
        return struct.pack('<BBBB',
                           self.spdm_version,
                           self.request_response_code,
                           self.param1,
                           self.param2)

    @classmethod
    def unpack(cls, data: bytes) -> 'SpdmMessageHeader':
        """Unpack header from bytes."""
        if len(data) < 4:
            raise ValueError("Insufficient data for header")
        version, code, param1, param2 = struct.unpack('<BBBB', data[:4])
        return cls(version, code, param1, param2)

    @property
    def size(self) -> int:
        """Size of header in bytes."""
        return 4


@dataclass
class SpdmKeyExchangeRequest:
    """
    SPDM KEY_EXCHANGE request message.

    Format (SPDM 1.1+):
        - Header (4 bytes): version, KEY_EXCHANGE (0xE4), measurement_hash_type, slot_id
        - ReqSessionId (2 bytes): Requester's session ID portion
        - SessionPolicy (1 byte): Session policy flags (SPDM 1.2+)
        - Reserved (1 byte)
        - RandomData (32 bytes): Random nonce from requester
        - ExchangeData (D bytes): DHE public key
        - OpaqueLength (2 bytes): Length of opaque data
        - OpaqueData (variable): Opaque application data
    """
    spdm_version: int = SpdmVersion.SPDM_1_2
    measurement_hash_type: int = SpdmMeasurementHashType.NO_MEASUREMENT_SUMMARY_HASH
    slot_id: int = 0
    req_session_id: int = 0
    session_policy: int = 0
    random_data: bytes = field(default_factory=lambda: os.urandom(SPDM_RANDOM_DATA_SIZE))
    exchange_data: bytes = field(default_factory=bytes)
    opaque_data: bytes = field(default_factory=bytes)

    def pack(self) -> bytes:
        """Pack request message into bytes."""
        header = SpdmMessageHeader(
            spdm_version=self.spdm_version,
            request_response_code=SpdmRequestCode.KEY_EXCHANGE,
            param1=self.measurement_hash_type,
            param2=self.slot_id
        )

        # Ensure random_data is correct size
        random_data = self.random_data[:SPDM_RANDOM_DATA_SIZE]
        if len(random_data) < SPDM_RANDOM_DATA_SIZE:
            random_data = random_data + b'\x00' * (SPDM_RANDOM_DATA_SIZE - len(random_data))

        msg = header.pack()
        msg += struct.pack('<H', self.req_session_id)  # ReqSessionId
        msg += struct.pack('<B', self.session_policy)   # SessionPolicy
        msg += struct.pack('<B', 0)                     # Reserved
        msg += random_data                               # RandomData
        msg += self.exchange_data                        # ExchangeData (DHE public key)
        msg += struct.pack('<H', len(self.opaque_data)) # OpaqueLength
        msg += self.opaque_data                          # OpaqueData

        return msg

    @classmethod
    def unpack(cls, data: bytes, dhe_key_size: int) -> 'SpdmKeyExchangeRequest':
        """Unpack request message from bytes."""
        if len(data) < 4 + 2 + 1 + 1 + SPDM_RANDOM_DATA_SIZE:
            raise ValueError("Insufficient data for KEY_EXCHANGE request")

        header = SpdmMessageHeader.unpack(data[:4])
        offset = 4

        req_session_id = struct.unpack('<H', data[offset:offset + 2])[0]
        offset += 2

        session_policy = data[offset]
        offset += 1

        # Skip reserved byte
        offset += 1

        random_data = data[offset:offset + SPDM_RANDOM_DATA_SIZE]
        offset += SPDM_RANDOM_DATA_SIZE

        exchange_data = data[offset:offset + dhe_key_size]
        offset += dhe_key_size

        opaque_length = struct.unpack('<H', data[offset:offset + 2])[0]
        offset += 2

        opaque_data = data[offset:offset + opaque_length]

        return cls(
            spdm_version=header.spdm_version,
            measurement_hash_type=header.param1,
            slot_id=header.param2,
            req_session_id=req_session_id,
            session_policy=session_policy,
            random_data=random_data,
            exchange_data=exchange_data,
            opaque_data=opaque_data
        )


@dataclass
class SpdmKeyExchangeResponse:
    """
    SPDM KEY_EXCHANGE_RSP response message.

    Format (SPDM 1.1+):
        - Header (4 bytes): version, KEY_EXCHANGE_RSP (0x64), heartbeat_period, reserved
        - RspSessionId (2 bytes): Responder's session ID portion
        - MutAuthRequested (1 byte): Mutual authentication request flags
        - ReqSlotIdParam (1 byte): Slot ID for mutual auth or echo of request slot
        - RandomData (32 bytes): Random nonce from responder
        - ExchangeData (D bytes): DHE public key
        - MeasurementSummaryHash (H bytes): Measurement hash (if requested)
        - OpaqueLength (2 bytes): Length of opaque data
        - OpaqueData (variable): Opaque application data
        - Signature (S bytes): Signature over transcript
        - VerifyData (H bytes): HMAC verify data
    """
    spdm_version: int = SpdmVersion.SPDM_1_2
    heartbeat_period: int = 0
    rsp_session_id: int = 0
    mut_auth_requested: int = SpdmMutAuthRequested.NO_MUT_AUTH
    req_slot_id_param: int = 0
    random_data: bytes = field(default_factory=lambda: os.urandom(SPDM_RANDOM_DATA_SIZE))
    exchange_data: bytes = field(default_factory=bytes)
    measurement_summary_hash: bytes = field(default_factory=bytes)
    opaque_data: bytes = field(default_factory=bytes)
    signature: bytes = field(default_factory=bytes)
    verify_data: bytes = field(default_factory=bytes)

    def pack(self) -> bytes:
        """Pack response message into bytes."""
        header = SpdmMessageHeader(
            spdm_version=self.spdm_version,
            request_response_code=SpdmResponseCode.KEY_EXCHANGE_RSP,
            param1=self.heartbeat_period,
            param2=0  # Reserved
        )

        # Ensure random_data is correct size
        random_data = self.random_data[:SPDM_RANDOM_DATA_SIZE]
        if len(random_data) < SPDM_RANDOM_DATA_SIZE:
            random_data = random_data + b'\x00' * (SPDM_RANDOM_DATA_SIZE - len(random_data))

        msg = header.pack()
        msg += struct.pack('<H', self.rsp_session_id)    # RspSessionId
        msg += struct.pack('<B', self.mut_auth_requested) # MutAuthRequested
        msg += struct.pack('<B', self.req_slot_id_param)  # ReqSlotIdParam
        msg += random_data                                 # RandomData
        msg += self.exchange_data                          # ExchangeData
        msg += self.measurement_summary_hash               # MeasurementSummaryHash
        msg += struct.pack('<H', len(self.opaque_data))   # OpaqueLength
        msg += self.opaque_data                            # OpaqueData
        msg += self.signature                              # Signature
        msg += self.verify_data                            # VerifyData

        return msg

    @classmethod
    def unpack(cls, data: bytes, dhe_key_size: int, hash_size: int,
               signature_size: int, has_measurement_hash: bool) -> 'SpdmKeyExchangeResponse':
        """Unpack response message from bytes."""
        if len(data) < 4 + 2 + 1 + 1 + SPDM_RANDOM_DATA_SIZE:
            raise ValueError("Insufficient data for KEY_EXCHANGE_RSP")

        header = SpdmMessageHeader.unpack(data[:4])
        offset = 4

        rsp_session_id = struct.unpack('<H', data[offset:offset + 2])[0]
        offset += 2

        mut_auth_requested = data[offset]
        offset += 1

        req_slot_id_param = data[offset]
        offset += 1

        random_data = data[offset:offset + SPDM_RANDOM_DATA_SIZE]
        offset += SPDM_RANDOM_DATA_SIZE

        exchange_data = data[offset:offset + dhe_key_size]
        offset += dhe_key_size

        measurement_hash_size = hash_size if has_measurement_hash else 0
        measurement_summary_hash = data[offset:offset + measurement_hash_size]
        offset += measurement_hash_size

        opaque_length = struct.unpack('<H', data[offset:offset + 2])[0]
        offset += 2

        opaque_data = data[offset:offset + opaque_length]
        offset += opaque_length

        signature = data[offset:offset + signature_size]
        offset += signature_size

        verify_data = data[offset:offset + hash_size]

        return cls(
            spdm_version=header.spdm_version,
            heartbeat_period=header.param1,
            rsp_session_id=rsp_session_id,
            mut_auth_requested=mut_auth_requested,
            req_slot_id_param=req_slot_id_param,
            random_data=random_data,
            exchange_data=exchange_data,
            measurement_summary_hash=measurement_summary_hash,
            opaque_data=opaque_data,
            signature=signature,
            verify_data=verify_data
        )


# =============================================================================
# DHE Key Exchange
# =============================================================================

class DheKeyExchange(ABC):
    """Abstract base class for DHE key exchange."""

    @abstractmethod
    def generate_key_pair(self) -> bytes:
        """Generate DHE key pair and return public key."""
        pass

    @abstractmethod
    def compute_shared_secret(self, peer_public_key: bytes) -> bytes:
        """Compute shared secret from peer's public key."""
        pass

    @abstractmethod
    def get_public_key_size(self) -> int:
        """Get size of public key in bytes."""
        pass


class EcdhKeyExchange(DheKeyExchange):
    """ECDH key exchange implementation using cryptography library."""

    CURVE_MAP = {
        SpdmDheNamedGroup.SECP_256_R1: ec.SECP256R1(),
        SpdmDheNamedGroup.SECP_384_R1: ec.SECP384R1(),
        SpdmDheNamedGroup.SECP_521_R1: ec.SECP521R1(),
    }

    def __init__(self, dhe_group: SpdmDheNamedGroup):
        if not HAS_CRYPTOGRAPHY:
            raise RuntimeError("cryptography library required for ECDH")
        if dhe_group not in self.CURVE_MAP:
            raise ValueError(f"Unsupported DHE group: {dhe_group}")

        self.dhe_group = dhe_group
        self.curve = self.CURVE_MAP[dhe_group]
        self._private_key: Optional[ec.EllipticCurvePrivateKey] = None
        self._public_key_bytes: Optional[bytes] = None

    def generate_key_pair(self) -> bytes:
        """Generate ECDH key pair and return raw public key (x || y)."""
        self._private_key = ec.generate_private_key(self.curve, default_backend())
        public_key = self._private_key.public_key()

        # Get raw public key bytes (x || y format, uncompressed without 0x04 prefix)
        public_numbers = public_key.public_numbers()
        key_size = (self.curve.key_size + 7) // 8

        x_bytes = public_numbers.x.to_bytes(key_size, byteorder='big')
        y_bytes = public_numbers.y.to_bytes(key_size, byteorder='big')

        self._public_key_bytes = x_bytes + y_bytes
        return self._public_key_bytes

    def compute_shared_secret(self, peer_public_key: bytes) -> bytes:
        """Compute shared secret from peer's raw public key (x || y)."""
        if self._private_key is None:
            raise RuntimeError("Key pair not generated")

        key_size = (self.curve.key_size + 7) // 8
        if len(peer_public_key) != 2 * key_size:
            raise ValueError(f"Invalid peer public key size: {len(peer_public_key)}")

        # Parse peer public key
        x = int.from_bytes(peer_public_key[:key_size], byteorder='big')
        y = int.from_bytes(peer_public_key[key_size:], byteorder='big')

        peer_public_numbers = ec.EllipticCurvePublicNumbers(x, y, self.curve)
        peer_public_key_obj = peer_public_numbers.public_key(default_backend())

        # Perform ECDH
        shared_key = self._private_key.exchange(ec.ECDH(), peer_public_key_obj)
        return shared_key

    def get_public_key_size(self) -> int:
        """Get size of public key in bytes."""
        key_size = (self.curve.key_size + 7) // 8
        return 2 * key_size


class MockDheKeyExchange(DheKeyExchange):
    """Mock DHE key exchange for testing without cryptography library."""

    def __init__(self, dhe_group: SpdmDheNamedGroup):
        self.dhe_group = dhe_group
        self._secret = os.urandom(32)
        self._public_key = os.urandom(get_dhe_key_size(dhe_group))

    def generate_key_pair(self) -> bytes:
        """Generate mock key pair."""
        return self._public_key

    def compute_shared_secret(self, peer_public_key: bytes) -> bytes:
        """Compute mock shared secret."""
        # XOR our secret with hash of peer's public key for determinism
        peer_hash = hashlib.sha256(peer_public_key).digest()
        return bytes(a ^ b for a, b in zip(self._secret, peer_hash))

    def get_public_key_size(self) -> int:
        """Get size of public key in bytes."""
        return get_dhe_key_size(self.dhe_group)


def create_dhe_key_exchange(dhe_group: SpdmDheNamedGroup) -> DheKeyExchange:
    """Factory function to create appropriate DHE key exchange instance."""
    if HAS_CRYPTOGRAPHY and dhe_group in EcdhKeyExchange.CURVE_MAP:
        return EcdhKeyExchange(dhe_group)
    else:
        return MockDheKeyExchange(dhe_group)


# =============================================================================
# Key Derivation
# =============================================================================

class SpdmKeyDerivation:
    """
    SPDM key derivation following the specification.

    Key schedule (SPDM 1.2):
        handshake_secret = HKDF-Extract(salt=zeros, IKM=DHE_secret)
        master_secret = HKDF-Extract(salt=derive_secret(handshake_secret, "derived", ""),
                                     IKM=zeros)

    For each direction (requester/responder):
        traffic_secret = derive_secret(handshake_secret, label, transcript_hash)
        key = HKDF-Expand(traffic_secret, "key", key_size)
        iv = HKDF-Expand(traffic_secret, "iv", iv_size)
        finished_key = HKDF-Expand(traffic_secret, "finished", hash_size)
    """

    def __init__(self, hash_algo: SpdmHashAlgorithm):
        self.hash_algo = hash_algo
        self.hash_size = get_hash_size(hash_algo)
        self.hash_name = get_hashlib_name(hash_algo)

    def hkdf_extract(self, salt: bytes, ikm: bytes) -> bytes:
        """HKDF-Extract: Extract a pseudorandom key from input keying material."""
        if not salt:
            salt = b'\x00' * self.hash_size
        return hmac.new(salt, ikm, self.hash_name).digest()

    def hkdf_expand(self, prk: bytes, info: bytes, length: int) -> bytes:
        """HKDF-Expand: Expand a pseudorandom key to desired length."""
        hash_len = self.hash_size
        n = (length + hash_len - 1) // hash_len

        okm = b''
        prev = b''

        for i in range(1, n + 1):
            prev = hmac.new(prk, prev + info + bytes([i]), self.hash_name).digest()
            okm += prev

        return okm[:length]

    def hkdf_expand_label(self, secret: bytes, label: bytes, context: bytes,
                          length: int) -> bytes:
        """
        HKDF-Expand-Label as defined in SPDM spec.

        HkdfLabel = struct {
            uint16 length;          // length of output
            uint8 label_length;     // length of label with prefix
            opaque label<7..255>;   // "spdm1.2 " + label
            uint8 context_length;   // length of context
            opaque context<0..255>; // hash or empty
        }
        """
        spdm_label = b"spdm1.2 " + label

        hkdf_label = struct.pack('<H', length)
        hkdf_label += struct.pack('<B', len(spdm_label))
        hkdf_label += spdm_label
        hkdf_label += struct.pack('<B', len(context))
        hkdf_label += context

        return self.hkdf_expand(secret, hkdf_label, length)

    def derive_secret(self, secret: bytes, label: bytes, messages: bytes) -> bytes:
        """Derive a secret using HKDF-Expand-Label with hashed messages."""
        context = compute_hash(self.hash_algo, messages) if messages else b''
        return self.hkdf_expand_label(secret, label, context, self.hash_size)

    def derive_handshake_secrets(self, dhe_secret: bytes,
                                 th1_hash: bytes) -> Tuple[bytes, bytes, bytes]:
        """
        Derive handshake secrets from DHE shared secret.

        Returns:
            Tuple of (handshake_secret, requester_handshake_secret, responder_handshake_secret)
        """
        # Extract handshake secret
        salt = b'\x00' * self.hash_size
        handshake_secret = self.hkdf_extract(salt, dhe_secret)

        # Derive directional secrets
        req_hs_secret = self.hkdf_expand_label(
            handshake_secret, b"req hs data", th1_hash, self.hash_size)
        rsp_hs_secret = self.hkdf_expand_label(
            handshake_secret, b"rsp hs data", th1_hash, self.hash_size)

        return handshake_secret, req_hs_secret, rsp_hs_secret

    def derive_finished_key(self, handshake_secret: bytes) -> bytes:
        """Derive the finished key for HMAC computation."""
        return self.hkdf_expand_label(handshake_secret, b"finished", b'', self.hash_size)

    def derive_session_keys(self, master_secret: bytes, th2_hash: bytes,
                            key_size: int, iv_size: int) -> dict:
        """
        Derive session keys from master secret.

        Returns dict with requester and responder keys/IVs.
        """
        # Derive application traffic secrets
        req_app_secret = self.hkdf_expand_label(
            master_secret, b"req app data", th2_hash, self.hash_size)
        rsp_app_secret = self.hkdf_expand_label(
            master_secret, b"rsp app data", th2_hash, self.hash_size)

        # Derive keys and IVs
        return {
            'requester_key': self.hkdf_expand_label(req_app_secret, b"key", b'', key_size),
            'requester_iv': self.hkdf_expand_label(req_app_secret, b"iv", b'', iv_size),
            'responder_key': self.hkdf_expand_label(rsp_app_secret, b"key", b'', key_size),
            'responder_iv': self.hkdf_expand_label(rsp_app_secret, b"iv", b'', iv_size),
        }


# =============================================================================
# SPDM Session
# =============================================================================

@dataclass
class SpdmSessionSecrets:
    """Container for SPDM session secrets and keys."""
    dhe_secret: bytes = field(default_factory=bytes)
    handshake_secret: bytes = field(default_factory=bytes)
    master_secret: bytes = field(default_factory=bytes)
    requester_handshake_secret: bytes = field(default_factory=bytes)
    responder_handshake_secret: bytes = field(default_factory=bytes)
    requester_finished_key: bytes = field(default_factory=bytes)
    responder_finished_key: bytes = field(default_factory=bytes)
    requester_key: bytes = field(default_factory=bytes)
    requester_iv: bytes = field(default_factory=bytes)
    responder_key: bytes = field(default_factory=bytes)
    responder_iv: bytes = field(default_factory=bytes)


class SpdmKeyExchangeSession:
    """
    SPDM key exchange session manager.

    This class manages the complete key exchange flow including:
    - DHE key pair generation
    - Message construction and parsing
    - Shared secret computation
    - Key derivation
    - HMAC verification
    """

    def __init__(self,
                 spdm_version: int = SpdmVersion.SPDM_1_2,
                 dhe_group: SpdmDheNamedGroup = SpdmDheNamedGroup.SECP_256_R1,
                 hash_algo: SpdmHashAlgorithm = SpdmHashAlgorithm.SHA_256,
                 aead_algo: SpdmAeadCipherSuite = SpdmAeadCipherSuite.AES_128_GCM):

        self.spdm_version = spdm_version
        self.dhe_group = dhe_group
        self.hash_algo = hash_algo
        self.aead_algo = aead_algo

        self.dhe = create_dhe_key_exchange(dhe_group)
        self.kdf = SpdmKeyDerivation(hash_algo)

        self.secrets = SpdmSessionSecrets()
        self.transcript = b''  # Message transcript for TH calculation

        # Session IDs
        self.req_session_id = 0
        self.rsp_session_id = 0

        # Public keys
        self.my_public_key: bytes = b''
        self.peer_public_key: bytes = b''

        # Nonces
        self.my_random_data: bytes = b''
        self.peer_random_data: bytes = b''

    def get_session_id(self) -> int:
        """Get combined session ID (requester || responder)."""
        return (self.req_session_id << 16) | self.rsp_session_id

    def generate_requester_key_exchange(self,
                                        slot_id: int = 0,
                                        measurement_hash_type: int = SpdmMeasurementHashType.NO_MEASUREMENT_SUMMARY_HASH,
                                        session_policy: int = 0,
                                        opaque_data: bytes = b'') -> SpdmKeyExchangeRequest:
        """
        Generate KEY_EXCHANGE request as requester.

        Args:
            slot_id: Certificate slot ID for responder authentication
            measurement_hash_type: Type of measurement summary hash to request
            session_policy: Session policy flags
            opaque_data: Optional opaque application data

        Returns:
            SpdmKeyExchangeRequest ready to send
        """
        # Generate session ID (avoid reserved values 0x0000 and 0xFFFF)
        while True:
            self.req_session_id = struct.unpack('<H', os.urandom(2))[0]
            if self.req_session_id != 0 and self.req_session_id != 0xFFFF:
                break

        # Generate DHE key pair
        self.my_public_key = self.dhe.generate_key_pair()

        # Generate random nonce
        self.my_random_data = os.urandom(SPDM_RANDOM_DATA_SIZE)

        # Create request
        request = SpdmKeyExchangeRequest(
            spdm_version=self.spdm_version,
            measurement_hash_type=measurement_hash_type,
            slot_id=slot_id,
            req_session_id=self.req_session_id,
            session_policy=session_policy,
            random_data=self.my_random_data,
            exchange_data=self.my_public_key,
            opaque_data=opaque_data
        )

        # Update transcript with request (excluding signature/verify data parts)
        self.transcript += request.pack()

        return request

    def process_responder_key_exchange(self,
                                       peer_public_key: bytes,
                                       peer_random_data: bytes,
                                       rsp_session_id: int,
                                       transcript_for_th1: bytes) -> bool:
        """
        Process KEY_EXCHANGE_RSP as requester.

        Args:
            peer_public_key: Responder's DHE public key
            peer_random_data: Responder's random nonce
            rsp_session_id: Responder's session ID
            transcript_for_th1: Transcript for TH1 calculation (request + partial response)

        Returns:
            True if processing succeeded
        """
        self.peer_public_key = peer_public_key
        self.peer_random_data = peer_random_data
        self.rsp_session_id = rsp_session_id

        # Compute shared secret
        try:
            self.secrets.dhe_secret = self.dhe.compute_shared_secret(peer_public_key)
        except Exception as e:
            print(f"Failed to compute shared secret: {e}")
            return False

        # Use same transcript as responder
        self.transcript = transcript_for_th1

        # Compute transcript hash (TH1)
        th1_hash = compute_hash(self.hash_algo, self.transcript)

        # Derive handshake secrets
        (self.secrets.handshake_secret,
         self.secrets.requester_handshake_secret,
         self.secrets.responder_handshake_secret) = self.kdf.derive_handshake_secrets(
            self.secrets.dhe_secret, th1_hash)

        # Derive finished keys
        self.secrets.requester_finished_key = self.kdf.derive_finished_key(
            self.secrets.requester_handshake_secret)
        self.secrets.responder_finished_key = self.kdf.derive_finished_key(
            self.secrets.responder_handshake_secret)

        return True

    def generate_responder_key_exchange(self,
                                        request: SpdmKeyExchangeRequest,
                                        heartbeat_period: int = 0,
                                        mut_auth_requested: int = SpdmMutAuthRequested.NO_MUT_AUTH,
                                        measurement_summary_hash: bytes = b'',
                                        opaque_data: bytes = b'') -> Tuple[SpdmKeyExchangeResponse, bytes]:
        """
        Generate KEY_EXCHANGE_RSP response as responder.

        Args:
            request: Received KEY_EXCHANGE request
            heartbeat_period: Heartbeat period in seconds
            mut_auth_requested: Mutual authentication flags
            measurement_summary_hash: Measurement summary hash (if requested)
            opaque_data: Optional opaque application data

        Returns:
            Tuple of (SpdmKeyExchangeResponse, transcript_for_th1) ready to send
        """
        # Store request info
        self.req_session_id = request.req_session_id
        self.peer_public_key = request.exchange_data
        self.peer_random_data = request.random_data

        # Generate session ID (avoid reserved values 0x0000 and 0xFFFF)
        while True:
            self.rsp_session_id = struct.unpack('<H', os.urandom(2))[0]
            if self.rsp_session_id != 0 and self.rsp_session_id != 0xFFFF:
                break

        # Generate DHE key pair
        self.my_public_key = self.dhe.generate_key_pair()

        # Generate random nonce
        self.my_random_data = os.urandom(SPDM_RANDOM_DATA_SIZE)

        # Compute shared secret
        self.secrets.dhe_secret = self.dhe.compute_shared_secret(self.peer_public_key)

        # Build transcript for TH1 calculation
        # TH1 = Hash(VCA || KEY_EXCHANGE request || KEY_EXCHANGE_RSP without sig/verify)
        request_bytes = request.pack()

        # Create response (without signature and verify_data for now)
        response = SpdmKeyExchangeResponse(
            spdm_version=self.spdm_version,
            heartbeat_period=heartbeat_period,
            rsp_session_id=self.rsp_session_id,
            mut_auth_requested=mut_auth_requested,
            req_slot_id_param=request.slot_id,
            random_data=self.my_random_data,
            exchange_data=self.my_public_key,
            measurement_summary_hash=measurement_summary_hash,
            opaque_data=opaque_data,
            signature=b'',  # Would be computed with private signing key
            verify_data=b''  # Will be computed after deriving keys
        )

        # Build partial response for transcript (everything up to signature)
        partial_response_bytes = response.pack()

        # Transcript for TH1 = request + partial response (without sig/verify_data)
        transcript_for_th1 = request_bytes + partial_response_bytes

        # Store for later use
        self.transcript = transcript_for_th1

        # Compute transcript hash (TH1)
        th1_hash = compute_hash(self.hash_algo, transcript_for_th1)

        # Derive handshake secrets
        (self.secrets.handshake_secret,
         self.secrets.requester_handshake_secret,
         self.secrets.responder_handshake_secret) = self.kdf.derive_handshake_secrets(
            self.secrets.dhe_secret, th1_hash)

        # Derive finished keys
        self.secrets.responder_finished_key = self.kdf.derive_finished_key(
            self.secrets.responder_handshake_secret)
        self.secrets.requester_finished_key = self.kdf.derive_finished_key(
            self.secrets.requester_handshake_secret)

        # Compute verify_data (HMAC over transcript hash using finished key)
        response.verify_data = compute_hmac(
            self.hash_algo, self.secrets.responder_finished_key, th1_hash)

        return response, transcript_for_th1

    def verify_response_hmac(self, verify_data: bytes) -> bool:
        """
        Verify the HMAC in KEY_EXCHANGE_RSP.

        Args:
            verify_data: HMAC from response

        Returns:
            True if HMAC is valid
        """
        th1_hash = compute_hash(self.hash_algo, self.transcript)
        expected_hmac = compute_hmac(
            self.hash_algo, self.secrets.responder_finished_key, th1_hash)

        return hmac.compare_digest(verify_data, expected_hmac)

    def derive_session_keys(self) -> dict:
        """
        Derive final session keys after FINISH handshake.

        Returns:
            Dictionary with session keys and IVs
        """
        # Compute TH2 (includes FINISH message in real implementation)
        th2_hash = compute_hash(self.hash_algo, self.transcript)

        # Derive master secret from handshake secret
        derived_secret = self.kdf.derive_secret(
            self.secrets.handshake_secret, b"derived", b'')
        zeros = b'\x00' * self.kdf.hash_size
        self.secrets.master_secret = self.kdf.hkdf_extract(derived_secret, zeros)

        # Get key/IV sizes based on AEAD algorithm
        key_sizes = {
            SpdmAeadCipherSuite.AES_128_GCM: (16, 12),
            SpdmAeadCipherSuite.AES_256_GCM: (32, 12),
            SpdmAeadCipherSuite.CHACHA20_POLY1305: (32, 12),
        }
        key_size, iv_size = key_sizes.get(self.aead_algo, (16, 12))

        # Derive session keys
        keys = self.kdf.derive_session_keys(
            self.secrets.master_secret, th2_hash, key_size, iv_size)

        self.secrets.requester_key = keys['requester_key']
        self.secrets.requester_iv = keys['requester_iv']
        self.secrets.responder_key = keys['responder_key']
        self.secrets.responder_iv = keys['responder_iv']

        return keys


# =============================================================================
# Utility Functions
# =============================================================================

def bytes_to_hex(data: bytes, sep: str = ' ') -> str:
    """Convert bytes to hex string with separator."""
    return sep.join(f'{b:02x}' for b in data)


def demo_key_exchange():
    """Demonstrate SPDM key exchange between requester and responder."""
    print("=" * 60)
    print("SPDM Key Exchange Demonstration")
    print("=" * 60)
    print()

    # Configuration
    spdm_version = SpdmVersion.SPDM_1_2
    dhe_group = SpdmDheNamedGroup.SECP_256_R1
    hash_algo = SpdmHashAlgorithm.SHA_256
    aead_algo = SpdmAeadCipherSuite.AES_128_GCM

    print(f"SPDM Version: {hex(spdm_version)}")
    print(f"DHE Group: {dhe_group.name}")
    print(f"Hash Algorithm: {hash_algo.name}")
    print(f"AEAD Algorithm: {aead_algo.name}")
    print(f"Cryptography library available: {HAS_CRYPTOGRAPHY}")
    print()

    # Create requester and responder sessions
    requester = SpdmKeyExchangeSession(spdm_version, dhe_group, hash_algo, aead_algo)
    responder = SpdmKeyExchangeSession(spdm_version, dhe_group, hash_algo, aead_algo)

    # Step 1: Requester generates KEY_EXCHANGE request
    print("Step 1: Requester generates KEY_EXCHANGE request")
    print("-" * 40)
    request = requester.generate_requester_key_exchange(
        slot_id=0,
        measurement_hash_type=SpdmMeasurementHashType.NO_MEASUREMENT_SUMMARY_HASH
    )
    request_bytes = request.pack()
    print(f"Request size: {len(request_bytes)} bytes")
    print(f"Requester session ID: {hex(request.req_session_id)}")
    print(f"Requester public key: {bytes_to_hex(request.exchange_data[:16])}...")
    print()

    # Step 2: Responder processes request and generates response
    print("Step 2: Responder generates KEY_EXCHANGE_RSP response")
    print("-" * 40)
    response, transcript_for_th1 = responder.generate_responder_key_exchange(request)
    response_bytes = response.pack()
    print(f"Response size: {len(response_bytes)} bytes")
    print(f"Responder session ID: {hex(response.rsp_session_id)}")
    print(f"Responder public key: {bytes_to_hex(response.exchange_data[:16])}...")
    print(f"Verify data (HMAC): {bytes_to_hex(response.verify_data[:16])}...")
    print()

    # Step 3: Requester processes response
    print("Step 3: Requester processes KEY_EXCHANGE_RSP")
    print("-" * 40)

    # Build the same transcript on requester side (simulating message exchange)
    # In real protocol, requester builds this from request it sent + response it received
    requester_request_bytes = request.pack()

    # Reconstruct partial response (without verify_data we need to reconstruct)
    partial_response = SpdmKeyExchangeResponse(
        spdm_version=response.spdm_version,
        heartbeat_period=response.heartbeat_period,
        rsp_session_id=response.rsp_session_id,
        mut_auth_requested=response.mut_auth_requested,
        req_slot_id_param=response.req_slot_id_param,
        random_data=response.random_data,
        exchange_data=response.exchange_data,
        measurement_summary_hash=response.measurement_summary_hash,
        opaque_data=response.opaque_data,
        signature=b'',
        verify_data=b''
    )
    requester_transcript = requester_request_bytes + partial_response.pack()

    success = requester.process_responder_key_exchange(
        peer_public_key=response.exchange_data,
        peer_random_data=response.random_data,
        rsp_session_id=response.rsp_session_id,
        transcript_for_th1=requester_transcript
    )
    print(f"Response processing: {'SUCCESS' if success else 'FAILED'}")

    # Verify HMAC
    hmac_valid = requester.verify_response_hmac(response.verify_data)
    print(f"HMAC verification: {'VALID' if hmac_valid else 'INVALID'}")
    print()

    # Step 4: Derive session keys
    print("Step 4: Derive session keys")
    print("-" * 40)
    requester_keys = requester.derive_session_keys()
    responder_keys = responder.derive_session_keys()

    print("Requester session keys:")
    print(f"  Key: {bytes_to_hex(requester_keys['requester_key'])}")
    print(f"  IV:  {bytes_to_hex(requester_keys['requester_iv'])}")
    print()
    print("Responder session keys:")
    print(f"  Key: {bytes_to_hex(responder_keys['responder_key'])}")
    print(f"  IV:  {bytes_to_hex(responder_keys['responder_iv'])}")
    print()

    # Verify keys match
    print("Step 5: Verify key agreement")
    print("-" * 40)
    dhe_match = requester.secrets.dhe_secret == responder.secrets.dhe_secret
    req_key_match = requester_keys['requester_key'] == responder_keys['requester_key']
    rsp_key_match = requester_keys['responder_key'] == responder_keys['responder_key']

    print(f"DHE shared secret match: {'YES' if dhe_match else 'NO'}")
    print(f"Requester key match: {'YES' if req_key_match else 'NO'}")
    print(f"Responder key match: {'YES' if rsp_key_match else 'NO'}")
    print()

    if dhe_match and req_key_match and rsp_key_match:
        print("KEY EXCHANGE SUCCESSFUL!")
        session_id = requester.get_session_id()
        print(f"Session ID: {hex(session_id)}")
    else:
        print("KEY EXCHANGE FAILED!")

    print("=" * 60)


if __name__ == "__main__":
    demo_key_exchange()
