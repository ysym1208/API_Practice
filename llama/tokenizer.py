# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List, Optional

from sentencepiece import SentencePieceProcessor

logger = getLogger()

class Tokenizer:
    def __init__(self, model_path: Optional[str] = None):
        if model_path and os.path.isfile(model_path):
            self.sp_model = SentencePieceProcessor(model_file=model_path)
            logger.info(f"Reloaded SentencePiece model from {model_path}")
        else:
            # 모델 파일이 없는 경우 기본 설정으로 초기화
            self.sp_model = SentencePieceProcessor()
            logger.warning(f"Model path '{model_path}' not found. Initialized with default settings.")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size() if model_path else 32000
        self.bos_id: int = self.sp_model.bos_id() if model_path else 0
        self.eos_id: int = self.sp_model.eos_id() if model_path else 1
        self.pad_id: int = self.sp_model.pad_id() if model_path else 2

        # token IDs for special infilling tokens
        self.prefix_id: Optional[int] = self.sp_model.piece_to_id("▁<PRE>") if model_path else None
        self.middle_id: Optional[int] = self.sp_model.piece_to_id("▁<MID>") if model_path else None
        self.suffix_id: Optional[int] = self.sp_model.piece_to_id("▁<SUF>") if model_path else None
        self.eot_id: Optional[int] = self.sp_model.piece_to_id("▁<EOT>") if model_path else None

        # marker for turn-based step format
        self.step_id: Optional[int] = self.sp_model.piece_to_id("<step>") if model_path else None

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} "
            f"- PRE ID: {self.prefix_id} - MID ID: {self.middle_id} - SUF ID: {self.suffix_id} - EOT ID: {self.eot_id} - STEP ID: {self.step_id}"
        )
        assert not model_path or self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        if not hasattr(self, 'sp_model'):
            return [self.bos_id] + list(s.encode('utf-8')) + [self.eos_id] if bos and eos else list(s.encode('utf-8'))
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        if not hasattr(self, 'sp_model'):
            return bytes(t).decode('utf-8')
        return self.sp_model.decode(list(filter(lambda tk: tk != -1, t)))

    def token_piece(self, t: int) -> str:
        if not hasattr(self, 'sp_model'):
            return str(t)
        return self.sp_model.id_to_piece(t)

    def encode_infilling(self, s: str) -> List[int]:
        """Encode a string without an implicit leading space."""
        if not hasattr(self, 'sp_model'):
            return list(s.encode('utf-8'))
        return self.sp_model.encode("☺" + s)[2:]

    def decode_infilling(self, t: List[int]) -> str:
        """Decode a string without an implicit leading space."""
        if not hasattr(self, 'sp_model'):
            return bytes(t).decode('utf-8')
        return self.sp_model.decode([self.sp_model.piece_to_id("☺")] + t)[1:]

