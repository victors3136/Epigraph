from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

from DeepPhonemizer.dp.model.utils import get_dedup_tokens, _make_len_mask, PositionalEncoding
from DeepPhonemizer.dp.preprocessing.text import Preprocessor


class Model(torch.nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates phonemes for a text batch

        Args:
          batch (Dict[str, torch.Tensor]): Dictionary containing 'text' (tokenized text tensor),
                       'text_len' (text length tensor),
                       'start_index' (phoneme start indices for AutoregressiveTransformer)

        Returns:
          Tuple[torch.Tensor, torch.Tensor]: The predictions. The first element is a tensor (phoneme tokens)
          and the second element  is a tensor (phoneme token probabilities)
        """
        pass


class ForwardTransformer(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=1024,
                 layers=4,
                 dropout=0.1,
                 heads=1) -> None:
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=heads,
                                                dim_feedforward=d_fft,
                                                dropout=dropout,
                                                activation='relu')
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layers,
                                          norm=encoder_norm)

        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    def forward(self,
                batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the model on a data batch.

        Args:
         batch (Dict[str, torch.Tensor]): Input batch entry 'text' (text tensor).

        Returns:
          Tensor: Predictions.
        """

        x = batch['text']
        x = x.transpose(0, 1)        # shape: [T, N]
        src_pad_mask = _make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x

    @torch.jit.export
    def generate(self,
                 batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (Dict[str, torch.Tensor]): Input batch with entry 'text' (text tensor).

        Returns:
          Tuple: The first element is a Tensor (phoneme tokens) and the second element
                 is a tensor (phoneme token probabilities).
        """

        with torch.no_grad():
            x = self.forward(batch)
        tokens, logits = get_dedup_tokens(x)
        return tokens, logits

    @classmethod
    def from_config(cls, config: dict) -> 'ForwardTransformer':
        preprocessor = Preprocessor.from_config(config)
        return ForwardTransformer(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


def create_model(config: Dict[str, Any]) -> Model:
    return ForwardTransformer.from_config(config)


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Tuple[Model, Dict[str, Any]]:
    """
    Initializes a model from a checkpoint (.pt file).

    Args:
        checkpoint_path (str): Path to checkpoint file (.pt).
        device (str): Device to put the model to ('cpu' or 'cuda').

    Returns: Tuple: The first element is a Model (the loaded model)
             and the second element is a dictionary (config).
    """

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(config=checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint
