import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict, Optional
import os
import json
from itertools import tee


class SelectiveTokenPooling(nn.Module):
    """This is baed on sentence_transformers and performs selective token pooling (max or mean) on the token embeddings.
    
    This pooling is performed with the remaining tokens excluding tokens that are likely to be either function words or stop words.
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 vocab: Dict[str, int],
                 function_tokens: List[str] = None,
                 stop_tokens: List[str] = None,
                 pooling_mode: str = None,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 device: Optional[str] = None
                 ):
        super(SelectiveTokenPooling, self).__init__()
        
        self.config_keys = ['word_embedding_dimension', 'vocab', 'function_tokens', 'stop_tokens', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens']
        self.vocab = vocab
        self.function_tokens = function_tokens
        self.stop_tokens = stop_tokens
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._target_device = torch.device(device)
        
        self.function_tokens_ids = None
        if function_tokens != None:
            self.function_tokens_ids = sorted([vocab[tok.strip()] for tok in function_tokens if vocab.get(tok.strip())])
            self.function_tokens_ids = torch.tensor(self.function_tokens_ids, device=self._target_device)
            
        self.stop_tokens_ids = None
        if stop_tokens != None:
            self.stop_tokens_ids = sorted([vocab[tok.strip()] for tok in stop_tokens if vocab.get(tok.strip())])
            self.stop_tokens_ids = torch.tensor(self.stop_tokens_ids, device=self._target_device)
            
        self.vocab_types = [1 if k.startswith('##') else 0 for k,_ in sorted(vocab.items(), key=lambda e: e[1])]
        self.vocab_types = torch.tensor(self.vocab_types, device=self._target_device)
        
        self.function_token_mask = torch.tensor((1,0), device=self._target_device) # function token is simply assumed to be the right end of a word
        self.stop_token_mask = torch.tensor((0,0), device=self._target_device) # stop token is assumed a single token with no consecutive subwords
        
        if pooling_mode is not None:
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max']
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')
            
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
            
        pooling_mode_multiplier = sum([pooling_mode_max_tokens, pooling_mode_mean_tokens])
        
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def __repr__(self):
        return "SelectiveTokenPooling({})".format(self.get_config_dict())


    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']
        input_ids = features['input_ids']
        
        ## find token type(prefixed ## or not)
        token_type_a = self.vocab_types[input_ids] # 1: prefixed ##, 0: others
        token_type_b = token_type_a.clone().detach().to(self._target_device)
        add_zero = torch.zeros(token_type_b.size(0), dtype=int, device=self._target_device).unsqueeze(-1)
        token_type_b = torch.cat((token_type_b[:, 1:], add_zero), dim=-1) # token_type_b is same as sliding token_type_a to the left by 1.
        token_pair = torch.stack((token_type_a, token_type_b), dim=-1)
        
        types_mask = None
        
        ## find function tokens
        if self.function_tokens_ids != None:
            function_token_mask = self.function_token_mask.unsqueeze(0).expand(token_pair.size())
            token_xor = torch.logical_xor(token_pair, function_token_mask)
            token_xor_sum = torch.sum(token_xor, dim=-1)
            token_xor_sum = token_xor_sum == 0
        
            function_token = torch.isin(input_ids, self.function_tokens_ids)
            function_types_mask = torch.logical_and(token_xor_sum, function_token)
            function_types_mask = function_types_mask == False
            
            types_mask = function_types_mask
            
        ## find stop tokens
        if self.stop_tokens_ids != None:
            stop_token_mask = self.stop_token_mask.unsqueeze(0).expand(token_pair.size())
            token_xor = torch.logical_xor(token_pair, stop_token_mask)
            token_xor_sum = torch.sum(token_xor, dim=-1)
            token_xor_sum = token_xor_sum == 0
            
            stop_token = torch.isin(input_ids, self.stop_tokens_ids)
            stop_types_mask = torch.logical_and(token_xor_sum, stop_token)
            stop_types_mask = stop_types_mask == False
            
            if types_mask != None:
                types_mask = torch.logical_and(types_mask, stop_types_mask)
            else:
                types_mask = stop_types_mask

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            if types_mask != None:
                types_mask_expanded = types_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                token_embeddings[((input_mask_expanded==0) | (types_mask_expanded==0))] = -1e9  # Set padding tokens to large negative value
            else:
                token_embeddings[input_mask_expanded==0] = -1e9
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            if types_mask != None:
                types_mask_expanded = types_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                mask_expanded = input_mask_expanded * types_mask_expanded
            else:
                mask_expanded = input_mask_expanded
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)

            sum_mask = mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return SelectiveTokenPooling(**config)