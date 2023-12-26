# Selective Token Pooling
[SBERT](https://www.sbert.net/index.html)의 출력토큰에 대한 pooling 단계에서 미리 정의한 한국어의 기능어 및 불용어에 해당하는 토큰을 제외시키고 pooling을 수행하여 문장 임베딩 품질 개선을 시도한다.

## Details
김명선, 한동희, 장진목, "문장 임베딩 품질 개선을 위한 선택적 토큰 풀링", 2023 한국소프트웨어종합학술대회 논문집

## Examples
아래와 같이 model을 생성할 수 있다.
```python
from sentence_transformers import SentenceTransformer, models
from selective_token import SelectiveTokenPooling
from torch import nn

def read_tokens(fname):
    with open(fname, 'r') as fp:
        tokens = [line.strip() for line in fp]
    return tokens

function_tokens = read_tokens('function_token.txt')
stop_tokens = read_tokens('stop_token.txt')

word_embedding_model = models.Transformer('klue/bert-base', max_seq_length=256)
vocab = word_embedding_model.tokenizer.get_vocab()
pooling_model = SelectiveTokenPooling(word_embedding_model.get_word_embedding_dimension(),
                                      vocab,
                                      function_tokens=function_tokens,
                                      stop_tokens=stop_tokens)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

## Dependencies
* [sentence_transformers](https://github.com/UKPLab/sentence-transformers)