import numpy as np

def positional_encoding(seq_length, d_model):
  # Generate a matrix of the size sequence length x embedding dimension
  PE = np.zeros((seq_length, d_model))
  position = np.arange(seq_length)[:, np.newaxis]
  div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

  PE[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices
  PE[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices

  return PE

def scaled_dot_product_attention(Q, K, V, mask=None):
  d_k = Q.shape[-1] # embedding dimension per head
  # Q is of size: (batch_size, num_heads, seq_length, d_k)
  # K needs size: (batch_size, num_heads, d_k, sequence_length)
  scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)
  if mask is not None: # ensure no future tokens are attended to
    # even if a position is0, it still contributes to the denominator in softmax (e^0=1)
    # so we set it as negatively as possible, the higher the better
    scores = np.where(mask == 0, -1e9, scores)  # apply the casual mask
  attn_weights = softmax(scores)
  return attn_weights @ V

def softmax(x):
  exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class MultiHeadAttention:
  def __init__(self, d_model, num_heads):
    self.d_model = d_model # embedding dimension
    self.num_heads = num_heads # number of attention heads
    self.d_k = d_model // num_heads # embedding dimension per head
    # Storing the weights for the linear transformation
    # of query, key, and value matrices
    self.W_q = np.random.randn(d_model, d_model) # query weights
    self.W_k = np.random.randn(d_model, d_model) # key weights
    self.W_v = np.random.randn(d_model, d_model) # value weights
    self.W_o = np.random.randn(d_model, d_model) # concat weights

  def split_heads(self, x):
    # X is of size batch_size x sequence_length (rows) x embedding dimension (columns)
    batch_size, seq_length, d_model = x.shape

    # split each sequence_length (rows) x embedding dimension (columns) in X into
    # num_heads different parts
    x = x.reshape(batch_size, seq_length, self.num_heads, self.d_k)
    # we swap around num_heads and sequence_length to go back to operating in
    # batch_size x num_heads x sequence_length x embedding dimension // num_heads
    # because we want to operate on each head independently
    return x.transpose(0, 2, 1, 3)

  def forward(self, X, mask=None):
    Q = self.split_heads(X @ self.W_q)
    K = self.split_heads(X @ self.W_k)
    V = self.split_heads(X @ self.W_v)
    # Q,K,V are of size:
    # batch_size x num_heads x sequence_length x embedding dimension // num_heads
    attn_output = scaled_dot_product_attention(Q, K, V, mask)
    # befor projecting back, we need to undo the transposition in our splitting heads operation
    # then we concat all the heads again
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(X.shape)
    return attn_output @ self.W_o


class FeedForward:
  def __init__(self, d_model, d_ff):
    # intialize weights and biasses
    self.W1 = np.random.randn(d_model, d_ff)
    self.b1 = np.zeros(d_ff)
    self.W2 = np.random.randn(d_ff, d_model)
    self.b2 = np.zeros(d_model)

  def forward(self, x):
    return (np.maximum(0, x @ self.W1 + self.b1)) @ self.W2 + self.b2  # ReLU activation

class DecoderBlock:
  def __init__(self, d_model, num_heads, d_ff):
    self.attn = MultiHeadAttention(d_model, num_heads)
    self.ffn = FeedForward(d_model, d_ff)

  def forward(self, X, mask):
    attn_output = self.attn.forward(X, mask)
    X = X + attn_output  # residual connection
    ffn_output = self.ffn.forward(X)
    return X + ffn_output  # residual connection


class TransformerDecoder:
  def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_seq_length=50):
    self.num_layers = num_layers
    # generate the initial embedding matrix: for each token
    # we crate a 1xd_model adjustable parameter
    self.token_embeddings = np.random.randn(vocab_size, d_model)

    # pre-calculate and store the fixed positional encoding matrix for the max sequence length
    self.positional_encoding = positional_encoding(max_seq_length, d_model)
    # create num_layers decoder blocks
    self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
    self.output_projection = np.random.randn(d_model, vocab_size)  # map function to vocabulary dimension

  def forward(self, X, mask):
    X = self.token_embeddings[X] + self.positional_encoding[:X.shape[1]]
    # X is of size batch_size x sequence_length (rows) x embedding dimension (columns)
    # mask is of size batch_size x sequence_length x sequence_length
    for layer in self.decoder_blocks:
        X = layer.forward(X, mask)
    # X is of size batch_size x sequence_length (rows) x embedding dimension (columns)
    # but needs to be of size batch_size x sequence_length x vocabulary size
    logits = X @ self.output_projection  # map to vocabulary size
    # logits are of size batch_size x sequence_length x vocabulary size
    return logits


def simple_tokenizer(text, vocab):
  return np.array([vocab[char] for char in text if char in vocab], dtype=np.int32)

def simple_detokenizer(tokens, vocab_inv):
  return "".join([vocab_inv[token] for token in tokens])


def predict_next_char(model, input_text, vocab, vocab_inv, max_length=20):
  """
  Generate text character by character using the decoder-only transformer.
  """
  tokenized_text = simple_tokenizer(input_text, vocab) # tokenize the input text
  tokenized_text = np.expand_dims(tokenized_text, axis=0)  # Add batch dimension
  for _ in range(max_length):
    seq_length = tokenized_text.shape[1] # the number of tokens
    # generating the causal mask (lower triangular matrix)
    mask = np.tril(np.ones((1, seq_length, seq_length)))

    logits = model.forward(tokenized_text, mask)

    # logits are of size batch_size x sequence_length x vocabulary size
    next_token_probs = softmax(logits[:, -1, :])  # get last token's logits

    # pick the next most probable token by getting the token at the index
    # of the highest softmax output
    next_token = np.argmax(next_token_probs)  # pick the most probable token
    tokenized_text = np.append(tokenized_text, [[next_token]], axis=1)  # append new token

    # adding some stopping critaria
    # in reality, actual start and stopping tokens are defined that the model
    # can predict to stop the sequence
    if vocab_inv[next_token] == " ":
        break

  return simple_detokenizer(tokenized_text[0], vocab_inv) # return detokenized text

if __name__ == "__main__":
  text = "hello w"

  vocab = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz ")}
  vocab_inv = {i: ch for ch, i in vocab.items()}

  model = TransformerDecoder(
      d_model=16, num_heads=2, d_ff=32, num_layers=2, vocab_size=len(vocab), max_seq_length=50
  )

  predicted_text = predict_next_char(model, text, vocab, vocab_inv, max_length=5)
  print("Generated Text:", predicted_text)
