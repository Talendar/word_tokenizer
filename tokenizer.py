#@title Tokenizer
import multiprocessing
import pickle
import re
from operator import itemgetter


def _count_tokens(tokens):
    count = {}
    for token in tokens:
        if token not in count:
            count[token] = 0
        count[token] += 1
    return count


def _merge_tokens_counts(counts):
    total_count = {}
    for count in counts:
        for token in count:
            if token not in total_count:
                total_count[token] = 0
            total_count[token] += count[token]
    return total_count


def _make_batches(items, num_batches):
    k, m = divmod(len(items), num_batches)
    return [items[(i * k + min(i, m)):((i + 1) * k + min(i + 1, m))]
            for i in range(num_batches)]


def _select_most_frequent(tokens_count, vocab_size):
    if len(tokens_count) <= vocab_size:
        return tokens_count

    # Sorting:
    sorted_counts = sorted(list(tokens_count.items()),
                           key=itemgetter(1),
                           reverse=True)
    
    # Selecting most frequent tokens:
    sorted_counts = sorted_counts[:vocab_size]

    # Re-building dict and returning:
    return {k:v for (k, v) in sorted_counts}



class Tokenizer:
    _NUMS = "(?:\d+\ ?)+(?:[.,]\d+)?"
    _WORDS = "[\w]+(?:-\w{4,})?"  # sexta-feira = 1 token; digo-lhe = 3 tokens
    _PUNCTUATION = '(?:\.{3})|[.,!?;:()\-"\/—]'

    def __init__(
        self,
        vocab_size=50_000,
        out_length=628,
        lower_texts=True,
        pad_token="[PAD]",
        oov_token="[OOV]",
        num_token="[NUM]",
    ):
        self._regex = re.compile(
            f"(?P<num>{Tokenizer._NUMS})|"
            f"(?P<word>{Tokenizer._WORDS})|"
            f"(?P<punct>{Tokenizer._PUNCTUATION})"
        )

        self._vocab_size = vocab_size - 3
        self._out_length = out_length
        self._lower_texts = lower_texts

        self._pad_token = pad_token
        self._oov_token = oov_token
        self._num_token = num_token

        self._vocab = {
            f"{pad_token}": 0,
            f"{oov_token}": 1,
            f"{num_token}": 2,
        }
        self._vocab_reverse = None

    @property
    def vocab(self):
        return self._vocab
    
    @property
    def out_length(self):
        return self._out_length

    @property
    def pad_token(self):
        return self._pad_token

    @property
    def oov_token(self):
        return self._oov_token

    @property
    def num_token(self):
        return self._num_token

    def tokenize(self, text):
        if self._lower_texts:
            text = text.lower()

        tokens = []
        matches = self._regex.findall(text)
        for m in matches:
            for i in range(len(m)):
                if m[i] != "":
                    tokens.append(m[i] if i != 0 else self._num_token)

        return tokens

    def _train_batch(self, batch, cache_size, bid):        
        print(f"[B{bid}] Tokenizing and counting...")
        counts = []
        for txt in batch:
            tokens = self.tokenize(txt)
            counts.append(_count_tokens(tokens))
        
        print(f"[B{bid}] Merging counts...")
        tokens_count = _merge_tokens_counts(counts)
        return tokens_count

        print(f"[B{bid}] Selecting most frequent tokens...")
        return _select_most_frequent(tokens_count, cache_size)

    def train(self, texts, batch_size=None, vocab_cache_size=1_024_000):
        if len(self._vocab) > 3:
            raise RuntimeError("This tokenizer has already been trained!")

        # Making batches:
        print(f"Making batches...")
        num_cpu = multiprocessing.cpu_count()
        if batch_size is not None:
            batch_count = min(max(int(len(texts) / batch_size), num_cpu),
                              len(texts))
        else:
            batch_count = min(num_cpu, len(texts))

        batches = _make_batches(texts, num_batches=batch_count)
        batches = [(b, vocab_cache_size, bid) for bid, b in enumerate(batches)]

        print(f"{len(batches)} batches created "
              f"(~{len(batches[0][0])} items per batch)!")
        print(f"Starting processes...\n" + "-" * 30)

        # Tokenizing and counting tokens:
        with multiprocessing.Pool(processes=num_cpu) as pool:
            tokens_count = pool.starmap(self._train_batch, batches)
            tokens_count = _merge_tokens_counts(tokens_count)

            # Removing the "num_token" from the dict:
            tokens_count.pop(self._num_token, None)

        print("-" * 30)

        # Sorting:
        print("Sorting...")
        sorted_counts = sorted(list(tokens_count.items()),
                               key=itemgetter(1),
                               reverse=True)
        
        # Selecting most frequent tokens:
        if len(sorted_counts) > self._vocab_size:
            sorted_counts = sorted_counts[:self._vocab_size]
        
        # Adding tokens to vocabulary:
        print("Adding tokens to vocabulary...")
        for idx, (token, _) in enumerate(sorted_counts):
            assert token not in self._vocab
            self._vocab[token] = idx + 3

        # Checking vocab integrity:
        print("Checking vocab ids integrity...")
        all_ids = set()
        for (token, idx) in self._vocab.items():
            assert idx not in all_ids
            all_ids.add(idx)

        print("Done!")
        return tokens_count, sorted_counts

    def encode(self, text=None, tokens=None, padding=True, cropping=True):
        # Validating arguments:
        if (text is None and tokens is None) or None not in [text, tokens]:
            raise ValueError("You must provide either a text or a list of tokens!")

        # Tokenizing:
        if tokens is None:
            tokens = self.tokenize(text)

        # Cropping:
        if cropping and len(tokens) > self._out_length:
            tokens = tokens[:self._out_length]
        
        # Encoding:
        encodings = []
        for token in tokens:
            encodings.append(self._vocab[token if token in self._vocab
                                         else self._oov_token])
        
        # Padding:
        if padding and len(encodings) < self._out_length:
            diff = self._out_length - len(encodings)
            encodings += [self._vocab[self._pad_token]] * diff
        
        return encodings
    
    def decode(self, encodings, cache_reverse_vocab=False):
        vocab_reverse = self._vocab_reverse
        if vocab_reverse is None:
            vocab_reverse = { v:k for k, v in self._vocab.items() }
            if cache_reverse_vocab:
                self._vocab_reverse = vocab_reverse
        
        return [vocab_reverse[e] for e in encodings]
    
    def _encode_batch_helper(self, batch, padding, cropping, pid):
        encodings = []
        for txt in batch:
            encodings.append(self.encode(text=txt,
                                         padding=padding,
                                         cropping=cropping))
        return encodings
    
    def encode_batch(self, texts, padding=True, cropping=True):
        # Making batches:
        num_cpu = multiprocessing.cpu_count()
        batches = _make_batches(texts, num_batches=min(num_cpu, len(texts)))
        batches = [(b, padding, cropping, pid) for pid, b in enumerate(batches)]

        # Tokenizing and encoding:
        with multiprocessing.Pool(processes=len(batches)) as pool:
            encodings_batches = pool.starmap(self._encode_batch_helper, batches)

            encodings = []
            for enc in encodings_batches:
                encodings += enc
        
        return encodings
    
    def save(self, path):
        if len(path) < 4 or path[-4:] != ".pkl":
            path += ".pkl"

        with open(path, "wb") as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path):
        with open(path, "rb") as inp:
            return pickle.load(inp)
