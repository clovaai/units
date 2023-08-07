import re
import unicodedata


class Tokenizer:
    def __init__(self, key, version=None, go=1, eos=2, unk=3, pad=0, special_tokens=()):
        self.go = go
        self.eos = eos
        self.unk = unk
        self.pad = pad
        self._key = key
        self._version = version
        self.special_tokens = special_tokens

        vocab = list(
            " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~£¥₩€＼"
        )
        self.vocab = {
            "[go]": self.go,
            "[eos]": self.eos,
            "[unk]": self.unk,
            "[pad]": self.pad,
        }

        for id, token in enumerate(special_tokens, start=self.n_vocab):
            self.vocab[token] = id

        for id, ch in enumerate(vocab, start=self.n_vocab):
            self.vocab[ch] = id

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def n_vocab(self):
        return max(self.vocab.values()) + 1

    def normalize(self, text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        chars = []

        for char in text:
            target = unicodedata.normalize("NFKC", char)
            if not (char != " " and target[0] == " "):
                char = target

            chars.append(char)
        text = "".join(chars)
        return text

    def __call__(self, text, add_go_eos=True, normalize=True):
        return self.encode(text, add_go_eos, normalize)

    def encode(self, text, add_go_eos=True, normalize=True):
        if normalize:
            text = normalize(text)

        codes = []
        for ch in text:
            try:
                codes.append(self.vocab[ch])

            except KeyError:
                codes.append(self.unk)

        if add_go_eos:
            return [self.go] + codes + [self.eos]

        return codes

    def decode(self, codes):
        text = []

        for code in codes:
            if code == self.eos:
                break

            text.append(self.inv_vocab[code])

        return "".join(text)


class UnitsTokenizer(Tokenizer):
    def __init__(
        self,
        key=None,
        version=None,
        go=1,
        eos=2,
        unk=3,
        pad=0,
        special_tokens=(
            "[mask]",
            "[noise]",
            "[text]",
            "[roi]",
            "[order]",
            "[point]",
            "[text_eos]",
        ),
    ):
        super().__init__(key, version, go, eos, unk, pad, special_tokens)

        char_vocab = list(
            " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~£¥₩€＼"
        )
        self.char_vocab_range = [self.vocab[char_vocab[0]], self.vocab[char_vocab[-1]]]

    def add_unify_annotation_vocab(
        self,
        special_tokens=(
            "[single]",
            "[box]",
            "[quad]",
            "[polygon]",
            "[case-sensitive]",
            "[case-insensitive]",
        ),
    ):
        for coord_key in special_tokens:
            coord_value = self.n_vocab
            self.vocab[coord_key] = coord_value
            self.inv_vocab[coord_value] = coord_key

    def add_detection_vocab(
        self,
        bin_size,
    ):
        self.bin_size = bin_size

        # coord tokens
        for i in range(bin_size):
            coord_key = f"[coord-{i}]"
            coord_value = self.n_vocab
            self.vocab[coord_key] = coord_value
            self.inv_vocab[coord_value] = coord_key

        # special coord tokens
        special_coord_tokens = ["[coord-out]"]
        for coord_key in special_coord_tokens:
            coord_value = self.n_vocab
            self.vocab[coord_key] = coord_value
            self.inv_vocab[coord_value] = coord_key

    def add_order_vocab(
        self,
        max_order,
    ):
        self.max_order = max_order

        # order tokens
        if max_order is not None:
            for i in range(self.max_order):
                order_key = f"[order-{i}]"
                order_value = self.n_vocab
                self.vocab[order_key] = order_value
                self.inv_vocab[order_value] = order_key

    def __call__(self, text, normalize=True):
        return self.encode(text, normalize)

    def encode(self, text, normalize=True):
        if normalize:
            text = self.normalize(text)

        codes = []
        for ch in text:
            try:
                codes.append(self.vocab[ch])

            except KeyError:
                codes.append(self.unk)

        return codes

    def decode(self, codes):
        text = []
        text_eos = self.vocab["[text_eos]"]

        for code in codes:
            if code in [self.pad, text_eos]:
                break

            text.append(self.inv_vocab[code])

        return "".join(text)

    def encode_coord(self, quantized_coord):
        # assert 0 <= quantized_coord <= self.bin_size - 1

        if quantized_coord < 0 or quantized_coord >= self.bin_size:
            return self.vocab["[coord-out]"]

        return self.vocab[f"[coord-{quantized_coord}]"]

    def encode_coord_xy(self, quantized_coord_x, quantized_coord_y):
        if (
            quantized_coord_x < 0
            or quantized_coord_x >= self.bin_size
            or quantized_coord_y < 0
            or quantized_coord_y >= self.bin_size
        ):
            return self.vocab["[coord-out]"], self.vocab["[coord-out]"]

        return (
            self.vocab[f"[coord-{quantized_coord_x}]"],
            self.vocab[f"[coord-{quantized_coord_y}]"],
        )

    def decode_coord(self, coord_vocab):
        min_coord_id, max_coord_id = 0, self.bin_size - 1
        assert (
            self.vocab[f"[coord-{min_coord_id}]"]
            <= coord_vocab
            <= self.vocab[f"[coord-{max_coord_id}]"]
        )
        return int(self.inv_vocab[coord_vocab][7:-1])

    def encode_order(self, order):
        assert 0 <= order <= self.max_order - 1
        return self.vocab[f"[order-{order}]"]
