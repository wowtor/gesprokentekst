import builtins
import hashlib


class open:
    def __init__(self, path, hashvalue, algorithm, mode='t', encoding='utf8'):
        self.path = path
        self.hashvalue = hashvalue
        if isinstance(algorithm, str):
            self._m = hashlib.new(algorithm)
        else:
            self._m = algorithm
        self._isbinary = 'b' in mode
        self._encoding = encoding

    def __enter__(self):
        self._f = builtins.open(self.path, 'br')
        return self

    def read(self, size=-1):
        b = self._f.read(size)
        self._m.update(b)
        if self._isbinary:
            return b
        else:
            return b.decode(self._encoding)

    def __iter__(self):
        return self.read().splitlines().__iter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None and self._m.hexdigest() != self.hashvalue:
            raise ValueError('digest mismatch')
        self._f.close()


def read_list(path):
    with builtins.open(path) as f:
        for line in f:
            pos = line.find('  ')
            filehash = line[:pos]
            filepath = line[pos+2:-1]
            yield filehash, filepath
