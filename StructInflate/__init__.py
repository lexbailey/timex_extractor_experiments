import sys
import re
import pyparsing as pp
import inspect

def _find_subclasses(module, base):
    return [
        cls for name, cls in inspect.getmembers(module)
            if inspect.isclass(cls) and issubclass(cls, base)
    ]

class InfNoType():
    _name = 'prim'

    def __init__(self, subtype=None, subinfo={}):
        if subtype is not None:
            raise ValueError("Attempted to create '%s' type (%s) with a subtype. (e.g. from string 'prim>list')\nSubtype was %s" %
                (self._name, self.__class__, subtype))
        if subinfo != {}:
            raise ValueError("Attempted to create '%s' type (%s) with subtype info. (e.g. from string 'prim(foo=prim)')\nSubinfo is %r" %
                (self._name, self.__class__, subinfo))

    def __repr__(self):
        return "InfNoType()"

    def construct(self):
        return None

    def convert_key(self, key):
        return key

class InfIter:
    def __init__(self, toplevel, path):
        self.type = toplevel
        self.path = iter(path)

    def __iter__(self):
        return self

    def __next__(self):
        result = self.type
        if isinstance(result, InfNoType):
            raise StopIteration
        key = next(self.path)
        self.type = result.next(key=key)
        return result

class InfType:
    def __init__(self, subtype=InfNoType(), subinfo={}):
        self.subtype = subtype
        self.subinfo = subinfo

    def __getitem__(self, key):
        if key == 0:
            return self
        return self.subtype[key-1]

    def iterate_path(self, path):
        return InfIter(self, path)

    def next(self, key=None):
        return self.subtype

    def get(self, data, key):
        return data[self.convert_key(key)]

    def set(self, data, key, item):
        self.inflate_next(data, key)
        data[self.convert_key(key)] = item

    def _build_type_tree(s):
        type_name = pp.Word( pp.alphas )
        ident = pp.Word( pp.alphas )
        lParen, rParen, eq = map(lambda x: pp.Literal(x).suppress(), '()=')
        type_expr = pp.Forward()
        type_expr <<= pp.delimitedList(
                          pp.Group(
                              type_name + pp.Group(
                                  pp.Optional(lParen + pp.Group(pp.delimitedList(
                                      pp.Group(ident + eq + pp.Group(type_expr))
                                  )) + rParen)
                              )
                          )
                      ,delim='>')
        return type_expr.parseString(s, parseAll=True)


    @classmethod
    def _string_to_type(cls, s):
        subs = _find_subclasses(sys.modules[__name__], cls) + [InfNoType]
        for sub in subs:
            try:
                subname = sub._name
                if subname == s:
                    return sub
            except:
                pass
        raise ValueError('Unable to convert string to inflator type. String was: "%s".\nIf you are using a custom type, make sure you have defined "_name" in your custom type\'s class.' % (s))

    def _build_subinfo_from_tree(wrapped_tree):
        try:
            tree = wrapped_tree[0]
        except IndexError:
            return {}
        return {name: InfType._build_type_from_tree(subtype) for name, subtype in tree}

    def _build_type_from_tree(tree):
        lower = InfNoType()
        first = True
        for item in reversed(tree):
            type_name, sub_info_tree = item

            sub_info = InfType._build_subinfo_from_tree(sub_info_tree)
            if not first or type_name != InfNoType._name:
                lower = InfType._string_to_type(type_name)(subtype=lower, subinfo=sub_info)
            first = False
        return lower

    @classmethod
    def from_string(cls, s):
        tree = cls._build_type_tree(s)
        return cls._build_type_from_tree(tree)

    def __repr__(self):
        if self.subinfo != {}:
            return "%s(%r, subinfo=%r)" % (type(self).__name__, self.subtype, self.subinfo)
        if not isinstance(self.subtype, InfNoType):
            return "%s(%r)" % (type(self).__name__, self.subtype)
        return "%s()" % (type(self).__name__)

class InfDict(InfType):
    _name = 'dict'

    def construct(self):
        return {}

    def convert_key(self, key):
        return key

    def next(self, key=None):
        return self.subinfo.get(key, self.subtype)

    def inflate_next(self, data, key):
        next_type = self.next(key=key)
        new_item = next_type.construct()

        data[self.convert_key(key)] = new_item
        return data[self.convert_key(key)]

class InfList(InfType):
    _name = 'list'

    def construct(self):
        return []

    def convert_key(self, key):
        return int(key)

    def inflate_next(self, data, key):
        index = self.convert_key(key)
        while index >= len(data):
            data.append(self.next().construct())
        return data[index]

class Inflator:
    def __init__(self, out_type):
        if isinstance(out_type, str):
            out_type = InfType.from_string(out_type)
        self.out_type = out_type
        self.data = out_type[0].construct()
        self.placeholder = object()

    def add(self, path, item):
        if isinstance(path, str):
            for c in './>':
                if c in path:
                    path = path.split('.')
                    break
        insert_point = self.data
        last = len(path)-1
        for i, (t, p) in enumerate(zip(self.out_type.iterate_path(path), path)):
            if i == last:
                break
            try:
                insert_point = t.get(insert_point, p)
            except (IndexError, KeyError):
                insert_point = t.inflate_next(insert_point, p)
        if item != self.placeholder:
            t.set(insert_point, p, item)

    def __repr__(self):
        return "Inflator(%s)\n=>%r" % (self.out_type, self.data)

if __name__ == "__main__":
    # Simple first test
    print("Running tests")
    print("Test 1...")
    i1 = Inflator(InfType.from_string('dict>list'))
    i1.add('a.1', 15)
    i1.add('a.0', 12)
    i1.add('b.0', 13)
    i1.add('b.1', 19)
    assert i1.data == {'a': [12, 15], 'b': [13, 19]}
    print(i1)

    # Slightly more complex test with types built from constructors instead of from a string
    print("\n\nTest 2...")
    i2 = Inflator(
        InfList(InfDict(InfDict(subinfo={'ids':InfList(), 'attrs':InfDict()})))
    )
    i2.add('1.left.ids.1', 15)
    i2.add('1.left.ids.2', 7)
    i2.add('1.left.ids.0', 2)
    i2.add('1.left.attrs.max', 15)
    i2.add('1.left.attrs.min', 2)
    i2.add('1.right.ids.0', 6)
    i2.add('1.right.attrs.max', 6)
    i2.add('1.right.attrs.min', 6)
    assert i2.data == [{}, {'left': {'ids': [2, 15, 7], 'attrs': {'max': 15, 'min': 2}}, 'right': {'ids': [6], 'attrs': {'max': 6, 'min': 6}}}]
    print(i2)

    # A test that tests most of the features of the inflator including dicts with primitive values and lists
    print("\n\nTest 3...")
    i3 = Inflator(InfType.from_string(
        '''
             dict(meta=dict)
            >list
            >dict(
                matrix=list
                      >list,
                attr=dict(
                         top=prim,
                         bottom=prim
                     )
                    >list
            )
        '''
    ))
    i3.add('meta.name', 'test_data')
    i3.add('identity.0.matrix.0.0', 1)
    i3.add('identity.0.matrix.0.1', 0)
    i3.add('identity.0.matrix.1.0', 0)
    i3.add('identity.0.matrix.1.1', 1)
    i3.add('identity.0.attr.top', 1)
    i3.add(['identity','0','attr','bottom'], 1) # Throw in one test of a sequence key instead of a string key
    i3.add('identity.0.attr.other.0', 'foo')
    assert i3.data == {'meta': {'name': 'test_data'}, 'identity': [{'matrix': [[1, 0], [0, 1]], 'attr': {'top': 1, 'bottom': 1, 'other': ['foo']}}]}
    print(i3)
    print("\n\nTests complete.")
