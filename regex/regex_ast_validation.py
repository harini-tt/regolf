# regex_ast_validator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any, Tuple, Set
import re
import math

# --------- Config ---------
MAX_PATTERN_LEN = 128          # post-canonicalization, characters
MAX_REPEAT_UPPER = 64          # cap for {m,n} and for static-width upper bounds

# Allowed flags
FLAG_MAP = {"i": re.I, "m": re.M, "s": re.S, "a": re.A}

# --------- AST Nodes ---------
@dataclass
class Node:
    def width(self) -> Tuple[int, Optional[int]]:
        """Return (min_width, max_width) in codepoints; max_width=None means unbounded."""
        raise NotImplementedError

    def stringify(self) -> str:
        """Render to RE string (not canonical by default)."""
        raise NotImplementedError

    def is_atomic_boundary(self) -> bool:
        """Whether this node is an atomic barrier for backtracking analysis."""
        return isinstance(self, Group) and self.kind == "atomic"

@dataclass
class Seq(Node):
    items: List[Node]
    def width(self) -> Tuple[int, Optional[int]]:
        mins, maxs = 0, 0
        unb = False
        for n in self.items:
            mn, mx = n.width()
            mins += mn
            if mx is None:
                unb = True
            else:
                maxs += mx
        return mins, (None if unb else maxs)
    def stringify(self) -> str:
        return "".join(stringify_with_parens_if_needed(child, parent="seq") for child in self.items)

@dataclass
class Alt(Node):
    options: List[Node]
    def width(self) -> Tuple[int, Optional[int]]:
        mins = []
        maxs: List[Optional[int]] = []
        for n in self.options:
            mn, mx = n.width()
            mins.append(mn)
            maxs.append(mx)
        minw = min(mins) if mins else 0
        maxw: Optional[int]
        if any(mx is None for mx in maxs):
            maxw = None
        else:
            maxw = max(mx for mx in maxs) if maxs else 0
        return (minw, maxw)
    def stringify(self) -> str:
        parts = [stringify_with_parens_if_needed(opt, parent="alt_item") for opt in self.options]
        return "|".join(parts)

@dataclass
class Lit(Node):
    value: str
    def width(self) -> Tuple[int, Optional[int]]:
        return (len(self.value), len(self.value))
    def stringify(self) -> str:
        return "".join(escape_literal(ch) for ch in self.value)

@dataclass
class Dot(Node):
    def width(self) -> Tuple[int, Optional[int]]:
        return (1, 1)
    def stringify(self) -> str:
        return "."

@dataclass
class BuiltinClass(Node):
    value: str  # one of d D w W s S
    def width(self) -> Tuple[int, Optional[int]]:
        return (1, 1)
    def stringify(self) -> str:
        return "\\" + self.value

@dataclass
class Anchor(Node):
    kind: str  # start end bos eos word_boundary not_word_boundary
    def width(self) -> Tuple[int, Optional[int]]:
        return (0, 0)
    def stringify(self) -> str:
        return {
            "start": "^",
            "end": "$",
            "bos": r"\A",
            "eos": r"\Z",
            "word_boundary": r"\b",
            "not_word_boundary": r"\B",
        }[self.kind]

@dataclass
class CharClassItem:
    kind: str    # 'char' | 'range' | 'predef'
    value: Optional[str] = None
    cfrom: Optional[str] = None
    cto: Optional[str] = None

@dataclass
class CharClass(Node):
    negated: bool
    items: List[CharClassItem]
    def width(self) -> Tuple[int, Optional[int]]:
        return (1, 1)
    def stringify(self) -> str:
        items = canonicalize_charclass_items(self.items)
        inner = "".join(stringify_charclass_item(it) for it in items)
        return "[" + ("^" if self.negated else "") + inner + "]"

@dataclass
class Group(Node):
    kind: str    # 'noncap' | 'atomic'
    child: Node
    def width(self) -> Tuple[int, Optional[int]]:
        return self.child.width()
    def stringify(self) -> str:
        if self.kind == "noncap":
            return "(?:" + self.child.stringify() + ")"
        elif self.kind == "atomic":
            return "(?>" + self.child.stringify() + ")"
        else:
            raise ValueError("Unknown group kind")

@dataclass
class Look(Node):
    kind: str    # 'ahead' | 'ahead_not' | 'behind' | 'behind_not'
    child: Node
    def width(self) -> Tuple[int, Optional[int]]:
        return (0, 0)  # zero-width assertion
    def stringify(self) -> str:
        tag = {"ahead":"?=", "ahead_not":"?!", "behind":"?<=", "behind_not":"?<!"}[self.kind]
        return "(" + tag + self.child.stringify() + ")"

@dataclass
class Repeat(Node):
    child: Node
    min: int
    max: Optional[int]  # None means unbounded
    possessive: bool
    def width(self) -> Tuple[int, Optional[int]]:
        mn, mx = self.child.width()
        minw = mn * self.min
        if self.max is None or mx is None:
            maxw = None
        else:
            maxw = mx * self.max
        return (minw, maxw)
    def stringify(self) -> str:
        body = stringify_with_parens_if_needed(self.child, parent="repeat")
        suf = quant_suffix(self.min, self.max)
        if self.possessive:
            suf += "+"
        return body + suf

NodeLike = Union[Seq, Alt, Lit, Dot, BuiltinClass, Anchor, CharClass, Group, Look, Repeat]

# --------- JSON -> AST parsing ---------
def node_from_json(d: Dict[str, Any]) -> Node:
    t = expect_type(d, "type", str)
    if t == "seq":
        items = [node_from_json(x) for x in expect_type(d, "items", list)]
        assert items, "seq requires at least 1 item"
        return Seq(items)
    if t == "alt":
        opts = [node_from_json(x) for x in expect_type(d, "options", list)]
        assert len(opts) >= 2, "alt requires at least 2 options"
        return Alt(opts)
    if t == "lit":
        return Lit(expect_type(d, "value", str))
    if t == "dot":
        return Dot()
    if t == "bclass":
        v = expect_type(d, "value", str)
        assert v in {"d","D","w","W","s","S"}
        return BuiltinClass(v)
    if t == "anchor":
        kind = expect_type(d, "kind", str)
        assert kind in {"start","end","bos","eos","word_boundary","not_word_boundary"}
        return Anchor(kind)
    if t == "charclass":
        neg = bool(d.get("negated", False))
        raw_items = expect_type(d, "items", list)
        items: List[CharClassItem] = []
        for it in raw_items:
            k = expect_type(it, "type", str)
            if k == "char":
                v = expect_type(it, "value", str)
                assert len(v) == 1
                items.append(CharClassItem(kind="char", value=v))
            elif k == "range":
                f = expect_type(it, "from", str); t2 = expect_type(it, "to", str)
                assert len(f) == len(t2) == 1
                items.append(CharClassItem(kind="range", cfrom=f, cto=t2))
            elif k == "predef":
                v = expect_type(it, "value", str)
                assert v in {"d","D","w","W","s","S"}
                items.append(CharClassItem(kind="predef", value=v))
            else:
                raise ValueError(f"Unknown charclass item type: {k}")
        return CharClass(negated=neg, items=items)
    if t == "group":
        kind = expect_type(d, "kind", str)
        assert kind in {"noncap","atomic"}
        child = node_from_json(expect_type(d, "child", dict))
        return Group(kind=kind, child=child)
    if t == "look":
        kind = expect_type(d, "kind", str)
        assert kind in {"ahead","ahead_not","behind","behind_not"}
        child = node_from_json(expect_type(d, "child", dict))
        return Look(kind=kind, child=child)
    if t == "repeat":
        child = node_from_json(expect_type(d, "child", dict))
        mn = expect_type(d, "min", int)
        mx = d.get("max", None)
        if mx is not None:
            mx = expect_type(d, "max", int)
        poss = bool(d.get("possessive", False))
        assert mn >= 0
        if mx is not None:
            assert mx >= mn
        return Repeat(child=child, min=mn, max=mx, possessive=poss)
    raise ValueError(f"Unknown node type: {t}")

def expect_type(d: Dict[str, Any], k: str, tp):
    if k not in d:
        raise ValueError(f"Missing key '{k}'")
    v = d[k]
    if not isinstance(v, tp):
        raise ValueError(f"Field '{k}' expected {tp}, got {type(v)}")
    return v

# --------- Validation (safety) ---------
class ValidationError(Exception):
    pass

def validate_solution(obj: Dict[str, Any]) -> Tuple[Node, int]:
    # flags
    flags_str = obj.get("flags", "")
    if not isinstance(flags_str, str):
        raise ValidationError("flags must be a string")
    invalid = set(flags_str) - set(FLAG_MAP)
    if invalid:
        raise ValidationError(f"invalid flags: {invalid}")
    flags_val = 0
    for ch in sorted(set(flags_str)):  # dedupe
        flags_val |= FLAG_MAP[ch]

    # unsat short-circuit
    if obj.get("unsat", False):
        return Seq([Lit("")]), flags_val  # dummy pattern

    if "ast" not in obj or not isinstance(obj["ast"], dict):
        raise ValidationError("missing 'ast' object")

    ast = node_from_json(obj["ast"])
    # run safety checks
    check_no_repeat_of_empty(ast)
    check_fixed_width_lookbehind(ast)
    check_repeat_safety(ast)
    check_repeat_caps(ast)

    # canonicalize & length check
    ast_c = canonicalize(ast)
    pat = ast_c.stringify()
    if len(pat) > MAX_PATTERN_LEN:
        raise ValidationError(f"pattern too long: {len(pat)} > {MAX_PATTERN_LEN}")

    # ensure anchors/lookarounds are not repeated (already enforced, but double-check)
    ensure_no_repeat_zero_width(ast_c)

    return ast_c, flags_val

def check_no_repeat_of_empty(node: Node) -> Tuple[int, Optional[int]]:
    """Ensure no Repeat wraps an empty-width-capable child."""
    def rec(n: Node):
        if isinstance(n, Repeat):
            mn, mx = n.child.width()
            if mn == 0:
                raise ValidationError("repetition of empty-width term is disallowed")
            rec(n.child)
        elif isinstance(n, (Seq, Alt, Group, Look, CharClass)):
            for ch in children(n):
                rec(ch)
        # Anchors, Dot, BuiltinClass, Lit handled by width/min == 0 rule
    rec(node)

def check_fixed_width_lookbehind(node: Node):
    def rec(n: Node):
        if isinstance(n, Look) and n.kind in {"behind","behind_not"}:
            mn, mx = n.child.width()
            if mx is None or mn != mx:
                raise ValidationError("lookbehind must be fixed-width")
        for ch in children(n):
            rec(ch)
    rec(node)

def check_repeat_caps(node: Node):
    def rec(n: Node):
        if isinstance(n, Repeat):
            if n.max is not None and n.max > MAX_REPEAT_UPPER:
                raise ValidationError(f"repeat upper bound too high: {n.max} > {MAX_REPEAT_UPPER}")
            rec(n.child)
        for ch in children(n):
            rec(ch)
    rec(node)

def check_repeat_safety(node: Node):
    """Two rules:
    (A) If repeating an alternation, require possessive OR make it an atomic group.
    (B) No nested variable-width repetition unless protected by atomic group on inner path OR outer is possessive.
    """
    def has_var_repeat(n: Node, atomic_guard: bool) -> bool:
        if isinstance(n, Group) and n.kind == "atomic":
            atomic_guard = True
        if isinstance(n, Repeat):
            mn, mx = n.child.width()
            var = (mx is None) or (mn != mx)
            if var and not atomic_guard:
                return True
            return has_var_repeat(n.child, atomic_guard)
        for ch in children(n):
            if has_var_repeat(ch, atomic_guard):
                return True
        return False

    def rec(n: Node, in_atomic: bool):
        if isinstance(n, Group) and n.kind == "atomic":
            in_atomic = True
        if isinstance(n, Repeat):
            # (A) alternation under repeat
            child = n.child
            if isinstance(child, Alt):
                if not n.possessive and not in_atomic:
                    # allow if explicitly wrapped in atomic group
                    raise ValidationError("repeating alternation requires atomic group or possessive quantifier")
            # (B) nested variable-width repetition
            if has_var_repeat(n.child, in_atomic) and not (n.possessive or in_atomic):
                raise ValidationError("nested variable-width repetition must be atomic or possessive")
            rec(n.child, in_atomic)
        else:
            for ch in children(n):
                rec(ch, in_atomic)
    rec(node, in_atomic=False)

def ensure_no_repeat_zero_width(node: Node):
    def rec(n: Node):
        if isinstance(n, Repeat):
            mn, _ = n.child.width()
            if mn == 0:
                raise ValidationError("zero-width term under repetition")
            rec(n.child)
        else:
            for ch in children(n):
                rec(ch)
    rec(node)

def children(n: Node) -> List[Node]:
    if isinstance(n, Seq):
        return n.items
    if isinstance(n, Alt):
        return n.options
    if isinstance(n, Group):
        return [n.child]
    if isinstance(n, Look):
        return [n.child]
    if isinstance(n, Repeat):
        return [n.child]
    if isinstance(n, CharClass):
        return []
    return []

# --------- Canonicalization ---------
def canonicalize(n: Node) -> Node:
    """Lightweight canonicalization for deterministic length/format."""
    if isinstance(n, Seq):
        items = [canonicalize(x) for x in n.items]
        # drop redundant noncap groups around a single non-alt item
        flat: List[Node] = []
        for x in items:
            if isinstance(x, Group) and x.kind == "noncap" and not contains_alt(x.child):
                flat.append(x.child)
            else:
                flat.append(x)
        return Seq(flat)
    if isinstance(n, Alt):
        opts = [canonicalize(x) for x in n.options]
        # remove duplicates by string form, then sort
        uniq = {}
        for o in opts:
            uniq[o.stringify()] = o
        opts2 = [uniq[k] for k in sorted(uniq.keys())]
        # collapse single-char literals a|b|c -> [abc]
        if all(isinstance(o, Lit) and len(o.value) == 1 for o in opts2):
            chars = [o.value for o in opts2]
            return CharClass(negated=False, items=[CharClassItem(kind="char", value=c) for c in chars])
        return Alt(opts2)
    if isinstance(n, Group):
        return Group(n.kind, canonicalize(n.child))
    if isinstance(n, Look):
        return Look(n.kind, canonicalize(n.child))
    if isinstance(n, Repeat):
        ch = canonicalize(n.child)
        return Repeat(ch, n.min, n.max, n.possessive)
    if isinstance(n, CharClass):
        return CharClass(n.negated, canonicalize_charclass_items(n.items))
    return n  # Lit, Dot, BuiltinClass, Anchor

def contains_alt(n: Node) -> bool:
    if isinstance(n, Alt):
        return True
    return any(contains_alt(c) for c in children(n))

def canonicalize_charclass_items(items: List[CharClassItem]) -> List[CharClassItem]:
    chars: Set[str] = set()
    ranges: List[Tuple[str,str]] = []
    predefs: Set[str] = set()
    for it in items:
        if it.kind == "char":
            chars.add(it.value)
        elif it.kind == "range":
            a, b = it.cfrom, it.cto
            if ord(a) <= ord(b):
                ranges.append((a,b))
            else:
                ranges.append((b,a))
        elif it.kind == "predef":
            predefs.add(it.value)
    # merge and sort ranges
    ranges = merge_ranges(sorted(ranges, key=lambda ab: (ab[0], ab[1])))
    # sort chars
    chars_sorted = sorted(chars)
    # rebuild
    out: List[CharClassItem] = []
    for a,b in ranges:
        out.append(CharClassItem(kind="range", cfrom=a, cto=b))
    for c in chars_sorted:
        out.append(CharClassItem(kind="char", value=c))
    for p in sorted(predefs):  # keep predefs last
        out.append(CharClassItem(kind="predef", value=p))
    return out

def merge_ranges(ranges: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    if not ranges: return []
    merged = [ranges[0]]
    for a,b in ranges[1:]:
        la, lb = merged[-1]
        if ord(a) <= ord(lb) + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a,b))
    return merged

# --------- Rendering helpers ---------
META = set(r".^$*+?{}[]()|\\")

def escape_literal(ch: str) -> str:
    if ch == "\t": return r"\t"
    if ch == "\n": return r"\n"
    if ch == "\r": return r"\r"
    if ch in META: return "\\" + ch
    return ch

def stringify_charclass_item(it: CharClassItem) -> str:
    if it.kind == "char":
        v = it.value
        if v in r"]\-^\\":  # escape specials within classes
            return "\\" + v
        if v == "\t": return r"\t"
        if v == "\n": return r"\n"
        if v == "\r": return r"\r"
        return v
    if it.kind == "range":
        a = stringify_charclass_item(CharClassItem(kind="char", value=it.cfrom))
        b = stringify_charclass_item(CharClassItem(kind="char", value=it.cto))
        return f"{a}-{b}"
    if it.kind == "predef":
        return "\\" + it.value
    raise ValueError("bad charclass item")

def quant_suffix(mn: int, mx: Optional[int]) -> str:
    if mn == 0 and mx == 1: return "?"
    if mn == 1 and mx is None: return "+"
    if mn == 0 and mx is None: return "*"
    if mx is None:
        return "{" + str(mn) + ",}"
    if mn == mx:
        return "{" + str(mn) + "}"
    return "{" + str(mn) + "," + str(mx) + "}"

def stringify_with_parens_if_needed(n: Node, parent: str) -> str:
    """Insert grouping when required by precedence."""
    s = n.stringify()
    need = False
    if isinstance(n, Alt):
        need = parent in {"seq", "repeat", "alt_item"}
    elif isinstance(n, Seq):
        need = parent in {"repeat"}
    elif isinstance(n, Look) or isinstance(n, Anchor):
        need = parent in {"repeat"}  # though repeats of zero-width are invalid anyway
    if need:
        # noncapturing group for structure
        return "(?:" + s + ")"
    return s

# --------- Public API ---------
def compile_from_model_json(solution: Dict[str, Any]) -> Tuple[str, int, re.Pattern]:
    """
    Validates, canonicalizes and compiles a model solution.
    Returns (pattern_str, flags_value, compiled_pattern).
    Raises ValidationError on failure.
    """
    ast, flags_val = validate_solution(solution)
    pat = ast.stringify()
    # final sanity: compile
    try:
        compiled = re.compile(pat, flags_val)
    except re.error as e:
        raise ValidationError(f"re.compile failed: {e}") from e
    return pat, flags_val, compiled

# --------- (Optional) demo ----------
if __name__ == "__main__":
    # Minimal example: ^(?:cat|dog)+$  (as atomic/possessive safe form)
    model_out = {
        "flags": "i",
        "unsat": False,
        "ast": {
            "type": "seq",
            "items": [
                {"type":"anchor","kind":"start"},
                {"type":"repeat","possessive":True,"min":1,"max":None,
                 "child":{"type":"group","kind":"atomic",
                          "child":{"type":"alt","options":[
                              {"type":"lit","value":"cat"},
                              {"type":"lit","value":"dog"}
                          ]}}},
                {"type":"anchor","kind":"end"}
            ]
        }
    }
    pat, fval, creg = compile_from_model_json(model_out)
    print("Pattern:", pat, " Flags:", fval)
    assert creg.fullmatch("dogcat") and not creg.fullmatch("doge")
