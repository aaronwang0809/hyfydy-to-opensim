#!/usr/bin/env python3
"""
Conversion.py — Hyfydy (.hfd) -> OpenSim (.osim) converter (best-effort)

Usage:
  python Conversion.py --in model.hfd --out model.osim
  python Conversion.py --in model.hfd --out model.osim --dump-json parsed.json
  python Conversion.py --in model.hfd --out model.osim --verbose

What it does well:
- Parses Hyfydy-ish .hfd text structure:
    - key = value
    - blocks: name { ... }
    - arrays: [ 1 2 3 ]
    - # line comments and /* block comments */
- Extracts model name / gravity, bodies, joints (best-effort).
- Writes a valid OpenSim XML skeleton + body/joint sets.

What you'll extend later:
- Muscle/actuator mapping (ForceSet)
- Wrap objects, geometry, contacts, constraints, controllers
- Precise joint frame math / orientations

This file is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


# =========================
# Tokenizer / Parser
# =========================

_TOKEN_RE = re.compile(
    r"""
    (?P<WS>\s+)
  | (?P<COMMENT_HASH>\#.*?$)
  | (?P<LBRACE>\{)
  | (?P<RBRACE>\})
  | (?P<LBRACK>\[)
  | (?P<RBRACK>\])
  | (?P<EQUAL>=)
  | (?P<DOTS>\.\.)
  | (?P<STRING>"([^"\\]|\\.)*")
  | (?P<NUMBER>[-+]?\d+(\.\d+)?([eE][-+]?\d+)?)
  | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<MISC>.)
    """,
    re.VERBOSE | re.MULTILINE,
)


@dataclass
class Token:
    kind: str
    value: str
    pos: int


def _strip_block_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def tokenize(text: str) -> List[Token]:
    text = _strip_block_comments(text)
    toks: List[Token] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup or "MISC"
        val = m.group(kind)
        if kind in ("WS", "COMMENT_HASH"):
            continue
        if kind == "MISC":
            raise SyntaxError(f"Unexpected character {val!r} at position {m.start()}")
        toks.append(Token(kind=kind, value=val, pos=m.start()))
    return toks


class Parser:
    """
    Parses a Hyfydy-ish .hfd grammar:

    - assignments: key = value
    - blocks:      key { ... }
    - arrays:      [ v1 v2 ... ]
    - values:      number | "string" | identifier | array | range (a..b inside array)
    """

    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Optional[Token]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def pop(self, kind: Optional[str] = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of file")
        if kind and tok.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {tok.kind} ({tok.value}) at {tok.pos}")
        self.i += 1
        return tok

    def parse(self) -> Dict[str, Any]:
        root: Dict[str, Any] = {}
        while self.peek() is not None:
            tok = self.peek()
            if tok.kind != "IDENT":
                raise SyntaxError(f"Expected IDENT at top-level, got {tok.kind} ({tok.value}) at {tok.pos}")
            name = self.pop("IDENT").value

            nxt = self.peek()
            if nxt and nxt.kind == "EQUAL":
                self.pop("EQUAL")
                val = self.parse_value()
                root[name] = val
            elif nxt and nxt.kind == "LBRACE":
                obj = self.parse_block()
                self._add_block(root, name, obj)
            else:
                raise SyntaxError(f"Unexpected token after {name}: {nxt.kind if nxt else None}")
        return root

    def parse_block(self) -> Dict[str, Any]:
        self.pop("LBRACE")
        obj: Dict[str, Any] = {}
        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed block (missing '}')")
            if tok.kind == "RBRACE":
                self.pop("RBRACE")
                return obj

            if tok.kind != "IDENT":
                raise SyntaxError(f"Expected IDENT in block, got {tok.kind} ({tok.value}) at {tok.pos}")
            key = self.pop("IDENT").value

            nxt = self.peek()
            if nxt and nxt.kind == "EQUAL":
                self.pop("EQUAL")
                val = self.parse_value()
                obj[key] = val
            elif nxt and nxt.kind == "LBRACE":
                child = self.parse_block()
                self._add_block(obj, key, child)
            else:
                raise SyntaxError(f"Unexpected token after key {key}: {nxt.kind if nxt else None} at {tok.pos}")

    def parse_value(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Expected value, got EOF")

        # STRING
        if tok.kind == "STRING":
            s = self.pop("STRING").value
            return bytes(s[1:-1], "utf-8").decode("unicode_escape")

        # NUMBER or RANGE
        if tok.kind == "NUMBER":
            v1_tok = self.pop("NUMBER")
            # Check for range syntax: a .. b
            if self.peek() and self.peek().kind == "DOTS":
                self.pop("DOTS")
                v2_tok = self.pop("NUMBER")
                v1 = int(v1_tok.value) if re.fullmatch(r"[-+]?\d+", v1_tok.value) else float(v1_tok.value)
                v2 = int(v2_tok.value) if re.fullmatch(r"[-+]?\d+", v2_tok.value) else float(v2_tok.value)
                return (v1, v2)
            else:
                n = v1_tok.value
                return int(n) if re.fullmatch(r"[-+]?\d+", n) else float(n)

        # IDENT
        if tok.kind == "IDENT":
            return self.pop("IDENT").value

        # ARRAY
        if tok.kind == "LBRACK":
            return self.parse_array()

        raise SyntaxError(f"Unsupported value token: {tok.kind} ({tok.value}) at {tok.pos}")

    def parse_array(self) -> List[Any]:
        self.pop("LBRACK")
        arr: List[Any] = []
        while True:
            tok = self.peek()
            if tok is None:
                raise SyntaxError("Unclosed array (missing ']')")
            if tok.kind == "RBRACK":
                self.pop("RBRACK")
                return arr

            if tok.kind == "NUMBER":
                v1_tok = self.pop("NUMBER")
                if self.peek() and self.peek().kind == "DOTS":
                    self.pop("DOTS")
                    v2_tok = self.pop("NUMBER")
                    v1 = int(v1_tok.value) if re.fullmatch(r"[-+]?\d+", v1_tok.value) else float(v1_tok.value)
                    v2 = int(v2_tok.value) if re.fullmatch(r"[-+]?\d+", v2_tok.value) else float(v2_tok.value)
                    arr.append((v1, v2))  # range
                else:
                    v1 = int(v1_tok.value) if re.fullmatch(r"[-+]?\d+", v1_tok.value) else float(v1_tok.value)
                    arr.append(v1)
                continue

            if tok.kind in ("STRING", "IDENT"):
                arr.append(self.parse_value())
                continue

            raise SyntaxError(f"Unexpected token in array: {tok.kind} ({tok.value}) at {tok.pos}")

    @staticmethod
    def _add_block(obj: Dict[str, Any], key: str, child: Dict[str, Any]) -> None:
        # store repeated blocks as lists
        if key in obj:
            if isinstance(obj[key], list):
                obj[key].append(child)
            else:
                obj[key] = [obj[key], child]
        else:
            obj[key] = [child]


# =========================
# Helpers for mapping
# =========================

def as_blocks(d: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    v = d.get(key, [])
    return v if isinstance(v, list) else []


def as_vec3(v: Any, default: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return float(v[0]), float(v[1]), float(v[2])
    return default


def vec3_str(v: Tuple[float, float, float]) -> str:
    return f"{v[0]} {v[1]} {v[2]}"


def inertia6_from_any(v: Any) -> str:
    """
    OpenSim expects 6 values: Ixx Iyy Izz Ixy Ixz Iyz.
    Hyfydy commonly uses 3 diag values [Ixx Iyy Izz].
    """
    if isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return f"{v[0]} {v[1]} {v[2]} 0 0 0"
    if isinstance(v, list) and len(v) == 6 and all(isinstance(x, (int, float)) for x in v):
        return " ".join(str(x) for x in v)
    return "1 1 1 0 0 0"


def pretty_indent(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            pretty_indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# =========================
# OpenSim writer (best-effort)
# =========================

def choose_joint_type(j: Dict[str, Any]) -> str:
    """
    Heuristics:
      - if j['dof'] looks like 0 => WeldJoint
      - if dof=1 => PinJoint
      - if dof=3 => BallJoint
      - if j has a 'type' field we try to honor common names.
    """
    jtype = str(j.get("type", "")).lower()
    if jtype in ("weld", "weldjoint", "fixed"):
        return "WeldJoint"
    if jtype in ("pin", "hinge", "revolute", "pinjoint"):
        return "PinJoint"
    if jtype in ("ball", "balljoint", "spherical"):
        return "BallJoint"

    dof = j.get("dof", None)
    if isinstance(dof, (int, float)):
        if int(dof) == 0:
            return "WeldJoint"
        if int(dof) == 1:
            return "PinJoint"
        if int(dof) == 3:
            return "BallJoint"

    # fallback
    return "PinJoint"


def build_osim(parsed: Dict[str, Any], verbose: bool = False) -> ET.Element:
    model_blocks = as_blocks(parsed, "model")
    if not model_blocks:
        raise ValueError("No top-level `model { ... }` block found in .hfd.")
    hmodel = model_blocks[0]

    model_name = str(hmodel.get("name", "ConvertedModel"))
    gravity = as_vec3(hmodel.get("gravity", [0, -9.81, 0]), default=(0.0, -9.81, 0.0))

    # Root
    osim = ET.Element("OpenSimDocument", {"Version": "40000"})
    model = ET.SubElement(osim, "Model", {"name": model_name})
    ET.SubElement(model, "gravity").text = vec3_str(gravity)

    # ---- Bodies ----
    bodyset = ET.SubElement(model, "BodySet")
    bodies_el = ET.SubElement(bodyset, "objects")

    # Add a ground body placeholder (OpenSim has Ground implicitly; this keeps references stable)
    ground = ET.SubElement(bodies_el, "Body", {"name": "ground"})
    ET.SubElement(ground, "mass").text = "0"
    ET.SubElement(ground, "mass_center").text = "0 0 0"
    ET.SubElement(ground, "inertia").text = "0 0 0 0 0 0"

    bodies = as_blocks(hmodel, "body")
    if verbose:
        print(f"[info] Found {len(bodies)} body blocks")

    for b in bodies:
        bname = str(b.get("name", "unnamed_body"))
        mass = b.get("mass", 1.0)
        com = as_vec3(b.get("com", b.get("mass_center", [0, 0, 0])), default=(0.0, 0.0, 0.0))
        inertia6 = inertia6_from_any(b.get("inertia", None))

        body_el = ET.SubElement(bodies_el, "Body", {"name": bname})
        ET.SubElement(body_el, "mass").text = str(mass)
        ET.SubElement(body_el, "mass_center").text = vec3_str(com)
        ET.SubElement(body_el, "inertia").text = inertia6

    # ---- Joints ----
    jointset = ET.SubElement(model, "JointSet")
    joints_el = ET.SubElement(jointset, "objects")

    joints = as_blocks(hmodel, "joint")
    if verbose:
        print(f"[info] Found {len(joints)} joint blocks")

    for j in joints:
        jname = str(j.get("name", "joint"))
        parent = str(j.get("parent", "ground"))
        child = str(j.get("child", "unnamed_body"))

        # Hyfydy files vary; we try common names for joint frames/locations
        loc_parent = as_vec3(j.get("parent_pos", j.get("location_in_parent", [0, 0, 0])))
        loc_child = as_vec3(j.get("child_pos", j.get("location_in_child", [0, 0, 0])))

        # Orientations in OpenSim are typically XYZ body-fixed Euler angles (radians or degrees depending on model settings).
        # We keep 0 0 0 unless you supply something explicit.
        ori_parent = as_vec3(j.get("parent_ori", j.get("orientation_in_parent", [0, 0, 0])))
        ori_child = as_vec3(j.get("child_ori", j.get("orientation_in_child", [0, 0, 0])))

        joint_type = choose_joint_type(j)
        jel = ET.SubElement(joints_el, joint_type, {"name": jname})

        ET.SubElement(jel, "parent_body").text = parent
        ET.SubElement(jel, "child_body").text = child
        ET.SubElement(jel, "location_in_parent").text = vec3_str(loc_parent)
        ET.SubElement(jel, "orientation_in_parent").text = vec3_str(ori_parent)
        ET.SubElement(jel, "location_in_child").text = vec3_str(loc_child)
        ET.SubElement(jel, "orientation_in_child").text = vec3_str(ori_child)

    # ---- Forces / Muscles placeholder ----
    forceset = ET.SubElement(model, "ForceSet")
    ET.SubElement(forceset, "objects")

    # ---- Controllers placeholder ----
    controllers = ET.SubElement(model, "ControllerSet")
    ET.SubElement(controllers, "objects")

    return osim


# =========================
# Main
# =========================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input .hfd")
    ap.add_argument("--out", dest="out", required=True, help="Output .osim")
    ap.add_argument("--dump-json", dest="dump_json", default=None, help="Write parsed .hfd as JSON to this file")
    ap.add_argument("--verbose", action="store_true", help="Print extra info")
    args = ap.parse_args()

    try:
        with open(args.inp, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"[error] Input file not found: {args.inp}", file=sys.stderr)
        return 2

    try:
        tokens = tokenize(text)
        parsed = Parser(tokens).parse()
    except Exception as e:
        print(f"[error] Failed to parse .hfd: {e}", file=sys.stderr)
        return 3

    if args.dump_json:
        try:
            with open(args.dump_json, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
            if args.verbose:
                print(f"[info] Wrote parsed JSON: {args.dump_json}")
        except Exception as e:
            print(f"[warn] Could not write JSON dump: {e}", file=sys.stderr)

    try:
        osim_root = build_osim(parsed, verbose=args.verbose)
        pretty_indent(osim_root)
        ET.ElementTree(osim_root).write(args.out, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        print(f"[error] Failed to build/write .osim: {e}", file=sys.stderr)
        return 4

    print(f"Converted {args.inp} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())