import re
import xml.etree.ElementTree as ET
from typing import Set, Dict, Optional

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

SHAPE_TAGS = [
    f"{{{SVG_NS}}}path",
    f"{{{SVG_NS}}}polygon",
    f"{{{SVG_NS}}}rect",
    f"{{{SVG_NS}}}circle",
    f"{{{SVG_NS}}}ellipse",
]

_RGB_RE = re.compile(
    r"rgba?\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*(?:,\s*([0-9.]+)\s*)?\)",
    re.I
)

def _parse_style(style_str: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not style_str:
        return out
    for part in style_str.split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out

def _normalize_hex(h: str) -> Optional[str]:
    if not h:
        return None
    s = h.strip().lower()
    if s.startswith("#"):
        s = s[1:]
        if len(s) == 3 and all(c in "0123456789abcdef" for c in s):
            s = "".join(ch * 2 for ch in s)  # #abc -> #aabbcc
        if len(s) == 6 and re.fullmatch(r"[0-9a-f]{6}", s):
            return f"#{s}"
    return None

def _normalize_color(c: Optional[str]) -> Optional[str]:
    """색을 #rrggbb 로 정규화. 알 수 없는 형태/none/gradient는 None."""
    if not c:
        return None
    s = c.strip().lower()
    if s in ("none", "transparent"):
        return None
    if s.startswith("url("):  # gradient/pattern
        return None

    hx = _normalize_hex(s)
    if hx:
        return hx

    m = _RGB_RE.fullmatch(s)
    if m:
        r, g, b = (max(0, min(255, int(m.group(i)))) for i in (1, 2, 3))
        return f"#{r:02x}{g:02x}{b:02x}"

    # 이름색(red/blue 등)은 그대로 반환(원하면 webcolors로 변환 가능)
    return s

def _build_parent_map(root: ET.Element) -> Dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}

def _resolve_fill(el: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> Optional[str]:
    """요소→상위 그룹을 거슬러 올라가며 fill을 결정(투명도 0이면 제외)."""
    def get_fill_from(e: ET.Element) -> Optional[str]:
        st = _parse_style(e.get("style", ""))
        f = st.get("fill") or e.get("fill")
        # 투명도 0은 제외
        op = st.get("fill-opacity", e.get("fill-opacity"))
        if op is not None:
            try:
                if float(op) == 0.0:
                    return None
            except ValueError:
                pass
        return f

    cur: Optional[ET.Element] = el
    while cur is not None:
        f = get_fill_from(cur)
        if f and f != "inherit":
            return _normalize_color(f)
        cur = parent_map.get(cur)
    # SVG 기본값은 'black'이지만, 명시 없으면 제외하고 싶다면 None 유지
    return None

def extract_svg_fills_from_bytes(svg_bytes: bytes) -> Set[str]:
    root = ET.fromstring(svg_bytes)
    parent_map = _build_parent_map(root)
    colors: Set[str] = set()

    for tag in SHAPE_TAGS:
        for el in root.findall(f".//{tag}"):
            col = _resolve_fill(el, parent_map)
            if col:
                colors.add(col)
    return colors

def extract_svg_fills_from_file(path: str) -> Set[str]:
    with open(path, "rb") as f:
        svg_bytes = f.read()
    return extract_svg_fills_from_bytes(svg_bytes)

colors = extract_svg_fills_from_file("KakaoTalk_20251028_014558453_02.svg")
print(len(colors), colors)