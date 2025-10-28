import io
from PIL import Image, ImageOps
import streamlit as st
from vtracer import convert_raw_image_to_svg

st.set_page_config(layout="wide")

st.header("image2svg")

with st.expander("세부 조절"):

    col1, col2, col3, col4 = st.columns(4, border=True)

    with col1:
        st.subheader("`colormode`")
        st.write("- `color`: 색상을 유지하며 벡터화")
        st.write("- `binary`: 흑백(단색)으로 벡터화")
        st.selectbox(" ", ["color", "binary"], 0, label_visibility="collapsed", key="colormode")

    with col2:
        st.subheader("`hierarchical`")
        st.write("- `stacked`: 각 색을 레이어처럼 위에 쌓는 방식 (일반적인 색칠 도안에 가까움)")
        st.write("- `cutout`: 위 색이 아래 색을 잘라내는 방식 (인쇄/레이저커팅 같은 용도에서 유용)")
        st.selectbox(" ", ["stacked", "cutout"], 0, label_visibility="collapsed", key="hierarchical")

    with col3:
        st.subheader("`mode`")
        st.write("- `spline`: 곡선(Bézier spline)으로 부드럽게 표현")
        st.write("- `polygon`: 직선 다각형 기반 (노드 수 많음, 각짐)")
        st.write("- `none`: 원시 픽셀 경계 그대로 (거칠지만 원본 충실")
        st.selectbox(" ", ["spline", "polygon", "none"], 0, label_visibility="collapsed", key="mode")

    with col4:
        st.subheader("`filter_speckle`")
        st.write("- 작은 점(픽셀 덩어리)을 무시하는 크기 기준")
        st.write("- 값 ↑ → 작은 점 무시, 결과 깔끔 / 값 ↓ → 작은 점도 보존")
        #st.slider(" ", 0, 10, 4, 1)
        st.number_input(" ", 0, 100, 4, 1, label_visibility="collapsed", key="filter_speckle")

    ###################
    col1, col2, col3, col4 = st.columns(4, border=True)

    with col1:
        st.subheader("`color_precision`")
        st.write("- 색상 구분 정밀도")
        st.write("- 값 ↑ → 색을 더 세밀히 나눔 (색 수 ↑)")
        st.write("- 값 ↓ → 색 단순화 (색 수 ↓, “페인트 by 넘버” 스타일)")
        st.number_input(" ", 0, 100, 6, 1, label_visibility="collapsed", key="color_precision")

    with col2:
        st.subheader("`layer_difference`")
        st.write("- 인접한 색 레이어를 다른 레이어로 분리할지 합칠지 결정하는 기준")
        st.write("- 값 ↑ → 색상 차이가 작아도 분리됨 (색상 세분화 ↑)")
        st.write("- 값 ↓ → 색상 차이가 작으면 같은 레이어로 합쳐짐")
        st.number_input(" ", 0, 100, 16, 1, label_visibility="collapsed", key="layer_difference")

    with col3:
        st.subheader("`corner_threshold`")
        st.write("- 코너(꺾이는 지점)를 검출할 민감도")
        st.write("- 값 ↑ → 더 많은 지점을 코너로 인식 (각짐)")
        st.write("- 값 ↓ → 둥글게 단순화")
        st.number_input(" ", 0, 100, 60, 1, label_visibility="collapsed", key="corner_threshold")

    with col4:
        st.subheader("`length_threshold`")
        st.write("- 선분 길이 기준")
        st.write("- 값 ↑ → 짧은 선분을 무시 (단순화 ↑)")
        st.write("- 값 ↓ → 짧은 선분도 유지 (세밀 ↑)")
        st.number_input(" ", 3.5, 10.0, 4.0, 0.1, label_visibility="collapsed", key="length_threshold")

    ###################

    col1, col2, col3, col4 = st.columns(4, border=True)

    with col1:
        st.subheader("`max_iterations`")
        st.write("- 경로 단순화를 반복하는 횟수 (최적화 정도)")
        st.write("- 값 ↑ → 더 매끈하지만 원본과 차이 ↑")
        st.number_input(" ", 0, 100, 10, 1, label_visibility="collapsed", key="max_iterations")

    with col2:
        st.subheader("`splice_threshold`")
        st.write("- 경로를 이어붙일 때 각도 기준")
        st.write("- 값 ↑ → 큰 각도 차이도 연결 (부드럽게)")
        st.write("- 값 ↓ → 각이 크면 따로 분리 (각진 결과)")
        st.number_input(" ", 0, 100, 45, 1, label_visibility="collapsed", key="splice_threshold")

    with col3:
        st.subheader("`path_precision`")
        st.write("- 최종 SVG 경로의 세밀도")
        st.write("- 값 ↑ → 원본에 더 가까운 경로, 노드 많음")
        st.write("- 값 ↓ → 더 단순화된 경로, 노드 적음")
        st.number_input(" ", 0, 100, 8, 1, label_visibility="collapsed", key="path_precision")

st.subheader("이미지 입력")
image = st.file_uploader(" ", label_visibility="collapsed", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="target_image")
if image:
    colormode = st.session_state["colormode"]
    hierarchical = st.session_state["hierarchical"]
    mode = st.session_state["mode"]
    filter_speckle = st.session_state["filter_speckle"]
    color_precision = st.session_state["color_precision"]
    layer_difference = st.session_state["layer_difference"]
    corner_threshold = st.session_state["corner_threshold"]
    length_threshold = st.session_state["length_threshold"]
    max_iterations = st.session_state["max_iterations"]
    splice_threshold = st.session_state["splice_threshold"]
    path_precision = st.session_state["path_precision"]

    # 여기서 변환 진행
    raw = image.read()

    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if img.mode == "LA" else "RGB")

    MAX_SIDE = 1000
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    buf = io.BytesIO()

    img.save(buf, format="PNG")
    resized_bytes = buf.getvalue()

    svg_bytes = convert_raw_image_to_svg(
        #img_bytes=raw,
        img_bytes=resized_bytes,
        colormode=colormode,
        hierarchical=hierarchical,
        mode=mode,
        filter_speckle=int(filter_speckle),
        color_precision=int(color_precision),
        layer_difference=int(layer_difference),
        corner_threshold=int(corner_threshold),
        length_threshold=float(length_threshold),
        max_iterations=int(max_iterations),
        splice_threshold=int(splice_threshold),
        path_precision=int(path_precision),
    )

    col1, col2 = st.columns(2, border=True)

    with col1:
        st.subheader("원본 이미지")
        #st.image(raw)
        st.image(resized_bytes)

    with col2:
        st.subheader("SVG 미리보기")
        # st.components.v1.html(
        #     svg_bytes.decode("utf-8"),
        #     height=600,
        #     scrolling=True
        # )
        st.image(svg_bytes)

        st.download_button(
            "⬇️ SVG 다운로드",
            data=svg_bytes,
            file_name=(image.name.rsplit(".", 1)[0] + ".svg"),
            mime="image/svg+xml"
        )