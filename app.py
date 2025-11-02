import io
from PIL import Image, ImageOps
import streamlit as st
from vtracer import convert_raw_image_to_svg
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# NEW: CairoSVG
# pip install cairosvg
import cairosvg

st.set_page_config(layout="wide")

st.header("image2svg")

with st.expander("세부 조절"):

    col1, col2, col3, col4 = st.columns(4, border=True)

    with col1:
        st.subheader("`colormode`")
        st.write("- `color`: 색상을 유지하며 벡터화")
        st.write("- `binary`: 흑백(단색)으로 벡터화")
        st.selectbox(" ", ["color", "binary"], 0, label_visibility="collapsed", key="colormode")

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

st.subheader("색상 샘플링")
st.number_input(label="샘플링할 색상의 수", min_value=1, max_value=100, value=20, key="num_colors")

st.subheader("이미지 입력")
image = st.file_uploader(" ", label_visibility="collapsed", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="target_image")

if image:
    colormode = st.session_state["colormode"]
    mode = st.session_state["mode"]
    filter_speckle = st.session_state["filter_speckle"]
    color_precision = st.session_state["color_precision"]
    layer_difference = st.session_state["layer_difference"]
    corner_threshold = st.session_state["corner_threshold"]
    length_threshold = st.session_state["length_threshold"]
    max_iterations = st.session_state["max_iterations"]
    splice_threshold = st.session_state["splice_threshold"]
    path_precision = st.session_state["path_precision"]

    raw = image.read()

    pil_in = Image.open(io.BytesIO(raw))
    pil_in = ImageOps.exif_transpose(pil_in)

    if pil_in.mode not in ("RGB", "RGBA"):
        pil_in = pil_in.convert("RGBA" if pil_in.mode == "LA" else "RGB")

    MAX_SIDE = 1000
    pil_in.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    h, w = pil_in.height, pil_in.width

    buf = io.BytesIO()
    pil_in.save(buf, format="PNG")
    resized_bytes = buf.getvalue()

    # vtracer: 래스터 -> SVG
    svg_bytes = convert_raw_image_to_svg(
        img_bytes=resized_bytes,
        colormode=colormode,
        hierarchical="cutoff",  # 라이브러리에 따라 'cutout'이 맞을 수 있음. 문서 확인 요망.
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

    # --- 여기부터 CairoSVG로 SVG -> PNG (메모리 변환) ---
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=w,
        output_height=h,
        # scale=1.0,                 # 필요시 배율로 조정
        # background_color='transparent',
        # dpi=96
    )

    # 바로 표시
    st.image(png_bytes)

    # PIL 이미지로 열어 후처리
    pil_png = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    np_img = np.array(pil_png)
    h, w, c = np_img.shape
    pixels = np_img.reshape(-1, c)

    # KMeans 팔레트 양자화
    kmeans = KMeans(n_clusters=st.session_state["num_colors"], random_state=42, n_init="auto")
    labels = kmeans.fit_predict(pixels)
    palette = np.round(kmeans.cluster_centers_).astype(np.uint8)
    new_pixels = palette[labels].reshape(h, w, c)

    result_img = Image.fromarray(new_pixels)
    st.image(result_img, caption="양자화 결과")

    # 최종 색상 개수 계산 (RGB 기준)
    use_alpha = False
    color_space = new_pixels[:, :, :4] if (use_alpha and new_pixels.shape[2] == 4) else new_pixels[:, :, :3]
    flat = color_space.reshape(-1, color_space.shape[2])
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)

    total_pixels = flat.shape[0]
    num_unique = unique_colors.shape[0]
    st.write(f"최종 고유 색상 수: **{num_unique}**")
    st.write(f"총 픽셀 수: {total_pixels:,}")

    # 상위 n개 색상 (많이 쓰인 순)
    n_top = min(20, num_unique)
    top_idx = np.argsort(-counts)[:n_top]
    top_colors = unique_colors[top_idx]
    top_counts = counts[top_idx]
    top_ratios = top_counts / total_pixels

    # 팔레트 미리보기
    swatch_h, swatch_w = 40, 60
    gap = 4
    palette_img = np.ones((swatch_h, n_top * (swatch_w + gap) - gap, 3), dtype=np.uint8) * 255
    for i, rgb in enumerate(top_colors[:, :3]):
        x0 = i * (swatch_w + gap)
        palette_img[:, x0:x0 + swatch_w, :] = rgb

    st.subheader("팔레트(상위 색상)")
    st.image(palette_img, caption="왼쪽부터 많이 쓰인 색")

    # 색상 표
    rows = []
    for i, (rgb, cnt, ratio) in enumerate(zip(top_colors[:, :3], top_counts, top_ratios), start=1):
        rows.append({
            "rank": i,
            "R": int(rgb[0]),
            "G": int(rgb[1]),
            "B": int(rgb[2]),
            "pixels": int(cnt),
            "ratio(%)": round(float(ratio * 100), 2),
            "hex": "#{:02X}{:02X}{:02X}".format(int(rgb[0]), int(rgb[1]), int(rgb[2])),
        })
    st.dataframe(pd.DataFrame(rows))

    # 다운로드 버튼
    out_buf = io.BytesIO()
    Image.fromarray(new_pixels).save(out_buf, format="PNG")
    st.download_button(
        "최종 이미지 다운로드 (PNG)",
        data=out_buf.getvalue(),
        file_name="quantized.png",
        mime="image/png"
    )

    pal_buf = io.BytesIO()
    Image.fromarray(palette_img).save(pal_buf, format="PNG")
    st.download_button(
        "팔레트 다운로드 (PNG)",
        data=pal_buf.getvalue(),
        file_name="palette.png",
        mime="image/png"
    )
