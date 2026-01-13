"""
Dependencies (예시):
pip install streamlit torch transformers accelerate bitsandbytes opencv-python pillow
# (가능하면) pip install decord

Windows에서 코덱 이슈가 있을 때 torchvision 백엔드 대신 OpenCV/Decord 사용을 선호하도록
환경변수를 'qwen_vl_utils' 임포트 전에 설정합니다.
"""
import os
import time  # 성능 측정을 위해 추가

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("QWEN_VL_VIDEO_READER", "opencv")  # "decord" 설치 시 "decord" 권장

import shutil, re
import tempfile
from pathlib import Path
from threading import Thread
from datetime import datetime

import streamlit as st
import torch
from PIL import Image
import cv2

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer,
)

from qwen_vl_utils import process_vision_info


# ---------------- UI setup ----------------
st.set_page_config(page_title="Auto Report Playground")
st.header("`Auto Report` VLM Demo (Video & Image)")

# Sidebar: model select and params
MODELS = {
    "Qwen 2.5 VL": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Video LLaMA3": "DAMO-NLP-SG/VideoLLaMA3-2B",
}
model_display_name = st.sidebar.radio("Select VLM Model", list(MODELS.keys()))
selected_model = MODELS[model_display_name]

max_tokens = st.sidebar.slider("Max Tokens", 10, 3000, 1500)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9, 0.05)
save_to_output = st.sidebar.checkbox("Save response", value=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 속도/안정화 옵션
use_4bit = st.sidebar.checkbox("Use 4-bit quantization (GPU only)", value=torch.cuda.is_available())
use_flash_attn2 = st.sidebar.checkbox("Use Flash-Attention 2 (if installed, GPU)", value=False)

# matmul 최적화(가능한 경우)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent

# ---------------- Helpers ----------------
def _slugify(name: str) -> str:
    # 폴더명에 문제될 수 있는 문자 치환 (한글/영문/숫자/공백/.-_만 허용)
    return re.sub(r'[^\w\-\.\s가-힣]', '_', name).strip() or "keyword"

def save_response_to_output(matched_keywords, response_text: str):
    """
    output/<키워드>/YYYYMMDD_HHMMSS.txt 로 저장.
    여러 키워드가 매칭되면 각각의 폴더에 동일한 내용 저장.
    """
    out_root = BASE_DIR / "output"
    out_root.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_paths = []
    for kw in matched_keywords:
        kw_dir = out_root / _slugify(kw)
        kw_dir.mkdir(parents=True, exist_ok=True)
        file_path = kw_dir / f"{ts}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response_text)
        saved_paths.append(file_path)
    return saved_paths

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

def save_uploaded_video(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmpdir = Path(tempfile.mkdtemp(prefix="st_vid_"))
    out = tmpdir / f"input{suffix}"
    with open(out, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out.resolve().as_posix()

def save_uploaded_image(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    tmpdir = Path(tempfile.mkdtemp(prefix="st_img_"))
    out = tmpdir / f"input{suffix}"
    with open(out, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out.resolve().as_posix()

def detect_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "video"

def build_conversation(media_type: str, media_path: str, user_prompt: str):
    """
    media_type: 'image' | 'video'
    media_path: 로컬 파일 경로
    """
    content_key = "image" if media_type == "image" else "video"
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": content_key, content_key: media_path},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

def extract_video_frames_cv2(video_path: str, max_frames: int = 32):
    """OpenCV로 프레임 샘플링(폴백용). 최소 2프레임을 보장하도록 시도."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video with OpenCV: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, (total // max_frames) if total > 0 else 1)

    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()

    if len(frames) < 2:
        if len(frames) == 1:
            frames = frames * 2
        else:
            raise ValueError(f"Too few frames extracted ({len(frames)}).")
    return frames

def _make_qwen(model_id: str):
    # 4-bit 양자화 시도
    quant_args = {}
    dtype = torch.float16 if device == "cuda" else torch.float32
    if use_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quant_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            dtype = None  # 4bit일 땐 dtype 지정 X
        except Exception:
            pass

    attn_impl = "flash_attention_2" if (use_flash_attn2 and device == "cuda") else "sdpa"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto",
        **quant_args,
    ).eval()

    # 해상도 조절로 인코딩 속도 개선
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=192 * 28 * 28,   # 기본 256 → 192
        max_pixels=640 * 28 * 28,   # 기본 896 → 640
    )
    return model, processor

def _make_videollama(model_id: str):
    quant_args = {}
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if use_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quant_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            dtype = None
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
        **quant_args,
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

@st.cache_resource(show_spinner=False)
def _cached_load(model_id: str, key: str):
    """캐시: 모델/프로세서를 생성해 보관. key로 옵션 변화를 반영."""
    if "Qwen" in model_id:
        return _make_qwen(model_id)
    else:
        return _make_videollama(model_id)

def get_model_and_processor(model_id: str):
    """세션 전역 모델 풀. 동일 모델+옵션이면 절대 재로딩 안 함."""
    pool = st.session_state.setdefault("model_pool", {})
    opt_key = f"{model_id}|4bit={use_4bit}|fa2={use_flash_attn2}|dev={device}"
    if opt_key not in pool:
        pool[opt_key] = _cached_load(model_id, key=opt_key)
    return pool[opt_key]

def prepare_inputs_with_fallback_for_qwen(processor, conversation):
    """
    Qwen 경로: process_vision_info가 실패하면
    - video면 OpenCV로 프레임 추출
    - image면 PIL로 열어 직접 전달
    """
    try:
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(conversation)
        return text, images, videos
    except Exception:
        contents = conversation[1]["content"]
        user_prompt = next(c["text"] for c in contents if c.get("type") == "text")
        media = next(c for c in contents if c.get("type") in ("video", "image"))

        if media["type"] == "video":
            video_path = media["video"]
            frames = extract_video_frames_cv2(video_path, max_frames=32)
            conv_fallback = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
        else:
            image_path = media["image"]
            img = Image.open(image_path).convert("RGB")
            conv_fallback = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

        text = processor.apply_chat_template(conv_fallback, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(conv_fallback)
        return text, images, videos

def prepare_inputs_with_fallback_for_videollama(processor, conversation):
    """
    VideoLLaMA 경로: 실패 시
    - video: 프레임 리스트로 재시도
    - image: 이미지를 2프레임으로 복제해 'video'처럼 전달(모델 호환성용)
    """
    try:
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return inputs
    except Exception:
        contents = conversation[1]["content"]
        user_prompt = next(c["text"] for c in contents if c.get("type") == "text")
        media = next(c for c in contents if c.get("type") in ("video", "image"))

        if media["type"] == "video":
            video_path = media["video"]
            frames = extract_video_frames_cv2(video_path, max_frames=32)
        else:
            image_path = media["image"]
            img = Image.open(image_path).convert("RGB")
            frames = [img, img]  # 최소 2프레임 보장

        conv_fallback = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        inputs = processor(
            conversation=conv_fallback,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return inputs

# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Drop a video or image below and tell me what to do."}]
if "media_path" not in st.session_state:
    st.session_state.media_path = None
if "media_type" not in st.session_state:
    st.session_state.media_type = None
if "active_model_key" not in st.session_state:
    st.session_state.active_model_key = None

# ---------------- Uploader + preview ----------------
st.markdown("### 1) Drop your **video or image** here")
uploaded = st.file_uploader(
    "Drag & drop media (mp4/mov/avi/mkv/jpg/png/webp/bmp)",
    type=["mp4", "mov", "avi", "mkv", "jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=False,
)
if uploaded is not None:
    ext = Path(uploaded.name).suffix.lower()
    if ext in IMAGE_EXTS:
        st.session_state.media_path = save_uploaded_image(uploaded)
        st.session_state.media_type = "image"
    else:
        st.session_state.media_path = save_uploaded_video(uploaded)
        st.session_state.media_type = "video"

if st.session_state.media_path:
    if st.session_state.media_type == "image":
        st.image(st.session_state.media_path, width="stretch")
    else:
        st.video(st.session_state.media_path)
    st.caption(f"Using {st.session_state.media_type}: {st.session_state.media_path}")
else:
    st.info("No media selected yet.")

st.markdown("### 2) Ask your question about the media")

# Render prior messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("e.g., 영상/이미지 자세히 요약해줘, 어떤 행위를 하는지도")


# ---------------- Generate (streaming) ----------------
def qwen_stream(model, processor, conversation, max_tokens: int, top_p: float):
    text, images, videos = prepare_inputs_with_fallback_for_qwen(processor, conversation)
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    ).to(device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=(top_p < 1.0),
        top_p=top_p,
        streamer=streamer,
    )

    th = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    th.start()
    for token in streamer:
        yield token

def videollama_stream(model, processor, conversation, max_tokens: int, top_p: float):
    inputs = prepare_inputs_with_fallback_for_videollama(processor, conversation)
    inputs = inputs.to(device)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=(top_p < 1.0),
            top_p=top_p,
            streamer=streamer,
        )
        th = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
        th.start()
        for token in streamer:
            yield token
    else:
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(top_p < 1.0),
                top_p=top_p,
            )
        resp = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        yield resp

def sanitize_korean_text(s: str) -> str:
    """
    한글/기본 라틴 문자/일부 구두점만 허용하고 나머지(한자·중국어·일본어 등)는 제거.
    """
    import unicodedata
    s = unicodedata.normalize("NFC", s)
    allowed = r"[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\u0020-\u007E\u2010-\u2015\u2018-\u201F\n\t]"
    return re.sub(allowed, "", s)

def clean_stream(stream):
    """토큰 스트림을 실시간 정제해서 st.write_stream에 넘기기 위한 래퍼."""
    for chunk in stream:
        yield sanitize_korean_text(chunk)

# ---------------- Performance Monitoring Helper ----------------
class StreamMonitor:
    def __init__(self):
        self.start_time = 0.0
        self.first_token_time = 0.0
        self.end_time = 0.0
        self.token_count = 0

    def measure(self, stream_generator):
        """스트림 제너레이터를 감싸서 측정"""
        self.start_time = time.time()
        self.token_count = 0
        
        for item in stream_generator:
            if self.token_count == 0:
                self.first_token_time = time.time()
            self.token_count += 1
            yield item
        
        self.end_time = time.time()

    @property
    def ttft(self):
        """Time to First Token (seconds)"""
        if self.first_token_time == 0:
            return 0.0
        return self.first_token_time - self.start_time

    @property
    def total_latency(self):
        """Total Generation Time (seconds)"""
        if self.end_time == 0:
            return 0.0
        return self.end_time - self.start_time

    @property
    def throughput(self):
        """Tokens per second (Decoding phase)"""
        # 첫 토큰 이후의 생성 속도를 측정 (일반적인 Throughput 정의)
        decoding_time = self.end_time - self.first_token_time
        if decoding_time <= 0 or self.token_count <= 1:
            return 0.0
        return (self.token_count - 1) / decoding_time

if prompt is not None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        if not st.session_state.media_path:
            st.error("먼저 영상 또는 이미지를 드래그&드롭으로 업로드해 주세요.")
        else:
            model, processor = get_model_and_processor(selected_model)
            conversation = build_conversation(st.session_state.media_type, st.session_state.media_path, prompt)

            with st.spinner("Analyzing..."):
                if "Qwen" in selected_model:
                    raw_stream = qwen_stream(model, processor, conversation, max_tokens, top_p)
                else:
                    raw_stream = videollama_stream(model, processor, conversation, max_tokens, top_p)

                # 성능 측정 래퍼 생성
                monitor = StreamMonitor()
                
                # 파이프라인: raw_stream -> monitor.measure -> clean_stream -> st.write_stream
                # 이렇게 해야 clean 전의 '토큰(청크) 수'를 정확히 측정 가능
                measured_stream = monitor.measure(raw_stream)
                cleaned_stream = clean_stream(measured_stream)

                response = st.write_stream(cleaned_stream)
                
                # 안전하게 한 번 더 정제
                response = sanitize_korean_text(response)

                # 키워드 검사 → 표시 + 저장
                keywords = ["교통사고", "교통 사고", "충돌", "사고"]
                matched = [kw for kw in keywords if kw in response]
                if matched:
                    tag_line = f"keyword: {', '.join(matched)}"
                    st.markdown(tag_line)
                    response = f"{response}\n\n{tag_line}"

                if save_to_output:
                    _ = save_response_to_output(matched, response)

            # 성능 지표 표시
            if monitor.token_count > 0:
                cols = st.columns(3)
                cols[0].metric("TTFT (첫 토큰 시간)", f"{monitor.ttft:.3f} s")
                cols[1].metric("Throughput (생성 속도)", f"{monitor.throughput:.2f} toks/s")
                cols[2].metric("Total Latency", f"{monitor.total_latency:.2f} s")

    st.session_state.messages.append({"role": "assistant", "content": response})


# Sidebar tools
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Drop a video or image below and tell me what to do."}]
    # st.session_state.media_path = None  # 미디어까지 초기화하려면 주석 해제

st.sidebar.button("Clear chat history", on_click=clear_chat_history)

# 모델 전환 시 리소스 교체 (선택): 필요 시 캐시 정리로 VRAM 확보
active_key = f"{selected_model}|4bit={use_4bit}|fa2={use_flash_attn2}|dev={device}"
if st.session_state.active_model_key and st.session_state.active_model_key != active_key:
    try:
        st.cache_resource.clear()
    except Exception:
        pass
st.session_state.active_model_key = active_key