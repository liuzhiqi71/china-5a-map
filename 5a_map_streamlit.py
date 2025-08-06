#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5a_map_streamlit.py
-------------------------------------------------
Author : Will
Purpose: Interactive map + AI Q&A for China's National 5A‑rated tourist attractions
    • Top banner: “全国 5A 级景点” + sub‑title “by Will”
    • Sidebar: province / city multiselect → map updates in real time
    • Folium map: CartoDB‑Positron style + MarkerCluster + minimalist circle icons
    • Bottom panel: DeepSeek‑powered chat assistant (answers 5A spot questions)
    • Responsive layout: desktop & mobile

Run      :  streamlit run 5a_map_streamlit.py
Requires :  pip install streamlit pandas folium requests python-dotenv
-------------------------------------------------
"""

from pathlib import Path

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster, BeautifyIcon
from streamlit_folium import st_folium

import os
import requests
import json
import threading, queue, time
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())  # load variables from .env if present
# ────────────── 路径配置 ──────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "5A_scenic_geo_places.csv"

# DeepSeek API Key (set DEEPSEEK_API_KEY in your shell or .env)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    st.warning("未检测到 DEEPSEEK_API_KEY，聊天功能将无法使用。", icon="⚠️")

# ────────────── 页面设置 ──────────────
st.set_page_config(page_title="全国5A级景点", layout="wide")

st.markdown(
    """
    <h1 style='margin-bottom:0.2em'>5A级景点｜为了下一次更好地出发！⛰️🏔️🗻</h1>
    <p style='color:gray;margin:0'>by&nbsp;Will</p>
    """,
    unsafe_allow_html=True,
)

# ────────────── 读取数据 ──────────────
@st.cache_data
def load_df(path: Path, mtime: float) -> pd.DataFrame:   # mtime 参与 key
    df_ = pd.read_csv(path)
    return df_.dropna(subset=["longitude", "latitude"])

df = load_df(DATA_CSV, DATA_CSV.stat().st_mtime)         # include mtime so cache invalidates on file update

# ────────────── 左栏筛选 ──────────────
with st.sidebar:
    st.header("筛选")
    provinces = sorted(df["province"].dropna().unique())
    sel_provinces = st.multiselect("省份", provinces, default=provinces)

    cities = sorted(
        df.loc[df["province"].isin(sel_provinces), "city"].dropna().unique()
    )
    sel_cities = st.multiselect("城市", cities, default=cities)

MUNICIPALITIES = {"北京市", "上海市", "天津市", "重庆市"}

# ────────────── 构建过滤掩码 ──────────────
mask_prov = df["province"].isin(sel_provinces)

# 直辖市：若省份被选中，则忽略 city 字段，直接全部包含
muni_selected = set(sel_provinces) & MUNICIPALITIES
mask_city = (
    df["city"].isin(sel_cities) |           # 普通城市正常匹配
    df["province"].isin(muni_selected)      # 选中的直辖市全量匹配
)

df_view = df[mask_prov & mask_city]

st.info(f"共 **{len(df_view)}** 个景点", icon="🚗")

# ────────────── 生成动态 Folium ──────────────
def build_map(df_subset: pd.DataFrame) -> folium.Map:
    # 空集则定位中国中心
    if df_subset.empty:
        m = folium.Map(location=[35.0, 103.8], zoom_start=4, tiles="CartoDB positron")
        return m

    m = folium.Map(
        location=[df_subset["latitude"].mean(), df_subset["longitude"].mean()],
        zoom_start=5,
        tiles="CartoDB positron",
    )
    cluster = MarkerCluster().add_to(m)

    for _, row in df_subset.iterrows():
        popup = (
            f'<div style="padding:6px 10px;border-radius:8px;'
            f'background:#ffffff;box-shadow:0 1px 4px rgba(0,0,0,.15);'
            f'font-weight:500;font-size:0.75em;">{row["scenic"]}'
            f'<br/><small style="font-size:0.4em;">{row["province"]}·{row["city"]}</small></div>'
        )
        folium.Marker(
            location=(row["latitude"], row["longitude"]),
            popup=folium.Popup(popup, max_width=250),
            icon=BeautifyIcon(
                icon_shape="circle",
                border_color="#4a4a4a",
                border_width=1,
                background_color="#f5f5f5",
                icon_size=[12, 12],
            ),
        ).add_to(cluster)

    return m


# ────────────── DeepSeek Chat Helper ──────────────
_END_TOKEN = "__END_OF_STREAM__"
def ask_deepseek(prompt: str, history: list[tuple[str, str]], temperature: float = 1.0):
    """
    Call DeepSeek Chat Completion API and stream the answer.
    Only uses the prompt text; model & parameters are fixed for simplicity.
    """
    if not DEEPSEEK_API_KEY:
        yield "⚠️ 未设置 API Key，无法调用 DeepSeek。"
        return

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    # Build message history: system prompt + last 6 turns to conserve tokens
    messages = [{"role": "system", "content": "你是一位中国旅行顾问，专门回答关于全国5A级景点的问答，并且应参考上下文连续回答。"}]

    # Append recent history
    for role, content in history[-6:]:
        messages.append({"role": role, "content": content})

    # Current user question
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "deepseek-reasoner",
        "messages": messages,
        "max_tokens": 1600,
        "temperature": temperature,
        "stream": True
    }

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            full_answer = ""
            for line in resp.iter_lines():
                if not line or line == b": ping":
                    continue  # skip keep‑alive

                if line.startswith(b"data: "):
                    data_str = line[6:]
                    # Stream finished
                    if data_str == b"[DONE]":
                        break
                    # Attempt to parse JSON; skip invalid chunks
                    try:
                        line_json = json.loads(data_str)
                        delta = (line_json.get("choices")
                                 and line_json["choices"][0]["delta"].get("content"))
                        if delta:
                            full_answer += delta
                            yield delta
                    except json.JSONDecodeError:
                        continue  # ignore malformed line
            return full_answer
    except Exception as exc:
        yield f"❌ 调用 DeepSeek 失败: {exc}"


fmap = build_map(df_view)
st_folium(fmap, use_container_width=True, height=680)

# ────────────── 景区问答对话框 ──────────────
st.divider()
st.subheader("🎤 5A 景点问答｜AI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# Input box at bottom of the page
user_question = st.chat_input("关于 5A 景点想问什么？")
if user_question:
    # --- 把用户问题写入界面 & 历史 ---
    st.session_state.chat_history.append(("user", user_question))
    st.chat_message("user").write(user_question)

    # --- 先显示“思考中”占位 ---
    assistant_container = st.chat_message("assistant")
    with assistant_container:
        thinking_slot = st.empty()
        thinking_slot.write("🤔 Thinking...")

    # --- 启动后台线程流式获取答案 ---
    q: queue.Queue[str] = queue.Queue()

    # Capture current history outside the thread (session_state is not thread‑safe)
    history_snapshot = list(st.session_state.chat_history)

    def _worker(history_local: list[tuple[str, str]]):
        """Background thread: stream from DeepSeek and push to queue."""
        try:
            for chunk in ask_deepseek(user_question, history_local, temperature=0.3):
                q.put(chunk or "")  # enqueue even empty chunk
            q.put(_END_TOKEN)
        except Exception as exc:
            q.put(f"❌ DeepSeek 错误: {exc}")
            q.put(_END_TOKEN)

    threading.Thread(target=_worker, args=(history_snapshot,), daemon=True).start()

    # --- 先阻塞直到拿到第一块内容，以便保持“Thinking...”更久 ---
    streamed_answer = ""
    try:
        first_chunk = q.get(timeout=25)  # 最多等 25 秒
    except queue.Empty:
        thinking_slot.markdown("❌ 25 秒无响应，请稍后再试")
        st.session_state.chat_history.append(("assistant", "25 秒无响应，请稍后再试"))
        first_chunk = _END_TOKEN
    if first_chunk == _END_TOKEN:
        thinking_slot.markdown("❓ 未收到回答")
        st.session_state.chat_history.append(("assistant", "未收到回答"))
    else:
        streamed_answer += first_chunk
        thinking_slot.markdown(streamed_answer + "▌")

        # --- 后续非阻塞轮询，继续拼接 ---
        cursor_visible = True
        last_update = time.time()

        while True:
            try:
                chunk = q.get(timeout=0.1)
                if chunk == _END_TOKEN:
                    break
                streamed_answer += chunk
                thinking_slot.markdown(streamed_answer + ("▌" if cursor_visible else ""))
            except queue.Empty:
                # toggle cursor every 0.5 s for blink effect
                if time.time() - last_update > 0.5:
                    cursor_visible = not cursor_visible
                    thinking_slot.markdown(streamed_answer + ("▌" if cursor_visible else ""))
                    last_update = time.time()

        # --- 写入历史并显示最终答案 ---
        thinking_slot.markdown(streamed_answer)
        st.session_state.chat_history.append(("assistant", streamed_answer))