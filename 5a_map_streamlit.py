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
            f'font-weight:500;font-size:1.5em;">{row["scenic"]}'
            f'<br/><small style="font-size:0.8em;">{row["province"]}·{row["city"]}</small></div>'
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
def ask_deepseek(prompt: str) -> str:
    """
    Call DeepSeek Chat Completion API and return the answer.
    Only uses the prompt text; model & parameters are fixed for simplicity.
    """
    if not DEEPSEEK_API_KEY:
        return "⚠️ 未设置 API Key，无法调用 DeepSeek。"

    # DeepSeek API uses an OpenAI-compatible endpoint under /v1
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "你是一位中国旅行顾问，专门回答关于全国5A级景点的问答。所有回答内容输出限制在1000token以内，且必须每一次都提供语言完整的答案"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1000,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # DeepSeek may respond in streaming (delta) or non‑stream (message) format – handle both.
        if "message" in data["choices"][0]:
            return data["choices"][0]["message"]["content"]
        else:
            return data["choices"][0]["delta"]["content"]
    except Exception as exc:
        return f"❌ 调用 DeepSeek 失败: {exc}"


fmap = build_map(df_view)
st_folium(fmap, use_container_width=True, height=680)

# ────────────── 景区问答对话框 ──────────────
st.divider()
st.subheader("🎤 5A 景点问答| AI")

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

    # --- 调用 API ---
    answer = ask_deepseek(user_question)

    # --- 用真实回答替换占位 ---
    thinking_slot.markdown(answer)

    # --- 写入历史 ---
    st.session_state.chat_history.append(("assistant", answer))

# —— 样式覆盖放在最后，确保高优先级 —— #
st.markdown(
    """
    <style>

      /* 侧边栏所有 BaseWeb 文本 → #99d8c9 */
      /* all BaseWeb text inside the sidebar → #99d8c9 */
      div[data-testid="stSidebar"] [data-baseweb],
      div[data-testid="stSidebar"] [data-baseweb] *{
        color:#99d8c9 !important;
      }

      /* 已选 tag 背景 + 边框 */
      div[data-testid="stSidebar"] [data-baseweb="tag"]{
      background:#d0f0ec !important;
        border-color:#99d8c9 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)