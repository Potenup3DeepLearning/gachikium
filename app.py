# 서비스 화면
## 1. 사용자 정보 입력(이름, 나이, 이상형, 얼굴 사진)
## 2. 자녀 가치관 설문 진행
## 3. 결과 표시 
### 3-1. 나의 동물상, 이상형, 자녀 가치관 정보
### 3-2. 매칭된 사람의 동물상, 이상형, 자녀 가치관 정보
### 3-3. 서로 연관성 관련된 지표 

import streamlit as st
import time
from PIL import Image

# 페이지 설정
st.set_page_config(
    page_title="Value-Match | AI 가치관 소개팅",
    page_icon="❤️",
    layout="centered"
)

# 세션 상태 초기화 (페이지 네비게이션 및 등록 상태 유지)
if 'page' not in st.session_state:
    st.session_state.page = 'register'
if 'registered' not in st.session_state:
    st.session_state.registered = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# 커스텀 CSS로 디자인 개선
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #1e40af;
        color: white;
    }
    .title-text {
        text-align: center;
        color: #1e293b;
        font-family: 'Merriweather', serif;
    }
    .report-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .value-tag {
        display: inline-block;
        background-color: #f1f5f9;
        color: #1e40af;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        margin-right: 5px;
        margin-bottom: 5px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

