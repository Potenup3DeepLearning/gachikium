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

def render_register():
    st.markdown("<h1 class='title-text'>❤️ 가치 키움</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>당신의 가치관과 닮은 꼴을 찾아드려요</p>", unsafe_allow_html=True)
    
    st.divider()

    # 등록이 완료되지 않았을 때 폼을 보여줌
    if not st.session_state.registered:
        st.subheader("Step 1. 기본 정보 등록")
        with st.form("profile_form"):
            col1, col2 = st.columns([1, 1])
            with col1:
                name = st.text_input("성함 (또는 닉네임)", placeholder="예: 홍길동")
                gender = st.selectbox("성별", ["선택하세요", "남성", "여성"])
                age = st.number_input("나이", min_value=19, max_value=60, value=25)
            with col2:
                st.write("본인 확인용 얼굴 사진")
                uploaded_file = st.file_uploader("사진을 업로드하세요", type=['jpg', 'jpeg', 'png'])
            
            submit_button = st.form_submit_button("프로필 등록 및 동물상 분석 시작")

        if submit_button:
            if not name or not uploaded_file or gender == "선택하세요":
                st.error("이름, 성별, 그리고 사진을 모두 입력해주세요.")
            else:
                # 분석 프로세스 시뮬레이션
                with st.status("AI 모델이 얼굴 이미지를 분석하고 있습니다...", expanded=True) as status:
                    st.write("EfficientNet-V2 모델 로드 중...")
                    time.sleep(1)
                    st.write("얼굴 특징점 추출 중...")
                    time.sleep(1)
                    status.update(label="분석 완료!", state="complete", expanded=False)
                
                # 상태 저장 후 리런
                st.session_state.registered = True
                st.session_state.user_name = name
                st.rerun()
    
    # 등록이 완료된 후 분석 결과 화면을 보여줌
    else:
        st.balloons()
        st.success(f"환영합니다, {st.session_state.user_name}님! 프로필 등록이 완료되었습니다.")
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            # 데모를 위해 아이콘이나 빈 공간으로 대체 가능 (실제 앱에선 이미지 세션 저장 필요)
            st.info("📷 사진 분석 완료")
        with res_col2:
            st.markdown("### 🦊 당신의 동물상 분석 결과")
            st.info("**귀여운 여우상** (일치도: 94.2%)")
            st.write("매력적이고 총명한 눈매를 가진 여우상입니다.")
            
        st.divider()
        st.warning("분석 결과에 기반하여 설문을 진행한 후 매칭 리포트를 확인하세요.")
        
        # 이제 버튼을 눌러도 상태가 유지되므로 정상 동작함
        if st.button("설문 건너뛰고 매칭 리포트 보기 (데모) →"):
            st.session_state.page = 'report'
            st.rerun()
            
        if st.button("← 다시 정보 수정하기"):
            st.session_state.registered = False
            st.rerun()

def render_report():
    st.markdown("<h1 class='title-text'>📑 Value-Match 리포트</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>UBCF 알고리즘이 찾아낸 최적의 파트너입니다</p>", unsafe_allow_html=True)
    st.divider()

    # 추천인 정보 섹션
    st.subheader("🌟 오늘의 베스트 매칭")
    with st.container():
        st.markdown("""
            <div class='report-card'>
                <div style='display: flex; align-items: center; gap: 20px;'>
                    <div style='font-size: 50px;'>🐶</div>
                    <div>
                        <h2 style='margin:0;'>김민수 (31세)</h2>
                        <p style='color: #64748b; margin:0;'>순한 인상의 <strong>강아지상</strong> | 가치관 유사도 <strong>92%</strong></p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 자녀 가치관 정보 섹션
    st.subheader("👨‍👩‍👧‍👦 자녀 가치관 분석 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎓 교육 및 훈육 철학")
        st.write("상대방은 자녀의 **자율성**을 가장 중요하게 생각합니다.")
        st.markdown("<span class='value-tag'>#창의성_중심</span><span class='value-tag'>#대화형_훈육</span>", unsafe_allow_html=True)
        st.markdown("<span class='value-tag'>#예체능_활동_장려</span>", unsafe_allow_html=True)
    with col2:
        st.markdown("#### 💰 경제 및 생활 가치")
        st.write("미래를 위한 **안정적인 자산 형성**을 선호하는 타입입니다.")
        st.markdown("<span class='value-tag'>#경제교육_조기시작</span><span class='value-tag'>#검소한_소비습관</span>", unsafe_allow_html=True)
        st.markdown("<span class='value-tag'>#주말은_가족과함께</span>", unsafe_allow_html=True)

    st.divider()
    
    # 상세 비교 그래프 시뮬레이션
    st.markdown("#### 📊 가치관 일치도 상세")
    st.progress(0.95, text="자녀 교육관 (95%)")
    st.progress(0.88, text="가사 분담 (88%)")
    st.progress(0.92, text="경제관념 (92%)")

    st.divider()
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("처음으로 돌아가기"):
            st.session_state.page = 'register'
            st.session_state.registered = False
            st.rerun()
    with col_b:
        if st.button("대화 신청하기 (❤️)"):
            st.confetti()
            st.success("매칭 신청이 전달되었습니다!")

def main():
    # 사이드바 메뉴 업데이트
    with st.sidebar:
        st.title("메뉴")
        if st.session_state.page == 'register':
            st.success("1. 프로필 등록 (진행 중)")
            st.info("2. 가치관 설문 (대기)")
            st.info("3. 매칭 리포트 (대기)")
        else:
            st.info("1. 프로필 등록 (완료)")
            st.info("2. 가치관 설문 (완료)")
            st.success("3. 매칭 리포트 (확인 중)")

    if st.session_state.page == 'register':
        render_register()
    else:
        render_report()

if __name__ == "__main__":
    main()