import streamlit as st
import requests
import io

#================================================================================
# --- [설정] ---
#================================================================================

# FastAPI 백엔드 서버 주소
API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="가치관 매칭 서비스", layout="centered")

#================================================================================
# --- [세션 상태 초기화 (페이지 이동 및 데이터 저장용)] ---
#================================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'       # 현재 페이지
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""      # 유저 이름
if 'user_photo' not in st.session_state:
    st.session_state.user_photo = None   # 유저 사진 (UploadedFile 객체)
if 'session_id' not in st.session_state:
    st.session_state.session_id = None   # 백엔드 세션 ID

#================================================================================
# --- [백엔드 세션 생성/복원] ---
#================================================================================
def ensure_backend_session():
    """백엔드 세션이 없으면 새로 생성"""
    if st.session_state.session_id is None:
        try:
            resp = requests.post(f"{API_BASE_URL}/api/session")
            resp.raise_for_status()
            data = resp.json()
            st.session_state.session_id = data["session_id"]
        except Exception as e:
            st.error(f"⚠️ 백엔드 서버 연결 실패: {e}")
            st.stop()

ensure_backend_session()

#================================================================================
# --- [학습 당시 10개 클래스 순서에 맞춘 이모지와 이름 매핑] ---
#================================================================================
animal_mapping = [
    "🐶 강아지상", "🐱 고양이상", "🐰 토끼상", "🦖 공룡상", "🐻 곰상",
    "🦊 여우상", "🐴 말상", "🐵 원숭이상", "🐭 쥐상", "🐷 돼지상"
]

#================================================================================
# --- [페이지 네비게이션 로직] ---
#================================================================================
def go_to(page_name):
    """페이지 이동 + 백엔드 동기화"""
    st.session_state.page = page_name
    # 백엔드에도 페이지 상태 동기화
    try:
        requests.post(
            f"{API_BASE_URL}/api/session/{st.session_state.session_id}/navigate",
            params={"page": page_name}
        )
    except Exception:
        pass  # 네비게이션 동기화 실패해도 프론트엔드 이동은 진행

    # 최신 버전이면 rerun(), 구버전이면 experimental_rerun() 실행
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


#================================================================================
# --- [CSS 스타일 정의] ---
#================================================================================

st.markdown("""
    <style>
    /* 1. 배경 및 카드 스타일 */
 

    /* 2. 진중한 매칭 서비스에 어울리는 버튼 스타일 (미드나잇 블루) */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-color: #1E3A8A !important; /* Deep Navy Blue */
        color: white !important;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* 버튼에 마우스 올렸을 때 효과 */
    .stButton>button:hover {
        background-color: #172554 !important; /* 더 깊은 네이비 */
        box-shadow: 0 5px 15px rgba(30, 58, 138, 0.3);
    }

    /* 3. 입력창 테두리 강조 색상 변경 */
    input:focus {
        border-color: #1E3A8A !important;
    }
    </style>
    """, unsafe_allow_html=True)

#================================================================================
# --- [사이드바 구성 정의] ---
#================================================================================
def render_sidebar():
    with st.sidebar:
        st.title("📌 매칭 프로세스")
        
        # 각 페이지별 상태 정의
        steps = {
            'home': "1. 기본 정보 입력",
            'survey': "2. 가치관 설문조사",
            'matching': "3. AI 매칭 리포트"
        }
        
        for key, label in steps.items():
            if st.session_state.page == key:
                # 현재 페이지 강조
                st.markdown(f"### **🎯 {label}**")
            else:
                # 진행 전 또는 완료된 페이지
                st.markdown(f" {label}")
        
        st.markdown("---")
        if st.session_state.user_name:
            st.write(f"👤 **{st.session_state.user_name}** 님 진행 중")
        
        # # 사진이 업로드되었다면 작게 미리보기 (선택 사항)
        # if st.session_state.user_photo:
        #     st.image(st.session_state.user_photo, caption="분석용 프로필", width=100)


#================================================================================
# --- [메인 실행부] ---
#================================================================================

render_sidebar() # 모든 페이지에서 사이드바가 보이도록 최상단 호출

# 이하 기존 Home / Survey / Matching 조건문 유지

# --- [1. 초기 화면 (Home) 적용 예시] ---
if st.session_state.page == 'home':
    st.title("👨‍👩‍👧‍👦 가치관 기반 파트너 매칭")
    st.write("당신의 소중한 가치관을 찾아드립니다.")

    with st.container():
        st.subheader("내 정보 입력")
        name = st.text_input("성함을 입력해주세요", value=st.session_state.user_name, placeholder="")
        photo = st.file_uploader("본인의 얼굴 사진을 업로드해주세요", type=['jpg', 'png', 'jpeg'])
        st.info("💡 사진은 동물 상 분석에 사용되며, 저장되지 않습니다.")
        
        # 이제 버튼이 고급스러운 네이비 색상으로 나옵니다.
        check_clicked = st.button("입력 완료")
        

    # 로직 체크 및 다음 단계 진행 (백엔드 API를 통해 DB 체크)
    if check_clicked:
        if name and photo:
            st.session_state.user_name = name
            st.session_state.user_photo = photo
            
            # 백엔드 API로 유저 체크 (이름 + 사진 전송 → DB 존재 여부 확인)
            try:
                photo_bytes = photo.getvalue()
                files = {"photo": (photo.name, photo_bytes, photo.type)}
                form_data = {
                    "session_id": st.session_state.session_id,
                    "name": name,
                }
                resp = requests.post(
                    f"{API_BASE_URL}/api/user/check",
                    data=form_data,
                    files=files,
                )
                resp.raise_for_status()
                result = resp.json()
                
                # DB 체크 결과를 세션에 저장
                if result["user_exists"]:
                    st.session_state.user_exists = True
                else:
                    st.session_state.user_exists = False
                    
            except Exception as e:
                st.error(f"⚠️ 서버 오류: {e}")
                st.session_state.user_exists = False
        else:
            st.error("이름과 사진을 모두 입력해주세요.")

    # 버튼 클릭 후 나타나는 선택지 (중첩 방지를 위해 if check_clicked 밖에 위치하거나 별도 처리)
    if st.session_state.user_name and st.session_state.user_photo:
        if getattr(st.session_state, 'user_exists', False):
            st.info(f"기존 기록이 있습니다. 어떻게 할까요?")
            c1, c2 = st.columns(2)
            if c1.button("기존 데이터로 매칭"):
                # 백엔드에도 기존 데이터 매칭 요청
                try:
                    requests.post(
                        f"{API_BASE_URL}/api/matching/{st.session_state.session_id}/existing"
                    )
                except Exception:
                    pass
                go_to('matching')
            if c2.button("설문 다시 하기"):
                go_to('survey')
        else:
            st.success("신규 유저입니다. 설문을 시작해주세요.")
            if st.button("설문 시작하기 ✍️"):
                go_to('survey') # 여기서 확실히 survey로 보냅니다.

# --- [2. 설문 화면 (survey) 통합 버전] ---
elif st.session_state.page == 'survey':
    st.title("📝 가치관 설문조사")
    st.write("진지한 만남을 위해 모든 항목에 정성껏 답해주세요.")

    # 10개 시나리오 질문 리스트 미리 정의
    questions = [
        "1. 아이의 자율성을 위해 위험하지 않다면 지켜보는 편이다.",
        "2. 교육을 위해 어느 정도의 엄격한 훈육은 필요하다고 생각한다.",
        "3. 경제적 여유보다 아이와 함께 보내는 시간이 더 중요하다.",
        "4. 아이의 사교육은 빠를수록 좋다고 생각한다.",
        "5. 배우자와의 양육 가치관이 다를 때 끝까지 설득하는 편이다.",
        "6. 조부모님의 양육 도움은 적극적으로 받는 것이 좋다.",
        "7. 아이의 성적보다는 인성 교육이 최우선이다.",
        "8. 주말에는 무조건 가족과 야외 활동을 나가야 한다.",
        "9. 아이에게 스마트폰 노출은 최대한 늦춰야 한다.",
        "10. 칭찬보다는 객관적인 피드백이 아이 성장에 도움이 된다."
    ]

    with st.container():
        # 전체 설문을 하나의 Form으로 감쌉니다.
        with st.form("total_survey_form"):
            
            # --- Section 1: 기본 가족 계획 ---
            st.subheader("📍 Part 1. 자녀 및 가족 계획")
            q_children = st.radio(
                "희망하는 자녀 수는 몇 명인가요?",
                ["딩크(0명)", "1명", "2명", "3명 이상"], index=1, key="final_q_child"
            )
            q_gender = st.radio(
                "선호하는 자녀의 성별 구성이 있나요?",
                ["상관없음", "아들 선호", "딸 선호", "아들/딸 골고루"], key="final_q_gen"
            )
            
            st.markdown("---")
            
            # --- Section 2: 항목 중요도 ---
            st.subheader("📍 Part 2. 매칭 가중치 설정")
            w_family = st.select_slider(
                "방금 답변한 '가족 계획' 항목이 상대방과 얼마나 일치해야 하나요?",
                options=["무관", "보통", "중요", "매우 중요"], value="중요", key="final_w_fam"
            )
            st.caption("💡 중요도가 높을수록 해당 조건이 맞는 사람을 우선적으로 추천합니다.")
            
            st.markdown("---")

            # --- Section 3: 육아 시나리오 ---
            st.subheader("📍 Part 3. 육아 시나리오 (10문항)")
            st.info("각 문항에 대해 본인의 동의 정도를 선택해주세요. (1: 매우 반대 ~ 5: 매우 찬성)")
            
            responses = []
            for i, q_text in enumerate(questions):
                st.write(f"**Q{i+1}. {q_text}**")
                score = st.select_slider(
                    "점수 선택",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"final_sc_{i}",
                    label_visibility="collapsed"
                )
                responses.append(score)
                # 마지막 문항 전까지만 얇은 선 추가
                if i < len(questions) - 1:
                    st.write("") # 간격 조절용
            
            st.markdown("---")

            # --- 최종 제출 버튼 ---
            submit_button = st.form_submit_button("설문 완료 및 AI 매칭 시작 🚀")

            if submit_button:
                # 모든 데이터 세션에 저장
                st.session_state.survey_answers = {
                    "q_children": q_children,
                    "q_gender": q_gender,
                    "w_family": w_family,
                    "responses": responses
                }
                
                # 백엔드 API로 설문 결과 전송
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "q_children": q_children,
                        "q_gender": q_gender,
                        "w_family": w_family,
                        "responses": responses
                    }
                    resp = requests.post(
                        f"{API_BASE_URL}/api/survey/submit",
                        json=payload,
                    )
                    resp.raise_for_status()
                    
                    # 성공 메시지와 함께 페이지 이동
                    st.success("데이터 전송 완료! 당신과 가장 잘 어울리는 파트너를 찾고 있습니다.")
                    go_to('matching')
                    
                except Exception as e:
                    st.error(f"⚠️ 설문 제출 실패: {e}")

# --- [3. 매칭 화면 (matching) 통합 버전] ---
elif st.session_state.page == 'matching':
    
    # --- [동물상 분석 실행 (백엔드 API 호출)] ---
    if 'user_animal_result' not in st.session_state:
        with st.spinner('AI가 당신의 사진에서 동물상을 분석하고 있습니다...'):
            if st.session_state.user_photo is not None:
                try:
                    # 사진 바이트 추출
                    photo_bytes = st.session_state.user_photo.getvalue()
                    photo_name = st.session_state.user_photo.name
                    photo_type = st.session_state.user_photo.type
                    
                    files = {"photo": (photo_name, photo_bytes, photo_type)}
                    form_data = {"session_id": st.session_state.session_id}
                    
                    resp = requests.post(
                        f"{API_BASE_URL}/api/animal/analyze",
                        data=form_data,
                        files=files,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    
                    # 결과 저장 (이모지 포함 이름 + 확률)
                    st.session_state.user_animal_result = result["animal_type"]
                    st.session_state.user_animal_prob = result["probability"]
                    
                except Exception as e:
                    # 방어 코드
                    st.session_state.user_animal_result = "❓ 미확인상"
                    st.session_state.user_animal_prob = "0%"
                    st.warning(f"동물상 분석 실패: {e}")
            else:
                # 방어 코드
                st.session_state.user_animal_result = "❓ 미확인상"
                st.session_state.user_animal_prob = "0%"
    
    
    
    st.title("💖 AI 가치관 매칭 리포트")
    
    # 상단: 유저 정보
    st.subheader(f"✨ {st.session_state.user_name}님을 위한 맞춤 분석 결과입니다.")
    st.markdown("---")

    # 매칭 리포트 데이터를 백엔드에서 가져오기
    matching_data = None
    with st.spinner('가치관 데이터를 분석하고 최적의 파트너를 찾는 중...'):
        try:
            resp = requests.get(
                f"{API_BASE_URL}/api/matching/{st.session_state.session_id}"
            )
            resp.raise_for_status()
            matching_data = resp.json()
        except Exception as e:
            st.error(f"⚠️ 매칭 리포트 로드 실패: {e}")

    if matching_data:
        best_match = matching_data["best_match"]
        user_profile = matching_data["user_profile"]
        top3_others = matching_data["top3_others"]
    else:
        # 백엔드 연결 실패 시 Fallback 임시 데이터 (원본과 동일)
        best_match = {
            "name": "이서연",
            "animal_type": "🐱 고양이상",
            "similarity_score": 94.8,
            "parenting_enthusiasm": 0.82,
            "education_passion": 0.45,
        }
        user_profile = {
            "tags": ["#자율성", "#체험학습", "#딩크희망"],
            "parenting_enthusiasm": 0.85,
            "education_passion": 0.40,
        }
        top3_others = [
            {"name": "김민수", "similarity_score": 88.2, "animal_type": "🐰 토끼상"},
            {"name": "박지혜", "similarity_score": 85.5, "animal_type": "🦊 여우상"},
            {"name": "최진우", "similarity_score": 82.9, "animal_type": "🐻 곰상"},
        ]

    # 중앙: 본인 vs 매칭 상대 비교 (Col 2)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### 👤 나의 프로필")
        # 구버전 호환: use_column_width=True
        st.image(st.session_state.user_photo, use_column_width=True)
        
        with st.container():
            # 카드 스타일 적용 (이전에 정의한 CSS 클래스 활용)
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            
            st.write(f"**이름:** {st.session_state.user_name}")
            
            # 🎯 요청하신 형식: 동물상 : 🦊여우상(83%)
            st.write(f"**동물상 :** {st.session_state.user_animal_result}({st.session_state.user_animal_prob})")
            
            # 주요 가치관 태그 (백엔드에서 받은 데이터 활용)
            tags_str = " ".join(user_profile.get("tags", ["#자율성", "#체험학습", "#딩크희망"]))
            st.write(f"**주요 가치관:** {tags_str}")
            
            # ... 나머지 코드
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 간단한 가치관 시각화 (Progress bar 활용)
            st.write("육아 적극성")
            st.progress(user_profile.get("parenting_enthusiasm", 0.85))
            st.write("교육 열정")
            st.progress(user_profile.get("education_passion", 0.40))

    with col2:
        st.markdown(f"#### 💝 BEST 매칭 파트너")
        st.image("https://via.placeholder.com/300x300.png?text=Partner+Photo", use_column_width=True)
        
        with st.container():
            st.write(f"**이름:** {best_match['name']}")
            st.write(f"**동물상:** {best_match['animal_type']}")
            
            # 상대방 수치 (나와 비교되게)
            st.write("육아 적극성")
            st.progress(best_match.get("parenting_enthusiasm", 0.82))
            st.write("교육 열정")
            st.progress(best_match.get("education_passion", 0.45))
            
            st.metric("매칭 일치율", f"{best_match['similarity_score']}%")

    st.markdown("---")

    # 하단: 그외 추천 파트너 TOP 3
    st.subheader("🔍 또 다른 인연들을 확인해보세요 (TOP 3)")
    
    t_col1, t_col2, t_col3 = st.columns(3)

    for i, col in enumerate([t_col1, t_col2, t_col3]):
        with col:
            st.image(f"https://via.placeholder.com/150?text=Top{i+2}", use_column_width=True)
            st.write(f"**{top3_others[i]['name']}**")
            st.caption(f"{top3_others[i]['animal_type']} | 일치율 {top3_others[i]['similarity_score']}%")
            if st.button(f"상세보기", key=f"btn_other_{i}"):
                # 백엔드에서 상세 리포트 요청
                partner_name = top3_others[i]['name']
                try:
                    resp = requests.get(
                        f"{API_BASE_URL}/api/matching/{st.session_state.session_id}/partner/{partner_name}"
                    )
                    resp.raise_for_status()
                    detail = resp.json()
                    st.toast(f"{detail['message']}")
                except Exception:
                    st.toast(f"{partner_name}님의 상세 리포트를 생성합니다.")

    st.write("")
    if st.button("← 처음으로 돌아가기", key="back_to_home"):
        # 백엔드 세션 초기화 요청
        try:
            resp = requests.post(
                f"{API_BASE_URL}/api/session/{st.session_state.session_id}/reset"
            )
            resp.raise_for_status()
            result = resp.json()
            # 새 세션 ID로 교체
            st.session_state.session_id = result["session_id"]
        except Exception:
            pass
        
        # 1. 모든 세션 데이터 삭제 (이름, 사진, 설문 결과 등)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # 2. 페이지 정보를 다시 'home'으로 설정
        st.session_state.page = 'home'
        
        # 3. 리런 (초기화된 상태로 홈 화면 렌더링)
        go_to('home')
