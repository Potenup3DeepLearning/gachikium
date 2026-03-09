import streamlit as st
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

#================================================================================
# --- [설정 및 데이터 로드] ---
#================================================================================

st.set_page_config(page_title="가치관 매칭 서비스", layout="centered")

# 세션 상태 초기화 (페이지 이동 및 데이터 저장용)
if 'page' not in st.session_state:
    st.session_state.page = 'home'      # 현재 페이지
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""     # 유저 이름
if 'user_photo' not in st.session_state:
    st.session_state.user_photo = None  # 유저 사진

# 기존 DB 로드 (전처리된 파일이 있다고 가정)
@st.cache_data
def load_db():
    # 파일이 없으면 빈 데이터프레임 생성 (테스트용)
    if os.path.exists('data/df_weighted_10k.csv'):
        return pd.read_csv('data/df_weighted_10k.csv', index_col=0)
    else:
        # 테스트를 위해 가짜 데이터 생성
        return pd.DataFrame(index=['하준우', '김철수', '이영희'])

df_db = load_db()

# 학습 당시 10개 클래스 순서에 맞춰 이모지와 이름을 매핑하세요.
animal_mapping = [
    "🐶 강아지상", "🐱 고양이상", "🐰 토끼상", "🦖 공룡상", "🐻 곰상",
    "🦊 여우상", "🐴 말상", "🐵 원숭이상", "🐭 쥐상", "🐷 돼지상"
]
#================================================================================
# --- [AI 모델 로드 함수] ---
#================================================================================
@st.cache_resource
def load_animal_model():
    # 1. 모델 구조 정의 (기존 pth를 만들 때 사용한 클래스 구조와 동일해야 합니다)
    # 예시: ResNet이나 간단한 CNN 구조라고 가정
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len( ["강아지상", "고양이상", "토끼상", "공룡상", "곰상", "여우상", "말상", "원숭이상", "쥐상", "돼지상"]))
    

    # 만약 아키텍처가 복잡하다면 해당 클래스 파일을 import 하세요.
    # model = torch.load('models/animal_model_full.pth') # 전체 저장 방식일 때
    
    # state_dict 저장 방식일 때 (권장):
    # model.load_state_dict(torch.load('models/animal_model.pth', map_location='cpu'))
    
    
     
    try:
        model.load_state_dict(torch.load('./models/animalface_resnet18_gradcam.pth', map_location='cpu')) 
        model.eval()
        print("✅ 동물상 분석 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        
    return model

# --- [이미지 전처리 함수] ---
def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    # 모델 학습 시 사용했던 이미지 크기와 정규화 값을 넣으세요 (예: 224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0) # (1, C, H, W)

# 모델 전역 변수화
animal_model = load_animal_model()
animal_classes = ["강아지상", "고양이상", "토끼상", "공룡상", "곰상", "여우상", "말상", "원숭이상", "쥐상", "돼지상"]



#================================================================================
# --- [페이지 네비게이션 로직] ---
#================================================================================
def go_to(page_name):
    st.session_state.page = page_name
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
        
        # 사진이 업로드되었다면 작게 미리보기 (선택 사항)
        if st.session_state.user_photo:
            st.image(st.session_state.user_photo, caption="분석용 프로필", width=100)


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
        

    # 로직 체크 및 다음 단계 진행
    if check_clicked:
        if name and photo:
            st.session_state.user_name = name
            st.session_state.user_photo = photo
            
            # DB 체크 후 '상태'만 표시하고, 실제 이동 버튼은 따로 둡니다.
            if name in df_db.index:
                st.session_state.user_exists = True
            else:
                st.session_state.user_exists = False
        else:
            st.error("이름과 사진을 모두 입력해주세요.")

    # 버튼 클릭 후 나타나는 선택지 (중첩 방지를 위해 if check_clicked 밖에 위치하거나 별도 처리)
    if st.session_state.user_name and st.session_state.user_photo:
        if getattr(st.session_state, 'user_exists', False):
            st.info(f"기존 기록이 있습니다. 어떻게 할까요?")
            c1, c2 = st.columns(2)
            if c1.button("설문 다시 하기"):
                go_to('survey')
            if c2.button("기존 데이터로 매칭"):
                go_to('matching')
        else:
            st.success("신규 유저입니다. 설문을 시작해주세요.")
            if st.button("설문 시작하기 ✍️"):
                go_to('survey') # 여기서 확실히 survey로 보냅니다.

# --- [2. 설문 화면 (survey) 통합 버전] ---
elif st.session_state.page == 'survey':
    st.title("📝 가치관 설문조사")
    st.caption("개인정보 수집 및 이용 동의 안내 : 성함, 가치관 설문 내용 등 개인정보를 수집하여 서비스 제공합니다. 2차 프로젝트 기한 이후 즉시 파기 예정입니다. 귀하는 개인정보 수집 및 이요에 대한 동의를 거부할 권리가 있습니다. 다만, 거부할 경우 설문 참여가 제한됩니다. ")

    with st.container():
        # 전체 설문을 하나의 Form으로 감쌉니다.
        with st.form("total_survey_form"):
            
            # --- Section 0: 당신의 정보 ---
            st.subheader("📍 Part 0. 당신의 정보")
            ideal_type = st.radio(
                "0. 당신의 이상형",
                ["강아지상", "고양이상", "사슴상", "토끼상", '꼬부기상','곰상','쿼카상','쥐상','도룡뇽상','여우상','말상','너구리상','꽃돼지상','햄스터상','늑대상','코알라상','오리상','원숭이','공룡상','알파카상, 라마상'], index=1, key="final_ideal_type"
            )
            st.markdown("---")

            # --- Section 1: 기본 가족 계획 ---
            st.subheader("📍 Part 1. 자녀 계획 및 가족 구성")
            p_children_count = st.radio(
                "1-1. 희망하는 자녀 수",
                ["1명", "2명", "3명","그 이상"], index=1, key="final_p_children_count"
            )
            p_children_composition = st.radio(
                "1-2. 희망하는 자녀 구성",
                ["오직 딸", "오직 아들", "딸 1명, 아들 1명"], index=1,key="final_p_children_composition"
            )

            p_children_timing = st.radio(
                "1-3. 자녀 갖고 싶은 시기",
                ["결혼 즉시", "결혼 후 1~2년 이내", "결혼 후 3~5년 이내",'경제적 안정 후'], index=1,key="final_p_children_timing"
            )

            p_infertility_alternative = st.radio(
                "1-4. 생물학적 출산이 어려움 발생 시 대안",
                ["의학적 도움 적극 시도", "입양 고려", "무자녀"], index=1,key="final_p_infertility_alternative"
            )

            st.caption("💡 중요도가 높을수록 해당 조건이 맞는 사람을 우선적으로 추천합니다.")

            imp_family_plan = st.select_slider(
                '"1. 자녀 계획 및 가족 구성 항목"에 대해 중요도(1점: 상대방에 맞출 수 있다. ~ 5점: 양보할 수 없다.)',
                options=[1, 2, 3,4, 5], value=3, key="final_imp_family_plan"
            )

            st.markdown("---")
            
            # --- Section 2: 항목 중요도 ---
            st.subheader("📍 Part 2. 시나리오 기반 자녀 가치관 분석")
            st.markdown("---")

            sc_toothbrushing = st.select_slider(
                '아이가 "오늘만 양치 안하고 그냥 자면 안돼요? 라고 칭얼거릴 때 어떻게 하시겠습니까?',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_toothbrushing"
            )
            st.caption('1점: "졸려도 위생 관리는 필수야. 세수와 양치는 타협할 수 없는 규칙이야"')
            st.caption('5점: "정말 피곤한 날도 있지. 오늘은 특별히 그냥 자고, 내일 아침에 깨끗이 닦자."')
            st.markdown("---")
            sc_bedtime_story = st.select_slider(
                '평소 밤 9시에 자기로 약속했습니다. 그런데 오늘 아이가 읽고 싶어 하던 동화책 시리즈의 마지막 권을 다 읽고 싶다며 30분만 더 시간을 달라고 합니다.',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_bedtime_story"
            )
            st.caption('1점: "수면 패턴이 깨지면 내일 힘들어. 약속한 시간이니 바로 불을 끈다."')
            st.caption('5점: "책 읽는 즐거움이 크구나! 오늘만 특별히 끝까지 읽고 자게 해준다."')
            st.markdown("---")
            sc_competition_2nd = st.select_slider(
                '경쟁 상황에서의 태도 아이가 운동 경기나 대회에서 아쉽게 2등을 했습니다. 아이는 충분히 잘했다고 기뻐하는데, 당신의 마음속 생각은?',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_competition_2nd"
            )
            st.caption('1점:  "조금만 더 보완하면 1등 할 수 있었는데 아쉽다. 다음엔 우승을 목표로 해보자."')
            st.caption('5점:  "결과와 상관없이 경기를 즐기고 만족해하는 아이의 모습이 대견하고 안심된다."')
            st.markdown("---")
            sc_talent_education = st.select_slider(
                '재능 발견과 교육 아이가 특정 분야(예: 피아노, 운동)에 천재적인 재능을 보입니다. 이때 당신의 교육 방향은?',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_talent_education"
            )
            st.caption('1점:  "재능이 썩지 않게 전문적인 훈련과 경쟁을 통해 세계 최고 수준으로 키우고 싶다."')
            st.caption('5점:  "재능도 좋지만 아이가 스트레스받지 않고 취미로서 즐기며 행복하게 크는 게 먼저다."')
            st.markdown("---")
            sc_discipline_conflict = st.select_slider(
                '두 사람의 훈육 방식이 부딪힐 때, 누구의 의견을 따라야 한다고 생각하시나요?',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_discipline_conflict"
            )
            st.caption('1점:  "끝장 토론을 해서라도 두 사람이 완벽하게 합의된 하나의 결론을 도출해야 한다."')
            st.caption('5점:  "아이와 더 많은 시간을 보내는 주양육자의 판단을 존중하고 그 방식에 맞춰주는 게 맞다."')
            st.markdown("---")
            sc_play_vs_chores = st.select_slider(
                '한 명은 퇴근 후 아이와 놀아주고, 한 명은 밀린 집안일을 해야 하는 상황입니다.',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_play_vs_chores"
            )
            st.caption('1점:  "아이 돌봄도, 집안일도 매번 역할을 바꿔가며 둘 다 똑같이 경험해야 한다."')
            st.caption('5점: "각자 더 잘하는 분야(예: 요리 vs 몸으로 놀아주기)를 정해 확실하게 역할을 나누는 게 편하다."')
            st.markdown("---")
            sc_grandparents_help = st.select_slider(
                '맞벌이 상황 등에서 조부모님이 아이를 봐주겠다고 제안하신다면?',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_grandparents_help"
            )
            st.caption('1점:"우리 아이는 우리가 결정한다. 간섭으로 느껴지므로 정중히 선을 긋고 거절한다."')
            st.caption('5점:"어른들의 오랜 경험에서 나오는 지혜다. 우리와 조금 달라도 귀담아듣고 따를 수 있다."')
            st.markdown("---")
            sc_inlaws_advice = st.select_slider(
                '양가 어르신들이 본인의 가치관과 다른 육아 조언(예: "애를 너무 손타게 키운다", "사탕 좀 주면 어떠냐")을 하실 때 당신의 생각은?',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_inlaws_advice"
            )
            st.caption('1점: "우리 아이는 우리가 결정한다. 간섭으로 느껴지므로 정중히 선을 긋고 거절한다."')
            st.caption('5점: "어른들의 오랜 경험에서 나오는 지혜다. 우리와 조금 달라도 귀담아듣고 따를 수 있다."')
            st.markdown("---")
            sc_rainy_zoo = st.select_slider(
                '주말에 아이와 동물원에 가기로 했는데, 아침에 일어나니 갑자기 비가 옵니다. 이때 당신의 반응은?',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_rainy_zoo"
            )
            st.caption('1점: "미리 짜놓은 계획이 틀어져 스트레스를 받으며, 즉시 대체할 수 있는 실내 플랜 B를 가동한다."')
            st.caption('5점: "비 오는 날도 나름의 운치가 있지! 집에서 전을 부쳐 먹거나 그냥 비 구경을 하며 느긋하게 보낸다."')
            st.markdown("---")
            sc_education_fund_risk = st.select_slider(
                '아이의 교육 자금이나 미래 리스크를 대비하는 당신의 생각은?',
                options=[1, 2, 3,4, 5], value=3, key="final_sc_education_fund_risk"
            )
            st.caption('1점: "아이가 태어나기 전후부터 대학 등록금이나 장래 비용을 위한 구체적인 저축/보험 플랜을 시작한다."')
            st.caption('5점: 미래의 걱정보다는 현재 아이에게 필요한 것에 집중하며, 상황이 닥쳤을 때 유연하게 대처하면 된다.')
            st.markdown("---")

            # --- Section 3: 육아 시나리오 ---
            st.subheader("📍 Part 4. 경제적 지원 및 가사 분담")
            
            e_childcare_cost_share = st.radio(
                "4-1. 자녀 교육비/양육비 지출 비중",
                ["노후보단 자녀 교육", "노후 먼저, 남는 예산으로 지원"], index=1, key="final_e_childcare_cost_share"
            )
            e_parental_leave_burden = st.radio(
                "4-2. 육아 휴직, 양육 부담",
                ["경제력 높은 사람 일하고, 한명은 전담 육아", "맞벌이하면서 외부 도움(조부모, 시터)"], index=1,key="final_e_parental_leave_burden"
            )

            st.caption("💡 중요도가 높을수록 해당 조건이 맞는 사람을 우선적으로 추천합니다.")

            imp_econ_housework = st.select_slider(
                '"4. 경제적 지원 및 가사 분담"에 대해 중요도 (1점: 상대방에 맞출 수 있다. ~ 5점: 양보할 수 없다.)',
                options=[1, 2, 3,4, 5], value=3, key="final_imp_econ_housework"
            )

            st.markdown("---")
           
            # --- Section 5: 자녀 가치관 ---
            st.subheader("📍 Part 5. 자녀의 가치")
            
            child_values_open = st.radio(
                "5-1. 자녀 가치관, 어떤 사람이 되길 바라는가? ",
                ["경제적 성공, 사회적 지위", "도덕적, 타인 배려",'자신이 좋아하는 일, 행복','회복탄력성, 생활력 강한 사람'], index=1, key="final_child_values_open"
            )
            
            st.caption("💡 중요도가 높을수록 해당 조건이 맞는 사람을 우선적으로 추천합니다.")

            imp_child_values = st.select_slider(
                '"5. 자녀 가치관"에 대한 중요도 (1점: 상대방에 맞출 수 있다. ~ 5점: 양보할 수 없다.)',
                options=[1, 2, 3,4, 5], value=3, key="final_imp_child_values"
            )

            st.markdown("---")

            # --- 최종 제출 버튼 ---
            submit_button = st.form_submit_button("설문 완료 및 AI 매칭 시작 🚀")

            if submit_button:
                # 모든 데이터 세션에 저장
                st.session_state.survey_answers = {
                    "ideal_type": ideal_type,
                    "p_children_count": p_children_count,
                    "p_children_composition": p_children_composition,
                    'p_children_timing' :p_children_timing,
                    'p_infertility_alternative' :p_infertility_alternative,
                    'imp_family_plan':imp_family_plan,
                    'sc_toothbrushing':sc_toothbrushing,
                    'sc_bedtime_story' :sc_bedtime_story,
                    'sc_competition_2nd' :sc_competition_2nd,
                    'sc_talent_education':sc_talent_education,
                    'sc_discipline_conflict':sc_discipline_conflict,
                    'sc_play_vs_chores':sc_play_vs_chores,
                    'sc_grandparents_help':sc_grandparents_help,
                    'sc_inlaws_advice':sc_inlaws_advice,
                    'sc_rainy_zoo':sc_rainy_zoo,
                    'sc_education_fund_risk':sc_education_fund_risk,
                    'e_childcare_cost_share':e_childcare_cost_share,
                    'e_parental_leave_burden':e_parental_leave_burden,
                    'imp_econ_housework':imp_econ_housework,
                    'child_values_open':child_values_open,
                    'imp_child_values':imp_child_values
                }
                
                # 성공 메시지와 함께 페이지 이동
                st.success("데이터 전송 완료! 당신과 가장 잘 어울리는 파트너를 찾고 있습니다.")
                go_to('matching')

# --- [3. 매칭 화면 (matching) 통합 버전] ---
elif st.session_state.page == 'matching':
    
    # --- [동물상 분석 실행] ---
    if 'user_animal_result' not in st.session_state:
        with st.spinner('AI가 당신의 사진에서 동물상을 분석하고 있습니다...'):
            if animal_model is not None and st.session_state.user_photo is not None:
                # 1. 전처리
                input_tensor = preprocess_image(st.session_state.user_photo)
                # 2. 추론
                with torch.no_grad():
                    outputs = animal_model(input_tensor)
                   # 1. Softmax로 확률 계산
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # 2. 가장 높은 확률과 인덱스 추출
                    conf, predicted = torch.max(probs, 1)
                    
                    result_idx = predicted.item()
                    probability = conf.item() * 100 # 퍼센트 변환

                    # 3. 결과 저장 (이모지 포함 이름 + 확률)
                    st.session_state.user_animal_result = animal_mapping[result_idx]
                    st.session_state.user_animal_prob = f"{probability:.0f}%" # 반올림 정수
            else:
                # 방어 코드
                st.session_state.user_animal_result = "❓ 미확인상"
                st.session_state.user_animal_prob = "0%"
    
    
    
    st.title("💖 AI 가치관 매칭 리포트")
    
    # 상단: 유저 정보
    st.subheader(f"✨ {st.session_state.user_name}님을 위한 맞춤 분석 결과입니다.")
    st.markdown("---")

    # 공통 설정
    fixed_height = "250px" # 이모지 박스이므로 높이를 살짝 줄여서 한눈에 들어오게 조절

    # (추후 알고리즘 연동 시 DB에서 가져올 임시 데이터)
    target_name = "이서연"
    target_emoji = "🐱" # 실제론 DB에서 가져올 정보
    similarity_score = 94.8
    
    with st.spinner('가치관 데이터를 분석하고 최적의 파트너를 찾는 중...'):
        import time
        time.sleep(1) 

    # 중앙: 본인 vs 매칭 상대 비교 (Col 2)
    col1, col2 = st.columns(2)

    with col1:
        
        st.markdown(f"#### 👤 나의 프로필")

        # 내 동물상 이모지만 추출 (예: "🐶 강아지상" -> "🐶")
        my_emoji = st.session_state.user_animal_result.split()[0] if st.session_state.user_animal_result else "👤"

        # 구버전 호환: use_column_width=True
        st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border: 2px solid #1E3A8A;
                border-radius: 15px;
                height: {fixed_height};
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                margin-bottom: 15px;
                box-shadow: 0 4px 15px rgba(30, 58, 138, 0.1);
            ">
                <div style="font-size: 100px;">{my_emoji}</div>
                <div style="color: #1E3A8A; font-weight: bold; margin-top: 10px;">나의 분석 타입</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.info(f"**이름:** {st.session_state.user_name}\n\n**동물상 :** {st.session_state.user_animal_result}({st.session_state.user_animal_prob})")

        with st.container():
            # 카드 스타일 적용 (이전에 정의한 CSS 클래스 활용)
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            
            st.write("**주요 가치관:** #자율성 #체험학습 #딩크희망")
            
            # ... 나머지 코드
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 간단한 가치관 시각화 (Progress bar 활용)
            st.write("육아 적극성")
            st.progress(0.85)
            st.write("교육 열정")
            st.progress(0.40)

    with col2:
        st.markdown(f"#### 💝 BEST 매칭 파트너")
        st.markdown(f"""
            <div style="
                background-color: #ffffff;
                border: 2px solid #1E3A8A;
                border-radius: 15px;
                height: {fixed_height};
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                margin-bottom: 15px;
                box-shadow: 0 4px 15px rgba(30, 58, 138, 0.1);
            ">
                <div style="font-size: 100px;">{target_emoji}</div>
                <div style="color: #1E3A8A; font-weight: bold; margin-top: 10px;">베스트 매칭 결과</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.success(f"**이름:** {target_name}\n\n**동물상 :** 🐱 고양이상(94%)")
        
        with st.container():
            
            st.write("**주요 가치관:** #자율성 #체험학습 #딩크희망")

            # 상대방 수치 (나와 비교되게)
            st.write("육아 적극성")
            st.progress(0.82)
            st.write("교육 열정")
            st.progress(0.45)
            
    st.metric("매칭 일치율", f"{similarity_score}%")

    st.markdown("---")

    # 하단: 그외 추천 파트너 TOP 3
    st.subheader("🔍 또 다른 인연들을 확인해보세요 (TOP 3)")
    
    t_col1, t_col2, t_col3 = st.columns(3)
    
    others = [
        {"name": "김민수", "score": 88.2, "animal": "🐰"},
        {"name": "박지혜", "score": 85.5, "animal": "🦊"},
        {"name": "최진우", "score": 82.9, "animal": "🐻"}
    ]

    for i, col in enumerate([t_col1, t_col2, t_col3]):
        with col:
            st.markdown(f"""
                <div style="
                    background-color: #ffffff;
                    border: 1px solid #e6e9ef;
                    border-radius: 10px;
                    height: 150px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 60px;
                    margin-bottom: 10px;
                ">
                    {others[i]['animal']}
                </div>
                """, unsafe_allow_html=True)
            st.write(f"**{others[i]['name']}**")
            st.caption(f"일치율 {others[i]['score']}%")
            if st.button(f"상세보기", key=f"btn_other_{i}"):
                st.info(f"{others[i]['name']}님과의 매칭 리포트 생성 중...")

    st.write("")
    if st.button("← 처음으로 돌아가기", key="back_to_home"):
        # 1. 모든 세션 데이터 삭제 (이름, 사진, 설문 결과 등)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # 2. 페이지 정보를 다시 'home'으로 설정
        st.session_state.page = 'home'
        
        # 3. 리런 (초기화된 상태로 홈 화면 렌더링)
        go_to('home')