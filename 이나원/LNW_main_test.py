"""
FastAPI Backend - 가치관 매칭 서비스
===================================
Streamlit 프론트엔드(test_frontend.py)의 모든 기능을 FastAPI REST API로 변환.

기능 목록:
1. 홈(Home): 유저 정보 입력 (이름 + 사진 업로드) & DB 존재 여부 체크
2. 설문(Survey): 가치관 설문조사 제출 (자녀계획, 성별선호, 가중치, 10개 시나리오)
3. 매칭(Matching): 동물상 AI 분석 + 매칭 리포트 생성
4. 세션 관리: 세션 초기화 (처음으로 돌아가기)
"""

import os
import io
import uuid
import time
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ==============================================================================
# --- [앱 초기화] ---
# ==============================================================================

app = FastAPI(
    title="가치관 매칭 서비스 API",
    description="가치관 기반 파트너 매칭 서비스의 백엔드 API",
    version="1.0.0",
)

# CORS 설정 (프론트엔드에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 환경에서는 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# --- [설정 및 데이터 로드] ---
# ==============================================================================

# 동물상 매핑 (학습 당시 10개 클래스 순서)
ANIMAL_MAPPING = [
    "🐶 강아지상", "🐱 고양이상", "🐰 토끼상", "🦖 공룡상", "🐻 곰상",
    "🦊 여우상", "🐴 말상", "🐵 원숭이상", "🐭 쥐상", "🐷 돼지상"
]

ANIMAL_CLASSES = [
    "강아지상", "고양이상", "토끼상", "공룡상", "곰상",
    "여우상", "말상", "원숭이상", "쥐상", "돼지상"
]

# 10개 시나리오 질문 리스트 (프론트엔드와 동일)
SURVEY_QUESTIONS = [
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


# --- DB 로드 함수 ---
def load_db() -> pd.DataFrame:
    """기존 DB 로드 (전처리된 CSV 파일)"""
    csv_path = "data/df_weighted_10k.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=0)
    else:
        # 테스트를 위한 가짜 데이터 생성
        return pd.DataFrame(index=["하준우", "김철수", "이영희"])


# 앱 시작 시 DB 로드
df_db = load_db()


# --- AI 모델 로드 함수 ---
def load_animal_model():
    """
    동물상 분석 AI 모델 로드
    - ResNet18 기반, 10개 동물상 클래스 분류
    - 학습된 가중치 파일: ./models/animalface_resnet18_gradcam.pth
    """
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(ANIMAL_CLASSES))

    try:
        model.load_state_dict(
            torch.load("./models/animalface_resnet18_gradcam.pth", map_location="cpu")
        )
        model.eval()
        print("✅ 동물상 분석 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")

    return model


# 앱 시작 시 모델 로드
animal_model = load_animal_model()


# --- 이미지 전처리 함수 ---
def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    업로드된 이미지를 모델 입력 형태로 전처리
    - 224x224 리사이즈
    - ImageNet 정규화
    - (1, C, H, W) 텐서 반환
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  # (1, C, H, W)


# ==============================================================================
# --- [세션 관리 (In-Memory)] ---
# ==============================================================================

# 실제 운영 환경에서는 Redis 등 외부 세션 스토어 사용 권장
sessions: Dict[str, Dict[str, Any]] = {}


def get_or_create_session(session_id: Optional[str] = None) -> tuple:
    """세션 조회 또는 새 세션 생성"""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = str(uuid.uuid4())
    sessions[new_id] = {
        "page": "home",
        "user_name": "",
        "user_photo": None,           # 바이트 데이터 저장
        "user_exists": False,
        "survey_answers": None,
        "user_animal_result": None,
        "user_animal_prob": None,
    }
    return new_id, sessions[new_id]


# ==============================================================================
# --- [Pydantic 모델 (요청/응답 스키마)] ---
# ==============================================================================

# --- 세션 관련 ---
class SessionResponse(BaseModel):
    session_id: str
    page: str
    user_name: str
    user_exists: bool


# --- 유저 정보 체크 응답 ---
class UserCheckResponse(BaseModel):
    session_id: str
    user_name: str
    user_exists: bool
    message: str


# --- 설문 요청 ---
class SurveyRequest(BaseModel):
    session_id: str
    q_children: str = Field(
        ...,
        description="희망 자녀 수",
        examples=["딩크(0명)", "1명", "2명", "3명 이상"],
    )
    q_gender: str = Field(
        ...,
        description="선호 자녀 성별 구성",
        examples=["상관없음", "아들 선호", "딸 선호", "아들/딸 골고루"],
    )
    w_family: str = Field(
        ...,
        description="가족 계획 매칭 가중치",
        examples=["무관", "보통", "중요", "매우 중요"],
    )
    responses: List[int] = Field(
        ...,
        description="10개 시나리오 응답 (각 1~5점)",
        min_length=10,
        max_length=10,
    )


class SurveyResponse(BaseModel):
    session_id: str
    message: str
    survey_answers: Dict[str, Any]


# --- 동물상 분석 응답 ---
class AnimalAnalysisResult(BaseModel):
    animal_type: str        # 예: "🦊 여우상"
    probability: str        # 예: "83%"
    class_name: str         # 예: "여우상"
    class_index: int        # 예: 5


# --- 매칭 파트너 정보 ---
class PartnerProfile(BaseModel):
    name: str
    animal_type: str
    similarity_score: float
    parenting_enthusiasm: float    # 육아 적극성 (0~1)
    education_passion: float       # 교육 열정 (0~1)
    tags: List[str] = []           # 주요 가치관 태그


# --- 매칭 리포트 전체 응답 ---
class MatchingReportResponse(BaseModel):
    session_id: str
    user_profile: Dict[str, Any]
    animal_analysis: AnimalAnalysisResult
    best_match: PartnerProfile
    top3_others: List[PartnerProfile]


# --- 설문 질문 목록 응답 ---
class SurveyQuestionsResponse(BaseModel):
    questions: List[str]
    children_options: List[str]
    gender_options: List[str]
    weight_options: List[str]
    score_range: Dict[str, int]


# --- 사이드바 상태 응답 ---
class SidebarStateResponse(BaseModel):
    session_id: str
    current_page: str
    user_name: str
    steps: Dict[str, str]


# --- 초기화 응답 ---
class ResetResponse(BaseModel):
    session_id: str
    message: str


# ==============================================================================
# --- [API 엔드포인트] ---
# ==============================================================================

# ─────────────────────────────────────────────
# 1. 세션 관리
# ─────────────────────────────────────────────

@app.post("/api/session", response_model=SessionResponse, tags=["세션"])
def create_session():
    """
    새 세션 생성.
    Streamlit의 st.session_state 초기화에 대응.
    """
    session_id, session = get_or_create_session()
    return SessionResponse(
        session_id=session_id,
        page=session["page"],
        user_name=session["user_name"],
        user_exists=session["user_exists"],
    )


@app.get("/api/session/{session_id}", response_model=SessionResponse, tags=["세션"])
def get_session(session_id: str):
    """
    기존 세션 조회.
    Streamlit에서 session_state 값 읽기에 대응.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    session = sessions[session_id]
    return SessionResponse(
        session_id=session_id,
        page=session["page"],
        user_name=session["user_name"],
        user_exists=session["user_exists"],
    )


@app.post("/api/session/{session_id}/reset", response_model=ResetResponse, tags=["세션"])
def reset_session(session_id: str):
    """
    세션 초기화 (처음으로 돌아가기).
    Streamlit의 '← 처음으로 돌아가기' 버튼 로직에 대응:
    - 모든 세션 데이터 삭제
    - 페이지를 'home'으로 리셋
    """
    if session_id in sessions:
        del sessions[session_id]

    new_id, _ = get_or_create_session()
    return ResetResponse(
        session_id=new_id,
        message="세션이 초기화되었습니다. 새 세션이 생성되었습니다.",
    )


# ─────────────────────────────────────────────
# 2. 사이드바 상태
# ─────────────────────────────────────────────

@app.get(
    "/api/session/{session_id}/sidebar",
    response_model=SidebarStateResponse,
    tags=["사이드바"],
)
def get_sidebar_state(session_id: str):
    """
    사이드바 렌더링에 필요한 상태 반환.
    Streamlit의 render_sidebar() 함수에 대응:
    - 현재 페이지 하이라이트
    - 매칭 프로세스 단계 표시
    - 유저 이름 표시
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]
    steps = {
        "home": "1. 기본 정보 입력",
        "survey": "2. 가치관 설문조사",
        "matching": "3. AI 매칭 리포트",
    }
    return SidebarStateResponse(
        session_id=session_id,
        current_page=session["page"],
        user_name=session["user_name"],
        steps=steps,
    )


# ─────────────────────────────────────────────
# 3. 홈(Home) - 유저 정보 입력 & DB 체크
# ─────────────────────────────────────────────

@app.post("/api/user/check", response_model=UserCheckResponse, tags=["홈"])
async def check_user(
    session_id: str = Form(..., description="세션 ID"),
    name: str = Form(..., description="유저 성함"),
    photo: UploadFile = File(..., description="유저 얼굴 사진 (jpg/png/jpeg)"),
):
    """
    유저 이름 + 사진 입력 후 DB 존재 여부 체크.
    Streamlit 홈 화면의 '입력 완료' 버튼 로직에 대응:
    - 이름과 사진 유효성 검증
    - DB에 기존 기록 존재 여부 확인
    - 세션에 유저 정보 저장
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="이름을 입력해주세요.")

    # 사진 파일 확장자 검증
    allowed_extensions = {"jpg", "jpeg", "png"}
    file_ext = photo.filename.split(".")[-1].lower() if photo.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. ({', '.join(allowed_extensions)}만 가능)",
        )

    # 사진 바이트 읽기
    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(status_code=400, detail="사진 파일이 비어 있습니다.")

    # 세션에 저장
    session = sessions[session_id]
    session["user_name"] = name.strip()
    session["user_photo"] = photo_bytes

    # DB 체크 (Streamlit의 `if name in df_db.index:` 에 대응)
    user_exists = name.strip() in df_db.index
    session["user_exists"] = user_exists

    if user_exists:
        message = "기존 기록이 있습니다. 기존 데이터로 매칭하거나 설문을 다시 할 수 있습니다."
    else:
        message = "신규 유저입니다. 설문을 시작해주세요."

    return UserCheckResponse(
        session_id=session_id,
        user_name=name.strip(),
        user_exists=user_exists,
        message=message,
    )


# ─────────────────────────────────────────────
# 4. 페이지 네비게이션
# ─────────────────────────────────────────────

@app.post("/api/session/{session_id}/navigate", tags=["네비게이션"])
def navigate_page(session_id: str, page: str):
    """
    페이지 이동.
    Streamlit의 go_to() 함수에 대응:
    - 'home', 'survey', 'matching' 페이지 간 이동

    Query Parameter:
    - page: 이동할 페이지 이름 (home / survey / matching)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    valid_pages = {"home", "survey", "matching"}
    if page not in valid_pages:
        raise HTTPException(
            status_code=400,
            detail=f"유효하지 않은 페이지입니다. ({', '.join(valid_pages)} 중 선택)",
        )

    # 매칭 페이지 진입 시 조건 체크
    if page == "matching":
        session = sessions[session_id]
        if not session["user_name"] or session["user_photo"] is None:
            raise HTTPException(
                status_code=400,
                detail="매칭 페이지로 이동하려면 먼저 유저 정보를 입력해주세요.",
            )

    # 설문 페이지 진입 시 조건 체크
    if page == "survey":
        session = sessions[session_id]
        if not session["user_name"]:
            raise HTTPException(
                status_code=400,
                detail="설문 페이지로 이동하려면 먼저 이름을 입력해주세요.",
            )

    sessions[session_id]["page"] = page
    return {"session_id": session_id, "page": page, "message": f"'{page}' 페이지로 이동했습니다."}


# ─────────────────────────────────────────────
# 5. 설문(Survey) - 가치관 설문조사
# ─────────────────────────────────────────────

@app.get(
    "/api/survey/questions",
    response_model=SurveyQuestionsResponse,
    tags=["설문"],
)
def get_survey_questions():
    """
    설문 질문 목록 및 선택지 반환.
    Streamlit 설문 화면의 모든 입력 필드 구성에 대응:
    - Part 1: 자녀 및 가족 계획 선택지
    - Part 2: 매칭 가중치 설정 선택지
    - Part 3: 육아 시나리오 10문항 + 점수 범위
    """
    return SurveyQuestionsResponse(
        questions=SURVEY_QUESTIONS,
        children_options=["딩크(0명)", "1명", "2명", "3명 이상"],
        gender_options=["상관없음", "아들 선호", "딸 선호", "아들/딸 골고루"],
        weight_options=["무관", "보통", "중요", "매우 중요"],
        score_range={"min": 1, "max": 5},
    )


@app.post("/api/survey/submit", response_model=SurveyResponse, tags=["설문"])
def submit_survey(request: SurveyRequest):
    """
    설문 결과 제출.
    Streamlit의 설문 Form 제출 로직에 대응:
    - Part 1: 희망 자녀 수 (q_children), 선호 성별 (q_gender)
    - Part 2: 가족 계획 가중치 (w_family)
    - Part 3: 10개 시나리오 응답 (responses, 각 1~5점)
    - 세션에 설문 결과 저장 후 매칭 페이지로 전환
    """
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # 선택지 유효성 검증
    valid_children = {"딩크(0명)", "1명", "2명", "3명 이상"}
    if request.q_children not in valid_children:
        raise HTTPException(status_code=400, detail="유효하지 않은 자녀 수 선택입니다.")

    valid_genders = {"상관없음", "아들 선호", "딸 선호", "아들/딸 골고루"}
    if request.q_gender not in valid_genders:
        raise HTTPException(status_code=400, detail="유효하지 않은 성별 선호 선택입니다.")

    valid_weights = {"무관", "보통", "중요", "매우 중요"}
    if request.w_family not in valid_weights:
        raise HTTPException(status_code=400, detail="유효하지 않은 가중치 선택입니다.")

    # 시나리오 점수 유효성 검증 (각 1~5)
    for i, score in enumerate(request.responses):
        if score < 1 or score > 5:
            raise HTTPException(
                status_code=400,
                detail=f"시나리오 {i+1}번 점수가 유효하지 않습니다. (1~5 사이 값 필요)",
            )

    # 세션에 설문 결과 저장
    survey_answers = {
        "q_children": request.q_children,
        "q_gender": request.q_gender,
        "w_family": request.w_family,
        "responses": request.responses,
    }
    sessions[session_id]["survey_answers"] = survey_answers
    sessions[session_id]["page"] = "matching"

    return SurveyResponse(
        session_id=session_id,
        message="데이터 전송 완료! 당신과 가장 잘 어울리는 파트너를 찾고 있습니다.",
        survey_answers=survey_answers,
    )


# ─────────────────────────────────────────────
# 6. 동물상 분석 (AI 모델 추론)
# ─────────────────────────────────────────────

@app.post(
    "/api/animal/analyze",
    response_model=AnimalAnalysisResult,
    tags=["동물상 분석"],
)
async def analyze_animal(
    session_id: str = Form(..., description="세션 ID"),
    photo: Optional[UploadFile] = File(
        None, description="분석할 사진 (없으면 세션에 저장된 사진 사용)"
    ),
):
    """
    동물상 AI 분석.
    Streamlit 매칭 화면의 동물상 분석 로직에 대응:
    - 세션에 저장된 사진 또는 새로 업로드한 사진을 모델에 입력
    - ResNet18 기반 10개 클래스 분류
    - Softmax 확률 계산 후 최고 확률 클래스 반환
    - 결과를 세션에 캐싱 (이미 분석된 경우 재사용)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]

    # 이미 분석 결과가 있으면 캐싱된 결과 반환
    # (Streamlit의 `if 'user_animal_result' not in st.session_state:` 에 대응)
    if session["user_animal_result"] is not None:
        # 캐싱된 결과에서 클래스 인덱스 복원
        cached_type = session["user_animal_result"]
        cached_prob = session["user_animal_prob"]
        class_idx = next(
            (i for i, m in enumerate(ANIMAL_MAPPING) if m == cached_type), -1
        )
        class_name = ANIMAL_CLASSES[class_idx] if class_idx >= 0 else "미확인"
        return AnimalAnalysisResult(
            animal_type=cached_type,
            probability=cached_prob,
            class_name=class_name,
            class_index=class_idx,
        )

    # 사진 데이터 결정 (새 업로드 > 세션 저장 사진)
    if photo is not None:
        photo_bytes = await photo.read()
    elif session["user_photo"] is not None:
        photo_bytes = session["user_photo"]
    else:
        raise HTTPException(
            status_code=400,
            detail="분석할 사진이 없습니다. 사진을 업로드해주세요.",
        )

    # 모델 체크
    if animal_model is None:
        # 방어 코드 (Streamlit의 else 분기에 대응)
        session["user_animal_result"] = "❓ 미확인상"
        session["user_animal_prob"] = "0%"
        return AnimalAnalysisResult(
            animal_type="❓ 미확인상",
            probability="0%",
            class_name="미확인",
            class_index=-1,
        )

    try:
        # 1. 전처리
        input_tensor = preprocess_image(photo_bytes)

        # 2. 추론
        with torch.no_grad():
            outputs = animal_model(input_tensor)

            # Softmax로 확률 계산
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # 가장 높은 확률과 인덱스 추출
            conf, predicted = torch.max(probs, 1)

            result_idx = predicted.item()
            probability = conf.item() * 100  # 퍼센트 변환

            # 3. 결과 저장 (이모지 포함 이름 + 확률)
            animal_type = ANIMAL_MAPPING[result_idx]
            prob_str = f"{probability:.0f}%"

            session["user_animal_result"] = animal_type
            session["user_animal_prob"] = prob_str

            return AnimalAnalysisResult(
                animal_type=animal_type,
                probability=prob_str,
                class_name=ANIMAL_CLASSES[result_idx],
                class_index=result_idx,
            )

    except Exception as e:
        # 오류 시 방어 처리
        session["user_animal_result"] = "❓ 미확인상"
        session["user_animal_prob"] = "0%"
        raise HTTPException(
            status_code=500,
            detail=f"동물상 분석 중 오류가 발생했습니다: {str(e)}",
        )


# ─────────────────────────────────────────────
# 7. 매칭 리포트 (종합)
# ─────────────────────────────────────────────

@app.get(
    "/api/matching/{session_id}",
    response_model=MatchingReportResponse,
    tags=["매칭"],
)
async def get_matching_report(session_id: str):
    """
    AI 가치관 매칭 리포트 생성.
    Streamlit 매칭 화면 전체에 대응:

    1. 동물상 분석 실행 (아직 안 된 경우)
    2. 유저 프로필 구성 (이름, 동물상, 가치관 태그, 육아적극성, 교육열정)
    3. BEST 매칭 파트너 추천
    4. TOP 3 추가 추천 파트너

    실제 운영 시에는 설문 결과 + DB 기반 유사도 계산 로직 적용 필요.
    현재는 Streamlit과 동일한 임시(하드코딩) 매칭 데이터를 반환.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]

    if not session["user_name"] or session["user_photo"] is None:
        raise HTTPException(
            status_code=400,
            detail="매칭 리포트를 생성하려면 먼저 유저 정보를 입력해주세요.",
        )

    # --- 동물상 분석 (아직 분석되지 않은 경우 자동 실행) ---
    if session["user_animal_result"] is None:
        if animal_model is not None and session["user_photo"] is not None:
            try:
                input_tensor = preprocess_image(session["user_photo"])
                with torch.no_grad():
                    outputs = animal_model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, predicted = torch.max(probs, 1)
                    result_idx = predicted.item()
                    probability = conf.item() * 100

                    session["user_animal_result"] = ANIMAL_MAPPING[result_idx]
                    session["user_animal_prob"] = f"{probability:.0f}%"
            except Exception:
                session["user_animal_result"] = "❓ 미확인상"
                session["user_animal_prob"] = "0%"
        else:
            session["user_animal_result"] = "❓ 미확인상"
            session["user_animal_prob"] = "0%"

    # --- 유저 프로필 구성 ---
    user_profile = {
        "name": session["user_name"],
        "animal_type": session["user_animal_result"],
        "animal_probability": session["user_animal_prob"],
        "tags": ["#자율성", "#체험학습", "#딩크희망"],  # 실제로는 설문 결과 기반으로 생성
        "parenting_enthusiasm": 0.85,  # 실제로는 설문 결과 기반으로 계산
        "education_passion": 0.40,     # 실제로는 설문 결과 기반으로 계산
    }

    # --- 동물상 분석 결과 객체 ---
    animal_type_str = session["user_animal_result"]
    class_idx = next(
        (i for i, m in enumerate(ANIMAL_MAPPING) if m == animal_type_str), -1
    )
    animal_analysis = AnimalAnalysisResult(
        animal_type=animal_type_str,
        probability=session["user_animal_prob"],
        class_name=ANIMAL_CLASSES[class_idx] if class_idx >= 0 else "미확인",
        class_index=class_idx,
    )

    # --- BEST 매칭 파트너 (Streamlit의 임시 데이터와 동일) ---
    best_match = PartnerProfile(
        name="이서연",
        animal_type="🐱 고양이상",
        similarity_score=94.8,
        parenting_enthusiasm=0.82,
        education_passion=0.45,
        tags=["#소통중시", "#자연육아", "#균형잡힌"],
    )

    # --- TOP 3 추가 추천 파트너 (Streamlit의 others 리스트와 동일) ---
    top3_others = [
        PartnerProfile(
            name="김민수",
            animal_type="🐰 토끼상",
            similarity_score=88.2,
            parenting_enthusiasm=0.78,
            education_passion=0.55,
            tags=["#교육열정", "#활동적"],
        ),
        PartnerProfile(
            name="박지혜",
            animal_type="🦊 여우상",
            similarity_score=85.5,
            parenting_enthusiasm=0.75,
            education_passion=0.60,
            tags=["#창의교육", "#독립심"],
        ),
        PartnerProfile(
            name="최진우",
            animal_type="🐻 곰상",
            similarity_score=82.9,
            parenting_enthusiasm=0.80,
            education_passion=0.50,
            tags=["#가정중심", "#안정추구"],
        ),
    ]

    # 페이지 상태 업데이트
    session["page"] = "matching"

    return MatchingReportResponse(
        session_id=session_id,
        user_profile=user_profile,
        animal_analysis=animal_analysis,
        best_match=best_match,
        top3_others=top3_others,
    )


# ─────────────────────────────────────────────
# 8. 기존 유저 데이터로 바로 매칭 (기존 데이터로 매칭 버튼)
# ─────────────────────────────────────────────

@app.post("/api/matching/{session_id}/existing", tags=["매칭"])
def match_with_existing_data(session_id: str):
    """
    기존 유저의 저장된 데이터로 매칭 진행.
    Streamlit의 '기존 데이터로 매칭' 버튼에 대응:
    - user_exists == True인 경우에만 사용 가능
    - 설문 없이 바로 매칭 페이지로 이동
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    session = sessions[session_id]
    if not session["user_exists"]:
        raise HTTPException(
            status_code=400,
            detail="기존 데이터가 없습니다. 설문을 먼저 진행해주세요.",
        )

    session["page"] = "matching"
    return {
        "session_id": session_id,
        "page": "matching",
        "message": "기존 데이터를 기반으로 매칭을 시작합니다.",
    }


# ─────────────────────────────────────────────
# 9. 추천 파트너 상세보기
# ─────────────────────────────────────────────

@app.get("/api/matching/{session_id}/partner/{partner_name}", tags=["매칭"])
def get_partner_detail(session_id: str, partner_name: str):
    """
    추천 파트너 상세 리포트 조회.
    Streamlit의 '상세보기' 버튼에 대응:
    - st.toast(f"{others[i]['name']}님의 상세 리포트를 생성합니다.")
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    # 임시 상세 데이터 (실제로는 DB에서 조회)
    partner_details = {
        "김민수": {
            "name": "김민수",
            "animal_type": "🐰 토끼상",
            "similarity_score": 88.2,
            "parenting_enthusiasm": 0.78,
            "education_passion": 0.55,
            "tags": ["#교육열정", "#활동적"],
            "detailed_comparison": {
                "자녀계획_일치": True,
                "양육관_유사도": 0.87,
                "교육관_유사도": 0.72,
            },
        },
        "박지혜": {
            "name": "박지혜",
            "animal_type": "🦊 여우상",
            "similarity_score": 85.5,
            "parenting_enthusiasm": 0.75,
            "education_passion": 0.60,
            "tags": ["#창의교육", "#독립심"],
            "detailed_comparison": {
                "자녀계획_일치": True,
                "양육관_유사도": 0.83,
                "교육관_유사도": 0.78,
            },
        },
        "최진우": {
            "name": "최진우",
            "animal_type": "🐻 곰상",
            "similarity_score": 82.9,
            "parenting_enthusiasm": 0.80,
            "education_passion": 0.50,
            "tags": ["#가정중심", "#안정추구"],
            "detailed_comparison": {
                "자녀계획_일치": False,
                "양육관_유사도": 0.80,
                "교육관_유사도": 0.68,
            },
        },
        "이서연": {
            "name": "이서연",
            "animal_type": "🐱 고양이상",
            "similarity_score": 94.8,
            "parenting_enthusiasm": 0.82,
            "education_passion": 0.45,
            "tags": ["#소통중시", "#자연육아", "#균형잡힌"],
            "detailed_comparison": {
                "자녀계획_일치": True,
                "양육관_유사도": 0.92,
                "교육관_유사도": 0.85,
            },
        },
    }

    if partner_name not in partner_details:
        raise HTTPException(
            status_code=404,
            detail=f"'{partner_name}'님의 데이터를 찾을 수 없습니다.",
        )

    return {
        "session_id": session_id,
        "partner": partner_details[partner_name],
        "message": f"{partner_name}님의 상세 리포트를 생성합니다.",
    }


# ─────────────────────────────────────────────
# 10. DB 정보 조회 (보조 엔드포인트)
# ─────────────────────────────────────────────

@app.get("/api/db/info", tags=["DB"])
def get_db_info():
    """
    현재 로드된 DB의 기본 정보 반환.
    Streamlit의 load_db() 결과 확인용.
    """
    return {
        "total_users": len(df_db),
        "user_names": df_db.index.tolist()[:20],  # 최대 20명까지만 반환 (보안)
        "columns": df_db.columns.tolist(),
        "db_file_exists": os.path.exists("data/df_weighted_10k.csv"),
    }


# ─────────────────────────────────────────────
# 11. 동물상 매핑 정보 (프론트엔드 참조용)
# ─────────────────────────────────────────────

@app.get("/api/animal/mapping", tags=["동물상 분석"])
def get_animal_mapping():
    """
    동물상 클래스 매핑 정보 반환.
    프론트엔드에서 동물상 표시에 필요한 전체 매핑 목록.
    """
    return {
        "animal_mapping": ANIMAL_MAPPING,
        "animal_classes": ANIMAL_CLASSES,
        "total_classes": len(ANIMAL_CLASSES),
    }


# ─────────────────────────────────────────────
# 12. 헬스체크
# ─────────────────────────────────────────────

@app.get("/health", tags=["시스템"])
def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "model_loaded": animal_model is not None,
        "db_loaded": df_db is not None and len(df_db) > 0,
        "active_sessions": len(sessions),
    }


# ==============================================================================
# --- [서버 실행] ---
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
