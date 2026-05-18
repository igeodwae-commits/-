#!/bin/bash
# ═══════════════════════════════════════════════
#  이거돼? 약 인식 모델 학습 (더블클릭으로 실행)
# ═══════════════════════════════════════════════

cd "$(dirname "$0")"

echo "============================================"
echo "  이거돼? 모델 학습 시작"
echo "============================================"
echo ""

# Python 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3이 없습니다!"
    echo "   brew install python3 실행 후 다시 시도하세요"
    read -p "아무 키나 누르세요..."
    exit 1
fi

# 가상환경 생성 (최초 1회)
if [ ! -d "venv" ]; then
    echo "📦 가상환경 생성 중..."
    python3 -m venv venv
    echo "✅ 가상환경 생성 완료"
fi

# 가상환경 활성화
source venv/bin/activate

# 패키지 설치 (최초 1회)
if [ ! -f "venv/.installed" ]; then
    echo ""
    echo "📦 패키지 설치 중... (최초 1회, 5~10분 소요)"
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.installed
    echo "✅ 패키지 설치 완료"
fi

echo ""
echo "🚀 학습 시작!"
echo ""

# 학습 실행
python3 train.py

echo ""
echo "============================================"
echo "  학습 완료! output/ 폴더를 확인하세요"
echo "============================================"
read -p "아무 키나 누르세요..."
