#!/usr/bin/env python3
"""
이거돼? — DL 모델 추론 서버
학습 완료 후 이 서버를 실행하면 앱에서 /api/model-inference 로 호출 가능

실행: python3 server.py
포트: 5001
"""

import os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from io import BytesIO
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app)

# ─── 경로 ───
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'

# ─── 모델 정의 (train.py와 동일) ───
class PillModel(nn.Module):
    def __init__(self, num_classes, emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone.num_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def get_embedding(self, x):
        return F.normalize(self.embedding(self.backbone(x)), dim=1)

# ─── 전역 변수 ───
model = None
ref_embeddings = None
ref_names = None
ood_config = None
transform = None
DEVICE = None

def load_model():
    global model, ref_embeddings, ref_names, ood_config, transform, DEVICE

    # 디바이스
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print(f'🔧 디바이스: {DEVICE}')

    # 체크포인트 로드
    ckpt_path = OUTPUT_DIR / 'best_model.pth'
    if not ckpt_path.exists():
        print(f'❌ 모델 파일 없음: {ckpt_path}')
        print('   train.py로 학습을 먼저 완료하세요.')
        sys.exit(1)

    checkpoint = torch.load(str(ckpt_path), map_location=DEVICE)
    num_classes = checkpoint['num_classes']
    emb_dim = checkpoint.get('emb_dim', 512)

    model = PillModel(num_classes, emb_dim).to(DEVICE)
    # ArcFace head는 추론에 불필요 → backbone + embedding만 로드
    state = checkpoint['model_state_dict']
    model_keys = {k for k in model.state_dict().keys()}
    filtered = {k: v for k, v in state.items() if k in model_keys}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f'✅ 모델 로드 완료 (클래스: {num_classes}, 임베딩: {emb_dim}차원)')

    # 레퍼런스 DB
    ref_embeddings = np.load(str(OUTPUT_DIR / 'ref_embeddings.npy'))
    with open(str(OUTPUT_DIR / 'ref_names.json'), 'r', encoding='utf-8') as f:
        ref_names = json.load(f)
    print(f'✅ 레퍼런스 DB: {len(ref_names)}개 약품')

    # OOD 설정
    ood_path = OUTPUT_DIR / 'ood_config.json'
    if ood_path.exists():
        with open(str(ood_path), 'r', encoding='utf-8') as f:
            ood_config = json.load(f)
        print(f'✅ OOD 설정: threshold={ood_config["threshold"]}')
    else:
        ood_config = {'threshold': 0.45, 'negative_label': '__NOT_A_PILL__'}

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@app.route('/api/model-inference', methods=['POST', 'OPTIONS'])
def inference():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'image (base64) 필드 필요'}), 400

        # base64 → PIL Image
        img_b64 = data['image']
        # data:image/jpeg;base64, 접두사 제거
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        img_bytes = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')

        # 추론
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            query = model.get_embedding(img_tensor).cpu().numpy()

        # 코사인 유사도
        sims = np.dot(ref_embeddings, query.T).flatten()

        # 네거티브 레이블 제외하고 약만 매칭
        neg_label = ood_config.get('negative_label', '__NOT_A_PILL__')
        pill_mask = np.array([n != neg_label for n in ref_names])

        sims_pill = sims.copy()
        sims_pill[~pill_mask] = -1  # 네거티브 제외

        top1_sim = float(sims_pill.max())
        top5_idxs = np.argsort(sims_pill)[::-1][:5]

        threshold = ood_config.get('threshold', 0.45)

        # OOD 판정
        if top1_sim < threshold:
            return jsonify({
                'success': True,
                'isPill': False,
                'confidence': top1_sim,
                'threshold': threshold,
                'message': '약으로 인식할 수 없습니다',
                'pills': [],
            })

        # 상위 5개 결과
        results = []
        for idx in top5_idxs:
            name = ref_names[idx]
            if name == neg_label:
                continue
            results.append({
                'drugName': name,
                'similarity': float(sims[idx]),
            })
            if len(results) >= 5:
                break

        return jsonify({
            'success': True,
            'isPill': True,
            'confidence': top1_sim,
            'threshold': threshold,
            'pills': results,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-inference/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'ref_count': len(ref_names) if ref_names else 0,
        'ood_threshold': ood_config.get('threshold', 0.45) if ood_config else None,
    })


if __name__ == '__main__':
    load_model()
    print('\n🚀 추론 서버 시작: http://localhost:5001')
    print('   /api/model-inference      — POST (이미지 분석)')
    print('   /api/model-inference/health — GET  (상태 확인)')
    app.run(host='0.0.0.0', port=5001, debug=False)
