#!/usr/bin/env python3
"""
이거돼? — 약 이미지 인식 모델 학습 (ArcFace Metric Learning)
M4 Pro Mac Mini 최적화 버전

방법 1: 고급 Augmentation (배경합성, 조명, 블러, 노이즈)
방법 2: AI 스타일 증강 (CutMix, 배경 생성, 멀티뷰 합성)
OOD 방어: 네거티브 이미지 학습 + 코사인 유사도 임계값

실행: python3 train.py
전체 소요시간: 약 4~6시간 (데이터 다운로드 + 네거티브 생성 + 증강 + 학습)
"""

import os, sys, json, time, math, random
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ════════════════════════════════════════════════════════════
#  설정
# ════════════════════════════════════════════════════════════

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / 'data'
IMAGE_DIR  = DATA_DIR / 'images'
OUTPUT_DIR = BASE_DIR / 'output'

API_KEY    = '73d5d41569516c5a7259f3928ea2de8af44ded772f2a6cb4d6f5dbe27b0f62cd'
MFDS_URL   = 'https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03'

EMB_DIM    = 512
BATCH_SIZE = 64
EPOCHS     = 40
LR         = 3e-4
PATIENCE   = 10
AUG_PER_IMAGE = 30   # 원본 1장 → 30장 증강

# OOD (Out-of-Distribution) 방어 설정
NEGATIVE_PER_TYPE  = 500   # 네거티브 카테고리당 생성 수
OOD_THRESHOLD      = 0.45  # 코사인 유사도 이 미만이면 "약 아님" 판정
NEGATIVE_LABEL_NAME = '__NOT_A_PILL__'
NEG_IMAGE_DIR = DATA_DIR / 'negatives'

# 디바이스 자동 감지
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print('✅ Apple Silicon GPU (MPS) 사용')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('✅ CUDA GPU 사용')
else:
    DEVICE = torch.device('cpu')
    print('⚠️ CPU 사용 (느릴 수 있음)')


# ════════════════════════════════════════════════════════════
#  Phase 1: 데이터 수집 (식약처 API)
# ════════════════════════════════════════════════════════════

def fetch_page(page_no=1, num_of_rows=100):
    params = {
        'serviceKey': API_KEY,
        'pageNo': page_no,
        'numOfRows': num_of_rows,
        'type': 'json',
    }
    try:
        resp = requests.get(MFDS_URL, params=params, timeout=15)
        resp.raise_for_status()
        body = resp.json().get('body', {})
        items = body.get('items', [])
        if isinstance(items, dict):
            items = [items.get('item', {})]
        elif not isinstance(items, list):
            items = []
        return items, int(body.get('totalCount', 0))
    except Exception as e:
        print(f'  오류: {e}')
        return [], 0


def download_image(url, save_path):
    if save_path.exists():
        return True
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        if img.width < 50 or img.height < 50:
            return False
        img.save(str(save_path), 'JPEG', quality=90)
        return True
    except:
        return False


def safe_dirname(name):
    return ''.join(c for c in name if c not in r'\/:*?"<>|').strip()[:50]


def collect_data():
    """식약처 API에서 약품 정보 + 이미지 다운로드"""
    csv_path = DATA_DIR / 'pills.csv'

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f'✅ 기존 데이터 로드: {len(df):,}장')
        return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print('\n📥 Phase 1-1: 약품 목록 수집')
    _, total = fetch_page(1, 1)
    print(f'  전체 약품 수: {total:,}개')

    all_items = []
    page = 1
    pbar = tqdm(total=total, desc='  약품 정보')
    while len(all_items) < total:
        items, _ = fetch_page(page, 100)
        if not items:
            break
        all_items.extend(items)
        pbar.update(len(items))
        page += 1
        time.sleep(0.3)
    pbar.close()
    print(f'  수집 완료: {len(all_items):,}개')

    sample = all_items[0]
    img_field  = 'ITEM_IMAGE' if 'ITEM_IMAGE' in sample else 'itemImage'
    name_field = 'ITEM_NAME'  if 'ITEM_NAME'  in sample else 'itemName'
    code_field = 'ITEM_SEQ'   if 'ITEM_SEQ'   in sample else 'itemSeq'

    with_image = [i for i in all_items if i.get(img_field)]
    print(f'  이미지 있는 항목: {len(with_image):,}개')

    print('\n📥 Phase 1-2: 이미지 다운로드')
    tasks = []
    metadata = []
    for item in with_image:
        name = item.get(name_field, '').strip()
        code = item.get(code_field, '').strip()
        url  = item.get(img_field, '').strip()
        if not name or not url:
            continue

        class_dir = IMAGE_DIR / safe_dirname(name)
        class_dir.mkdir(exist_ok=True)
        save_path = class_dir / f'{code}.jpg'

        tasks.append((url, save_path))
        metadata.append({
            'item_code': code,
            'item_name': name,
            'shape': item.get('DRUG_SHAPE') or '',
            'color_front': item.get('COLOR_CLASS1') or '',
            'color_back': item.get('COLOR_CLASS2') or '',
            'print_front': item.get('PRINT_FRONT') or '',
            'print_back': item.get('PRINT_BACK') or '',
            'image_path': str(save_path),
        })

    print(f'  다운로드 대상: {len(tasks):,}장')

    success_rows = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(download_image, url, path): idx
            for idx, (url, path) in enumerate(tasks)
        }
        pbar = tqdm(total=len(futures), desc='  이미지 다운로드')
        for future in as_completed(futures):
            idx = futures[future]
            if future.result():
                success_rows.append(metadata[idx])
            pbar.update(1)
        pbar.close()

    df = pd.DataFrame(success_rows)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'  ✅ 완료! {len(df):,}장 저장')
    return df


# ════════════════════════════════════════════════════════════
#  OOD 방어: 네거티브 이미지 생성
#  약이 아닌 것들을 프로그래밍으로 생성 → 모델이 거부 학습
# ════════════════════════════════════════════════════════════

def _draw_cartoon_pill(size=224):
    """만화/일러스트 스타일 약 그림 (실제 약이 아님)"""
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 굵은 외곽선 + 단색 채우기 = 그림체
    cx, cy = size // 2, size // 2
    rx = random.randint(40, 90)
    ry = random.randint(25, 70)
    fill_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    outline_color = (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))

    # 외곽선 두껍게 (그림체 특징)
    for thickness in range(4, 0, -1):
        draw.ellipse(
            [cx - rx - thickness, cy - ry - thickness,
             cx + rx + thickness, cy + ry + thickness],
            outline=outline_color, width=3,
        )
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill_color)

    # 가운데 선 (캡슐 느낌)
    if random.random() > 0.4:
        draw.line([(cx, cy - ry), (cx, cy + ry)], fill=outline_color, width=3)
        # 반대쪽 다른 색
        fill2 = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.pieslice([cx - rx, cy - ry, cx + rx, cy + ry], 0, 180, fill=fill2)

    # 텍스트 (그림체에 흔한 라벨)
    if random.random() > 0.5:
        texts = ['PILL', 'RX', '약', 'MED', '100mg', 'DRUG', '💊']
        txt = random.choice(texts)
        try:
            draw.text((cx - 15, cy - 8), txt, fill=outline_color)
        except:
            pass

    return img


def _draw_candy_button(size=224):
    """사탕, 버튼, 동전 등 약과 비슷한 둥근 물체"""
    bg_color = (
        random.randint(180, 250), random.randint(180, 250), random.randint(180, 250)
    )
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    obj_type = random.choice(['candy', 'button', 'coin', 'bead'])
    cx, cy = size // 2 + random.randint(-20, 20), size // 2 + random.randint(-20, 20)
    r = random.randint(30, 80)

    if obj_type == 'candy':
        # 사탕: 매우 채도 높은 색 + 줄무늬/소용돌이
        color = (random.randint(200, 255), random.randint(0, 100), random.randint(0, 100))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        # 소용돌이 줄무늬
        stripe_color = (255, 255, 255)
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x2 = cx + int(r * 0.8 * math.cos(rad))
            y2 = cy + int(r * 0.8 * math.sin(rad))
            draw.line([(cx, cy), (x2, y2)], fill=stripe_color, width=3)

    elif obj_type == 'button':
        # 버튼: 구멍 2~4개
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color,
                     outline=(max(0, color[0]-50), max(0, color[1]-50), max(0, color[2]-50)), width=2)
        # 구멍
        holes = random.choice([2, 4])
        hole_r = r // 8
        offsets = [(-8, -8), (8, 8), (-8, 8), (8, -8)][:holes]
        for ox, oy in offsets:
            draw.ellipse([cx + ox - hole_r, cy + oy - hole_r,
                         cx + ox + hole_r, cy + oy + hole_r],
                        fill=(max(0, color[0]-80), max(0, color[1]-80), max(0, color[2]-80)))

    elif obj_type == 'coin':
        # 동전: 금속색 + 테두리 + 숫자
        gold = random.choice([
            (218, 165, 32), (192, 192, 192), (205, 127, 50), (184, 115, 51),
        ])
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=gold)
        draw.ellipse([cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4],
                    outline=(gold[0]-30, gold[1]-30, gold[2]-30), width=2)
        texts = ['100', '500', '50', '10', '1']
        try:
            draw.text((cx - 10, cy - 6), random.choice(texts),
                     fill=(gold[0]-60, gold[1]-60, gold[2]-60))
        except:
            pass

    else:  # bead
        # 구슬: 하이라이트 + 투명감
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        # 하이라이트
        hl_r = r // 3
        hl_x, hl_y = cx - r // 3, cy - r // 3
        draw.ellipse([hl_x, hl_y, hl_x + hl_r, hl_y + hl_r],
                    fill=(min(255, color[0]+80), min(255, color[1]+80), min(255, color[2]+80)))

    # 살짝 블러 (더 사실적으로)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img


def _draw_random_object(size=224):
    """완전히 랜덤한 물체 (약과 무관)"""
    bg = (random.randint(100, 250), random.randint(100, 250), random.randint(100, 250))
    img = Image.new('RGB', (size, size), bg)
    draw = ImageDraw.Draw(img)

    # 랜덤 도형 여러 개
    for _ in range(random.randint(3, 10)):
        shape = random.choice(['rect', 'line', 'polygon'])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        if shape == 'rect':
            x1, y1 = random.randint(0, size), random.randint(0, size)
            x2, y2 = x1 + random.randint(10, 80), y1 + random.randint(10, 80)
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape == 'line':
            pts = [(random.randint(0, size), random.randint(0, size)) for _ in range(random.randint(2, 5))]
            draw.line(pts, fill=color, width=random.randint(1, 5))
        else:
            pts = [(random.randint(0, size), random.randint(0, size)) for _ in range(random.randint(3, 6))]
            draw.polygon(pts, fill=color)

    return img


def _draw_food_like(size=224):
    """음식처럼 보이는 이미지 (약과 혼동 가능한 것들)"""
    bg = (random.randint(180, 240), random.randint(170, 230), random.randint(160, 220))
    img = Image.new('RGB', (size, size), bg)
    draw = ImageDraw.Draw(img)

    food_type = random.choice(['gummy', 'chocolate', 'mint', 'tablet_candy'])
    cx, cy = size // 2 + random.randint(-15, 15), size // 2 + random.randint(-15, 15)

    if food_type == 'gummy':
        # 젤리곰 모양
        color = random.choice([
            (255, 50, 50), (50, 200, 50), (255, 200, 0), (255, 130, 0), (200, 50, 200),
        ])
        # 몸통
        draw.ellipse([cx - 25, cy - 10, cx + 25, cy + 40], fill=color)
        # 머리
        draw.ellipse([cx - 18, cy - 35, cx + 18, cy], fill=color)
        # 귀
        draw.ellipse([cx - 25, cy - 42, cx - 10, cy - 28], fill=color)
        draw.ellipse([cx + 10, cy - 42, cx + 25, cy - 28], fill=color)

    elif food_type == 'chocolate':
        # 초콜릿 조각
        brown = (random.randint(60, 100), random.randint(30, 60), random.randint(10, 40))
        r = random.randint(25, 50)
        if random.random() > 0.5:
            draw.rounded_rectangle([cx - r, cy - r, cx + r, cy + r],
                                   radius=8, fill=brown)
        else:
            draw.ellipse([cx - r, cy - r * 0.7, cx + r, cy + r * 0.7], fill=brown)

    elif food_type == 'mint':
        # 민트캔디 (흰색 + 줄무늬 → 약과 혼동)
        r = random.randint(30, 55)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
        stripe_color = random.choice([(255, 0, 0), (0, 150, 0), (0, 0, 200)])
        for angle in range(0, 360, 20):
            rad1 = math.radians(angle)
            rad2 = math.radians(angle + 10)
            pts = [
                (cx, cy),
                (cx + int(r * math.cos(rad1)), cy + int(r * math.sin(rad1))),
                (cx + int(r * math.cos(rad2)), cy + int(r * math.sin(rad2))),
            ]
            draw.polygon(pts, fill=stripe_color)

    else:  # tablet_candy
        # 정제형 캔디 (약과 가장 비슷)
        color = (random.randint(200, 255), random.randint(200, 255), random.randint(0, 100))
        rx, ry = random.randint(35, 60), random.randint(25, 45)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
        # 약과 다르게: 반짝이 효과
        for _ in range(5):
            sx = cx + random.randint(-rx, rx)
            sy = cy + random.randint(-ry, ry)
            draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3],
                        fill=(255, 255, min(255, color[2] + 100)))

    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    return img


def _generate_screenshot_text(size=224):
    """스크린샷/텍스트 이미지 (약 사진이 아닌 것)"""
    bg = random.choice([(255, 255, 255), (240, 240, 240), (30, 30, 30), (245, 245, 220)])
    img = Image.new('RGB', (size, size), bg)
    draw = ImageDraw.Draw(img)

    text_color = (0, 0, 0) if bg[0] > 128 else (220, 220, 220)
    # 랜덤 텍스트 줄
    y = 10
    for _ in range(random.randint(5, 15)):
        line_len = random.randint(5, 30)
        # 랜덤 블록으로 텍스트 흉내
        x = random.randint(5, 20)
        line_h = random.randint(6, 12)
        draw.rectangle([x, y, x + line_len * 6, y + line_h], fill=text_color)
        y += line_h + random.randint(3, 8)
        if y > size - 20:
            break

    return img


def _generate_pure_noise(size=224):
    """순수 노이즈/그라데이션 (아무 물체도 아닌 것)"""
    mode = random.choice(['noise', 'gradient', 'solid', 'pattern'])

    if mode == 'noise':
        arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    elif mode == 'gradient':
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        for c in range(3):
            start = random.randint(0, 255)
            end = random.randint(0, 255)
            arr[:, :, c] = np.linspace(start, end, size).reshape(-1, 1).astype(np.uint8)
        img = Image.fromarray(arr)
    elif mode == 'solid':
        img = Image.new('RGB', (size, size),
                        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    else:
        # 체크무늬
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        block = random.randint(8, 32)
        c1 = np.array([random.randint(0, 255)] * 3)
        c2 = np.array([random.randint(0, 255)] * 3)
        for y in range(0, size, block):
            for x in range(0, size, block):
                c = c1 if ((y // block) + (x // block)) % 2 == 0 else c2
                arr[y:y+block, x:x+block] = c
        img = Image.fromarray(arr)

    return img


def collect_negative_data():
    """네거티브 이미지 자동 생성 (6가지 카테고리)"""
    NEG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    categories = {
        'cartoon_pill':    _draw_cartoon_pill,      # 그림체 약
        'candy_button':    _draw_candy_button,       # 사탕/버튼/동전
        'random_object':   _draw_random_object,      # 랜덤 물체
        'food_like':       _draw_food_like,           # 음식 (젤리, 초콜릿)
        'screenshot_text': _generate_screenshot_text, # 스크린샷/텍스트
        'pure_noise':      _generate_pure_noise,      # 노이즈/패턴
    }

    total_generated = 0
    for cat_name, gen_func in categories.items():
        cat_dir = NEG_IMAGE_DIR / cat_name
        cat_dir.mkdir(exist_ok=True)

        existing = list(cat_dir.glob('*.jpg'))
        if len(existing) >= NEGATIVE_PER_TYPE:
            print(f'  ✅ {cat_name}: 이미 {len(existing)}장 존재')
            total_generated += len(existing)
            continue

        need = NEGATIVE_PER_TYPE - len(existing)
        print(f'  🎨 {cat_name}: {need}장 생성 중...')
        for i in tqdm(range(need), desc=f'    {cat_name}', leave=False):
            try:
                img = gen_func(224)
                # 추가 변형 (다양성)
                if random.random() < 0.5:
                    img = camera_simulation(img)
                save_path = cat_dir / f'{cat_name}_{len(existing) + i:04d}.jpg'
                img.save(str(save_path), 'JPEG', quality=85)
            except Exception as e:
                pass
        total_generated += NEGATIVE_PER_TYPE

    print(f'  ✅ 네거티브 이미지 총 {total_generated:,}장 준비 완료')
    return total_generated


# ════════════════════════════════════════════════════════════
#  방법 1: 고급 Augmentation — 실제 촬영환경 시뮬레이션
# ════════════════════════════════════════════════════════════

def extract_pill_mask(img):
    """흰 배경에서 약 영역 마스크 추출 (간단한 임계값 방식)"""
    gray = img.convert('L')
    arr = np.array(gray)
    # 흰 배경(>220)이 아닌 부분 = 약
    mask = (arr < 220).astype(np.uint8) * 255
    # 모폴로지 연산으로 마스크 정리
    from PIL import ImageFilter
    mask_img = Image.fromarray(mask, 'L')
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    return mask_img


def generate_random_background(size=(224, 224)):
    """랜덤 배경 생성 (테이블, 천, 종이 등 시뮬레이션)"""
    w, h = size
    bg_type = random.choice(['solid', 'gradient', 'noise', 'texture', 'wood'])

    if bg_type == 'solid':
        # 단색 배경 (다양한 테이블 색상)
        colors = [
            (200+random.randint(-30,30), 180+random.randint(-30,30), 160+random.randint(-30,30)),  # 나무
            (240+random.randint(-15,15), 240+random.randint(-15,15), 240+random.randint(-15,15)),  # 흰색
            (180+random.randint(-20,20), 180+random.randint(-20,20), 190+random.randint(-20,20)),  # 회색
            (200+random.randint(-20,20), 200+random.randint(-20,20), 180+random.randint(-20,20)),  # 베이지
            (60+random.randint(-20,20),  60+random.randint(-20,20),  60+random.randint(-20,20)),   # 어두운
        ]
        color = random.choice(colors)
        bg = Image.new('RGB', size, color)

    elif bg_type == 'gradient':
        # 그라데이션 (조명 효과)
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        c1 = np.array([random.randint(150,240)]*3)
        c2 = np.array([random.randint(100,200)]*3)
        for y in range(h):
            t = y / h
            arr[y] = (c1 * (1-t) + c2 * t).astype(np.uint8)
        # 랜덤 회전
        bg = Image.fromarray(arr)
        bg = bg.rotate(random.randint(0, 360), fillcolor=(int(c1[0]),int(c1[1]),int(c1[2])))

    elif bg_type == 'noise':
        # 노이즈 텍스처 (천, 종이)
        base_color = random.randint(140, 230)
        arr = np.random.normal(base_color, 15, (h, w, 3)).clip(0, 255).astype(np.uint8)
        bg = Image.fromarray(arr)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=1))

    elif bg_type == 'texture':
        # 패턴 텍스처
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        base = random.randint(160, 220)
        arr[:] = base
        # 줄무늬 또는 격자
        stripe = random.randint(10, 30)
        for i in range(0, max(w, h), stripe):
            if random.random() > 0.5:
                arr[max(0,i):min(h,i+2), :] = base - 20
            else:
                arr[:, max(0,i):min(w,i+2)] = base - 20
        bg = Image.fromarray(arr)
        bg = bg.filter(ImageFilter.GaussianBlur(radius=2))

    else:  # wood
        # 나무결 시뮬레이션
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        base_r, base_g, base_b = 180+random.randint(-20,20), 150+random.randint(-20,20), 120+random.randint(-20,20)
        for y in range(h):
            wave = math.sin(y * 0.05 + random.random() * 10) * 15
            arr[y, :, 0] = np.clip(base_r + wave + np.random.normal(0, 5, w), 0, 255)
            arr[y, :, 1] = np.clip(base_g + wave + np.random.normal(0, 5, w), 0, 255)
            arr[y, :, 2] = np.clip(base_b + wave + np.random.normal(0, 5, w), 0, 255)
        bg = Image.fromarray(arr.astype(np.uint8))

    return bg


def add_shadow(img):
    """랜덤 그림자 추가"""
    shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    # 랜덤 위치에 반투명 원/타원
    x = random.randint(-50, img.width)
    y = random.randint(-50, img.height)
    r = random.randint(50, 200)
    opacity = random.randint(20, 80)
    draw.ellipse([x, y, x+r, y+r], fill=(0, 0, 0, opacity))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=random.randint(20, 50)))
    img_rgba = img.convert('RGBA')
    result = Image.alpha_composite(img_rgba, shadow)
    return result.convert('RGB')


def camera_simulation(img):
    """핸드폰 카메라 시뮬레이션 (블러, 노이즈, JPEG 압축)"""
    # 모션 블러 (손떨림)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

    # 가우시안 노이즈 (저조도)
    if random.random() < 0.4:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 15), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # JPEG 압축 아티팩트
    if random.random() < 0.3:
        buf = BytesIO()
        img.save(buf, 'JPEG', quality=random.randint(30, 70))
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

    # 밝기/대비 변화 (조명)
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.6, 1.4))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    return img


def composite_pill_on_background(pill_img, bg_size=(256, 256)):
    """약 이미지를 배경 위에 합성"""
    bg = generate_random_background(bg_size)

    # 약 마스크 추출
    mask = extract_pill_mask(pill_img)

    # 랜덤 크기/위치/회전
    scale = random.uniform(0.4, 0.8)
    new_size = (int(bg_size[0] * scale), int(bg_size[1] * scale))
    pill_resized = pill_img.resize(new_size, Image.BILINEAR)
    mask_resized = mask.resize(new_size, Image.BILINEAR)

    # 회전
    angle = random.randint(0, 360)
    pill_rotated = pill_resized.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    mask_rotated = mask_resized.rotate(angle, expand=True, fillcolor=0)

    # 배경 위에 합성
    paste_x = random.randint(0, max(0, bg_size[0] - pill_rotated.width))
    paste_y = random.randint(0, max(0, bg_size[1] - pill_rotated.height))
    bg.paste(pill_rotated, (paste_x, paste_y), mask_rotated)

    return bg


# ════════════════════════════════════════════════════════════
#  방법 2: CutMix + 멀티뷰 합성
# ════════════════════════════════════════════════════════════

def cutmix_data(images, labels, alpha=1.0):
    """CutMix: 두 이미지를 랜덤 영역으로 섞어서 새 학습 데이터 생성"""
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(batch_size)

    _, _, h, w = images.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return mixed_images, labels, labels[rand_index], lam


# ════════════════════════════════════════════════════════════
#  모델 정의 (ArcFace Metric Learning)
# ════════════════════════════════════════════════════════════

class ArcFaceHead(nn.Module):
    def __init__(self, emb_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = torch.cos(theta + one_hot * self.m)
        return output * self.s


class PillModel(nn.Module):
    def __init__(self, num_classes, emb_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=True, num_classes=0,
        )
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone.num_features, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )
        self.arcface = ArcFaceHead(emb_dim, num_classes)

    def get_embedding(self, x):
        return F.normalize(self.embedding(self.backbone(x)), dim=1)

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        emb = self.embedding(feat)
        if labels is not None:
            return self.arcface(emb, labels), F.normalize(emb, dim=1)
        return F.normalize(emb, dim=1)


# ════════════════════════════════════════════════════════════
#  Dataset (고급 augmentation 통합)
# ════════════════════════════════════════════════════════════

class AugmentedPillDataset(Dataset):
    """
    방법 1 + 2 통합 Dataset
    원본 1장 → aug_per_image장 생성
    - 50% 확률: 배경합성 + 카메라 시뮬레이션 (방법 1)
    - 50% 확률: 일반 augmentation (기본)
    """
    def __init__(self, dataframe, transform=None, aug_per_image=30):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.aug_per_image = aug_per_image

        # 배경합성용 기본 transform (ToTensor 전까지만)
        self.to_tensor_norm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df) * self.aug_per_image

    def __getitem__(self, idx):
        real_idx = idx // self.aug_per_image
        aug_idx  = idx % self.aug_per_image
        row = self.df.iloc[real_idx]
        label = int(row['label'])

        try:
            img = Image.open(row['image_path']).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), (128, 128, 128))

        if aug_idx == 0:
            # 원본 (augmentation 없이)
            img = self.to_tensor_norm(img)
        elif aug_idx < self.aug_per_image // 2:
            # 방법 1: 배경합성 + 카메라 시뮬레이션
            try:
                img = composite_pill_on_background(img, bg_size=(256, 256))
                img = camera_simulation(img)
                if random.random() < 0.3:
                    img = add_shadow(img)
            except:
                pass  # 합성 실패 시 원본 사용
            img = self.to_tensor_norm(img)
        else:
            # 기본 augmentation
            if self.transform:
                img = self.transform(img)
            else:
                img = self.to_tensor_norm(img)

        return img, label


class PillDataset(Dataset):
    """검증/레퍼런스용 (augmentation 없음)"""
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, int(row['label'])


# Transform 정의
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ════════════════════════════════════════════════════════════
#  학습 함수
# ════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, use_cutmix=True):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc='  학습', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # 방법 2: CutMix (50% 확률)
        if use_cutmix and random.random() < 0.5:
            imgs_mixed, labels_a, labels_b, lam = cutmix_data(imgs, labels)
            logits, _ = model(imgs_mixed, labels_a)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        else:
            logits, _ = model(imgs, labels)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # gradient clipping (안정성)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=f'{loss.item():.3f}')

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_recall(model, val_loader, ref_loader, top_k=5):
    model.eval()

    ref_embs, ref_labels = [], []
    for imgs, labels in ref_loader:
        emb = model.get_embedding(imgs.to(DEVICE))
        ref_embs.append(emb.cpu())
        ref_labels.extend(labels.tolist())
    ref_embs = torch.cat(ref_embs)
    ref_labels = torch.tensor(ref_labels)

    top1, top5, total = 0, 0, 0
    for imgs, labels in val_loader:
        emb = model.get_embedding(imgs.to(DEVICE)).cpu()
        sim = torch.mm(emb, ref_embs.T)
        topk = sim.topk(top_k, dim=1).indices

        for i in range(len(labels)):
            gt = labels[i].item()
            preds = ref_labels[topk[i]].tolist()
            if preds[0] == gt:
                top1 += 1
            if gt in preds:
                top5 += 1
            total += 1

    return top1 / total, top5 / total


# ════════════════════════════════════════════════════════════
#  메인 실행
# ════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('  이거돼? — 약 이미지 인식 모델 학습')
    print('  ArcFace + 고급증강 + CutMix + OOD 방어 | M4 Pro')
    print('=' * 60)

    # ── 데이터 수집 ──
    df = collect_data()

    # ── 네거티브 이미지 생성 ──
    print('\n🎨 Phase 1-3: 네거티브 이미지 생성 (약이 아닌 것)')
    collect_negative_data()

    # 네거티브 이미지를 DataFrame에 추가
    neg_rows = []
    for cat_dir in NEG_IMAGE_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        for img_path in cat_dir.glob('*.jpg'):
            neg_rows.append({
                'item_code': f'NEG_{img_path.stem}',
                'item_name': NEGATIVE_LABEL_NAME,
                'image_path': str(img_path),
            })
    neg_df = pd.DataFrame(neg_rows)
    print(f'  네거티브 이미지: {len(neg_df):,}장')

    # ── 레이블 (네거티브 포함) ──
    all_pill_names = sorted(df['item_name'].unique())
    # 네거티브는 마지막 레이블
    unique_names = all_pill_names + [NEGATIVE_LABEL_NAME]
    label_map = {name: idx for idx, name in enumerate(unique_names)}
    reverse_map = {idx: name for name, idx in label_map.items()}
    NEGATIVE_LABEL = label_map[NEGATIVE_LABEL_NAME]

    df['label'] = df['item_name'].map(label_map)
    neg_df['label'] = NEGATIVE_LABEL
    NUM_CLASSES = len(label_map)

    # 합치기
    full_df = pd.concat([df, neg_df], ignore_index=True)
    full_df = full_df[full_df['image_path'].apply(os.path.exists)].reset_index(drop=True)

    n_pills = len(full_df[full_df['label'] != NEGATIVE_LABEL])
    n_neg = len(full_df[full_df['label'] == NEGATIVE_LABEL])
    print(f'\n📊 데이터:')
    print(f'  약 이미지: {n_pills:,}장 ({len(all_pill_names):,}종)')
    print(f'  네거티브: {n_neg:,}장 (NOT_A_PILL)')
    print(f'  전체 클래스: {NUM_CLASSES:,}개 (약 {NUM_CLASSES-1} + 네거티브 1)')
    print(f'  증강 후 예상: ~{n_pills * AUG_PER_IMAGE + n_neg * AUG_PER_IMAGE:,}장')

    # ── 분리 (약 / 네거티브 각각 분리 후 합침) ──
    pill_df = full_df[full_df['label'] != NEGATIVE_LABEL].reset_index(drop=True)
    neg_only_df = full_df[full_df['label'] == NEGATIVE_LABEL].reset_index(drop=True)

    pill_train, pill_val = train_test_split(pill_df, test_size=0.1, random_state=42)
    neg_train, neg_val = train_test_split(neg_only_df, test_size=0.1, random_state=42)

    train_df = pd.concat([pill_train, neg_train], ignore_index=True)
    val_df = pd.concat([pill_val, neg_val], ignore_index=True)

    # 증강 Dataset (방법 1+2)
    train_dataset = AugmentedPillDataset(train_df, train_transform, aug_per_image=AUG_PER_IMAGE)
    val_dataset   = PillDataset(val_df, val_transform)
    # 레퍼런스 DB에는 네거티브도 포함 (추론 시 NOT_A_PILL 클러스터 활용)
    ref_df = train_df.groupby('label').first().reset_index()
    ref_dataset = PillDataset(ref_df, val_transform)

    nw = 0 if DEVICE.type == 'mps' else 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=False)
    ref_loader   = DataLoader(ref_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=False)

    print(f'  Train: {len(train_dataset):,}장 ({len(train_loader):,} 배치)')
    print(f'  Val: {len(val_dataset):,}장')

    # ── 모델 ──
    model = PillModel(NUM_CLASSES, EMB_DIM).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f'\n🧠 모델: EfficientNet-B0 + ArcFace')
    print(f'  임베딩: {EMB_DIM}차원 | 파라미터: {params/1e6:.1f}M')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )

    # ── 학습 ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUTPUT_DIR / 'best_model.pth'
    best_recall = 0
    no_improve = 0

    print(f'\n🚀 학습 시작! (총 {EPOCHS} epoch)')
    print(f'  증강: 배경합성 + 카메라시뮬 + CutMix')
    print('-' * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, use_cutmix=True
        )
        elapsed = time.time() - t0

        # 매 에폭 Recall 측정 (처음 5에폭), 이후 3에폭마다
        if epoch <= 5 or epoch % 3 == 0 or epoch == EPOCHS:
            r1, r5 = evaluate_recall(model, val_loader, ref_loader)
            print(f'Epoch {epoch:02d}/{EPOCHS} | '
                  f'Loss {train_loss:.4f} | Acc {train_acc:.3f} | '
                  f'R@1 {r1:.4f} R@5 {r5:.4f} | '
                  f'{elapsed:.0f}s')

            if r5 > best_recall:
                best_recall = r5
                no_improve = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'label_map': label_map,
                    'reverse_map': reverse_map,
                    'emb_dim': EMB_DIM,
                    'num_classes': NUM_CLASSES,
                    'recall_1': r1,
                    'recall_5': r5,
                    'epoch': epoch,
                }, str(best_path))
                print(f'  💾 Best 저장! R@1={r1:.4f} R@5={r5:.4f}')
            else:
                no_improve += 1
                print(f'  ⚠️ 개선 없음 ({no_improve}/{PATIENCE})')
        else:
            print(f'Epoch {epoch:02d}/{EPOCHS} | '
                  f'Loss {train_loss:.4f} | Acc {train_acc:.3f} | '
                  f'{elapsed:.0f}s')

        if no_improve >= PATIENCE:
            print('  ⏹️ Early Stopping!')
            break

    print('-' * 60)
    print(f'✅ 학습 완료! Best R@1={best_recall:.4f}')

    # ── 레퍼런스 DB ──
    print('\n📦 레퍼런스 DB 구축 중...')
    checkpoint = torch.load(str(best_path), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_ref_df = df.groupby('label').first().reset_index()
    all_ref_dataset = PillDataset(all_ref_df, val_transform)
    all_ref_loader = DataLoader(all_ref_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)

    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(all_ref_loader, desc='  임베딩 추출'):
            emb = model.get_embedding(imgs.to(DEVICE))
            all_embs.append(emb.cpu().numpy())
            all_labels.extend(labels.tolist())

    all_embs = np.concatenate(all_embs, axis=0)
    all_names = [reverse_map[l] for l in all_labels]

    np.save(str(OUTPUT_DIR / 'ref_embeddings.npy'), all_embs)
    with open(str(OUTPUT_DIR / 'ref_names.json'), 'w', encoding='utf-8') as f:
        json.dump(all_names, f, ensure_ascii=False)

    # OOD 설정 저장
    ood_config = {
        'threshold': OOD_THRESHOLD,
        'negative_label': NEGATIVE_LABEL_NAME,
        'negative_label_idx': NEGATIVE_LABEL,
        'total_classes': NUM_CLASSES,
        'pill_classes': NUM_CLASSES - 1,
        'emb_dim': EMB_DIM,
    }
    with open(str(OUTPUT_DIR / 'ood_config.json'), 'w', encoding='utf-8') as f:
        json.dump(ood_config, f, ensure_ascii=False, indent=2)

    print(f'  ✅ {all_embs.shape[0]:,}개 항목 × {all_embs.shape[1]}차원')
    print(f'  📋 OOD 설정 저장 (threshold={OOD_THRESHOLD})')

    # ── 테스트 (OOD 임계값 적용) ──
    print('\n🧪 추론 테스트 (OOD 방어 포함):')
    print(f'  OOD 임계값: {OOD_THRESHOLD} (이 미만 = "약 아님" 판정)')
    print('-' * 60)

    # 약 이미지 테스트
    pill_test = val_df[val_df['label'] != NEGATIVE_LABEL].sample(
        min(12, len(val_df[val_df['label'] != NEGATIVE_LABEL])), random_state=42
    )
    # 네거티브 이미지 테스트
    neg_test = val_df[val_df['label'] == NEGATIVE_LABEL].sample(
        min(5, len(val_df[val_df['label'] == NEGATIVE_LABEL])), random_state=42
    )
    test_samples = pd.concat([pill_test, neg_test])

    correct_1, correct_5, ood_correct, ood_total = 0, 0, 0, 0
    pill_total = 0

    # 레퍼런스에서 네거티브 제외한 인덱스 (약만)
    pill_ref_mask = np.array([n != NEGATIVE_LABEL_NAME for n in all_names])

    for _, row in test_samples.iterrows():
        try:
            img = Image.open(row['image_path']).convert('RGB')
        except:
            continue
        img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            query = model.get_embedding(img_tensor).cpu().numpy()

        # 전체 유사도 (네거티브 포함)
        sims_all = np.dot(all_embs, query.T).flatten()
        # 약만 유사도
        sims_pill = sims_all.copy()
        sims_pill[~pill_ref_mask] = -1  # 네거티브 제외

        top1_sim = sims_pill.max()
        top5_idxs = np.argsort(sims_pill)[::-1][:5]
        top5_names = [all_names[i] for i in top5_idxs]
        true_name = row['item_name']
        is_negative = (true_name == NEGATIVE_LABEL_NAME)

        if is_negative:
            ood_total += 1
            # 네거티브인데 약 유사도가 임계값 미만이면 정확히 거부한 것
            if top1_sim < OOD_THRESHOLD:
                ood_correct += 1
                print(f'🛡️ 네거티브 정확히 거부! (유사도 {top1_sim:.3f} < {OOD_THRESHOLD})')
            else:
                print(f'⚠️ 네거티브를 약으로 오인! (유사도 {top1_sim:.3f} → {top5_names[0][:25]})')
        else:
            pill_total += 1
            if top1_sim < OOD_THRESHOLD:
                print(f'🚨 실제 약인데 거부됨! 정답: {true_name[:25]} (유사도 {top1_sim:.3f})')
            else:
                if top5_names[0] == true_name:
                    correct_1 += 1
                if true_name in top5_names:
                    correct_5 += 1

                ok = '✅' if top5_names[0] == true_name else ('🔶' if true_name in top5_names else '❌')
                print(f'{ok} 정답: {true_name[:30]}')
                print(f'   Top1: {top5_names[0][:30]} ({top1_sim*100:.1f}%)')

    if pill_total > 0:
        print(f'\n📊 약 인식: Top1={correct_1}/{pill_total} ({correct_1/pill_total*100:.1f}%) | '
              f'Top5={correct_5}/{pill_total} ({correct_5/pill_total*100:.1f}%)')
    if ood_total > 0:
        print(f'🛡️ OOD 방어: {ood_correct}/{ood_total} 정확 거부 ({ood_correct/ood_total*100:.1f}%)')

    # ── 요약 ──
    print('\n' + '=' * 60)
    print('📁 생성된 파일:')
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size / (1024*1024)
        print(f'  {f.name} ({size:.1f}MB)')

    print(f'\n🎯 최종 성능: Recall@1={checkpoint["recall_1"]:.4f} | Recall@5={checkpoint["recall_5"]:.4f}')
    print(f'🛡️ OOD 임계값: {OOD_THRESHOLD} (이 미만 = "약이 아닙니다" 판정)')
    print('=' * 60)
    print('🎉 완료! output/ 폴더의 파일 4개를 앱에 연동하면 됩니다.')
    print('   - best_model.pth     : 학습된 모델')
    print('   - ref_embeddings.npy : 레퍼런스 임베딩 DB')
    print('   - ref_names.json     : 약품명 매핑')
    print('   - ood_config.json    : OOD 방어 설정')


if __name__ == '__main__':
    main()
