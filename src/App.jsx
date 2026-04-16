/**
 * 이거돼? — AI 기반 약물 판독 & 복약 가이드
 * Stack: React + Vite + Tailwind CSS + Firebase + Groq + 식약처 API
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Camera, ImagePlus, Send, ChevronRight, Clock, AlertTriangle,
  CheckCircle, XCircle, Pill, MessageCircle, History,
  Loader2, Sparkles, RefreshCw, ChevronLeft,
  Shield, Zap, X, Database
} from 'lucide-react'

import { initializeApp, getApps } from 'firebase/app'
import { getAuth, signInAnonymously, onAuthStateChanged } from 'firebase/auth'
import { getFirestore, collection, addDoc, query, orderBy, limit, onSnapshot, serverTimestamp } from 'firebase/firestore'

// ─── 환경변수 ─────────────────────────────────────────────────────────────────
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
}

const APP_ID = import.meta.env.VITE_APP_ID || 'igeordwae-dev'
const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_KEY
const MFDS_API_KEY = import.meta.env.VITE_MFDS_API_KEY
const GROQ_MODEL = 'llama-3.3-70b-versatile'
const GROQ_VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'
const GROQ_BASE = 'https://api.groq.com/openai/v1'

// ─── 식약처 API 엔드포인트 (Vercel 프록시 경유) ───────────────────────────────
const MFDS_PROXY = '/api/mfds-proxy'
const MFDS_DRUG_INFO_URL = `${MFDS_PROXY}?endpoint=drugInfo`
const MFDS_PILL_INFO_URL = `${MFDS_PROXY}?endpoint=pillInfo`
const MFDS_PRMISN_URL   = `${MFDS_PROXY}?endpoint=permission`

// ─── DUR API 엔드포인트 ───────────────────────────────────────────────────────
const DUR_BASE = 'https://apis.data.go.kr/1471000/DURPrdlstInfoService03'
const DUR_ENDPOINTS = {
  병용금기:   `${DUR_BASE}/getUsjntTabooInfoList03`,
  임부금기:   `${DUR_BASE}/getPwnmTabooInfoList03`,
  노인주의:   `${DUR_BASE}/getOdsnAtentInfoList03`,
  효능군중복: `${DUR_BASE}/getEfcyDplctInfoList03`,
}

// ─── Firebase 초기화 ──────────────────────────────────────────────────────────
let app, auth, db
try {
  app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig)
  auth = getAuth(app)
  db = getFirestore(app)
} catch (e) {
  console.warn('Firebase 초기화 실패:', e.message)
}

const LOGS_PATH = () => collection(db, `artifacts/${APP_ID}/public/data/analysis_logs`)

// ─── Groq API 호출 (지수 백오프) ─────────────────────────────────────────────
async function safeFetchGroq(body, retries = 3, delay = 1000) {
  if (!GROQ_API_KEY) throw new Error('VITE_GROQ_API_KEY 환경변수가 설정되지 않았습니다.')
  for (let i = 0; i < retries; i++) {
    try {
      const res = await fetch(`${GROQ_BASE}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${GROQ_API_KEY}` },
        body: JSON.stringify(body),
      })
      if (res.status === 401) throw new Error('API 키가 유효하지 않습니다.')
      if (res.status === 429 || res.status >= 500) {
        if (i < retries - 1) { await new Promise(r => setTimeout(r, delay * Math.pow(2, i))); continue }
      }
      if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error?.message || `HTTP ${res.status}`) }
      return await res.json()
    } catch (e) {
      if (i === retries - 1) throw e
      await new Promise(r => setTimeout(r, delay * Math.pow(2, i)))
    }
  }
}

// ─── 식약처 텍스트 AI 요약 ───────────────────────────────────────────────────
async function summarizeMfdsText(label, text) {
  if (!GROQ_API_KEY || !text || text.length < 50) return text
  try {
    const data = await safeFetchGroq({
      model: GROQ_MODEL,
      messages: [{
        role: 'user',
        content: `다음 의약품 "${label}" 내용을 환자가 이해하기 쉽게 2문장 이내로 요약해주세요. 핵심만 간결하게:\n\n${text}`
      }],
      temperature: 0.3,
      max_tokens: 150,
    })
    return data.choices?.[0]?.message?.content?.trim() || text.slice(0, 100)
  } catch {
    return text.slice(0, 100)
  }
}

// ─── 식약처 API: 의약품 개요정보 조회 ────────────────────────────────────────
async function fetchMfdsInfo(drugName) {
  if (!drugName) return null
  try {
    const params = new URLSearchParams({ itemName: drugName, numOfRows: '3', pageNo: '1' })
    const res = await fetch(`${MFDS_DRUG_INFO_URL}&${params}`)
    const data = await res.json()
    const items = data?.body?.items
    if (!items || items.length === 0) return null
    const item = items[0]
    return {
      itemName: item.itemName,
      entpName: item.entpName,
      efcyQesitm: item.efcyQesitm,
      useMethodQesitm: item.useMethodQesitm,
      atpnWarnQesitm: item.atpnWarnQesitm,
      atpnQesitm: item.atpnQesitm,
      intrcQesitm: item.intrcQesitm,
      seQesitm: item.seQesitm,
      depositMethodQesitm: item.depositMethodQesitm,
      source: '식품의약품안전처',
    }
  } catch (e) {
    console.warn('식약처 API 오류:', e.message)
    return null
  }
}

// ─── 식약처 API: 낱알식별 - 이름으로 검색 ───────────────────────────────────
async function fetchPillByName(drugName) {
  if (!drugName) return null
  try {
    const params = new URLSearchParams({ itemName: drugName, numOfRows: '5', pageNo: '1' })
    const res = await fetch(`${MFDS_PILL_INFO_URL}&${params}`)
    const data = await res.json()
    const items = data?.body?.items
    if (!items || items.length === 0) return null
    return items[0]
  } catch (e) {
    console.warn('낱알식별(이름) API 오류:', e.message)
    return null
  }
}

// ─── 식약처 API: 낱알식별 - 색상/모양/각인으로 검색 ────────────────────────
async function fetchPillByFeature({ color, shape, imprint, form }) {
  try {
    const params = new URLSearchParams({ numOfRows: '3', pageNo: '1' })
    if (color) params.append('colorClass1', color)
    if (shape) params.append('chart', shape)
    if (imprint) params.append('markKorEng', imprint)
    if (form) params.append('formCodeName', form)
    const res = await fetch(`${MFDS_PILL_INFO_URL}&${params}`)
    const data = await res.json()
    const items = data?.body?.items
    if (!items || items.length === 0) return null
    return items[0]
  } catch (e) {
    console.warn('낱알식별 API 오류:', e.message)
    return null
  }
}

// ─── 식약처 API: 의약품 제품허가정보 상세 ────────────────────────────────────
async function fetchDrugPermission(drugName) {
  if (!drugName) return null
  try {
    const params = new URLSearchParams({ item_name: drugName, numOfRows: '3', pageNo: '1' })
    const res = await fetch(`${MFDS_PRMISN_URL}&${params}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    const raw = data?.body?.items
    if (!raw) return null
    const items = Array.isArray(raw) ? raw : Array.isArray(raw.item) ? raw.item : [raw.item]
    if (!items || items.length === 0) return null
    const it = items[0]
    return {
      itemName:       it.ITEM_NAME        || it.itemName        || null,
      entpName:       it.ENTP_NAME        || it.entpName        || null,
      itemPermitDate: it.ITEM_PERMIT_DATE || it.itemPermitDate  || null,
      ingrName:       it.INGR_NAME        || it.ingrName        || null,
      etcOtcName:     it.ETC_OTC_NAME     || it.etcOtcName      || null,
      storageMethod:  it.STORAGE_METHOD   || it.storageMethod   || null,
      validTerm:      it.VALID_TERM       || it.validTerm       || null,
      packUnit:       it.PACK_UNIT        || it.packUnit        || null,
      cancelName:     it.CANCEL_NAME      || it.cancelName      || null,
      source: '식약처_제품허가',
    }
  } catch (e) {
    console.warn('제품허가 API 오류:', e.message)
    return null
  }
}

// ─── DUR API 헬퍼 ────────────────────────────────────────────────────────────
async function fetchDurApi(endpoint, drugName) {
  if (!MFDS_API_KEY || !drugName) return []
  try {
    const params = new URLSearchParams({
      serviceKey: MFDS_API_KEY,
      itemName: drugName,
      type: 'json',
      numOfRows: '5',
      pageNo: '1',
    })
    const res = await fetch(`${endpoint}?${params}`)
    if (!res.ok) return []
    const data = await res.json()
    const raw = data?.body?.items
    if (!raw) return []
    return Array.isArray(raw) ? raw : Array.isArray(raw.item) ? raw.item : raw.item ? [raw.item] : []
  } catch (e) {
    console.warn('DUR API 오류:', e.message)
    return []
  }
}

async function checkDurCombination(drugNames) {
  if (!drugNames || drugNames.length < 2) return []
  const warnings = []
  for (let i = 0; i < drugNames.length; i++) {
    for (let j = i + 1; j < drugNames.length; j++) {
      const drugA = drugNames[i], drugB = drugNames[j]
      const items = await fetchDurApi(DUR_ENDPOINTS.병용금기, drugA)
      const matched = items.filter(item => {
        const prohibitName = item.PROHBT_CONTENT || item.prohibtContent || item.MIXTURE_ITEM_NAME || ''
        return prohibitName.includes(drugB) || drugB.includes(prohibitName.slice(0, 4))
      })
      if (matched.length > 0) {
        warnings.push({
          type: '병용금기', level: 'danger', drugs: [drugA, drugB],
          reason: matched[0].PROHBT_CONTENT || matched[0].prohibtContent || `${drugA}와 ${drugB}는 함께 복용하면 안 돼요.`,
          note: matched[0].REMARK || matched[0].remark || '',
        })
      }
    }
  }
  return warnings
}

async function checkDurPregnancy(drugNames) {
  const warnings = []
  for (const name of drugNames) {
    const items = await fetchDurApi(DUR_ENDPOINTS.임부금기, name)
    if (items.length > 0) {
      const item = items[0]
      warnings.push({
        type: '임부금기', level: 'danger', drugs: [name],
        reason: item.PROHBT_CONTENT || item.prohibtContent || `${name}은(는) 임산부가 복용하면 안 돼요.`,
        grade: item.PROHBT_GRADE || item.prohibtGrade || '',
        note: item.REMARK || item.remark || '',
      })
    }
  }
  return warnings
}

async function checkDurElderly(drugNames) {
  const warnings = []
  for (const name of drugNames) {
    const items = await fetchDurApi(DUR_ENDPOINTS.노인주의, name)
    if (items.length > 0) {
      const item = items[0]
      warnings.push({
        type: '노인주의', level: 'caution', drugs: [name],
        reason: item.ATENT_CONTENT || item.atentContent || `${name}은(는) 노인이 복용 시 주의가 필요해요.`,
        note: item.REMARK || item.remark || '',
      })
    }
  }
  return warnings
}

async function checkDurDuplicate(drugNames) {
  if (!drugNames || drugNames.length < 2) return []
  const warnings = []
  const efcyGroups = {}
  for (const name of drugNames) {
    const items = await fetchDurApi(DUR_ENDPOINTS.효능군중복, name)
    if (items.length > 0) {
      const code = items[0].EFCY_GROUP_NO || items[0].efcyGroupNo || null
      const groupName = items[0].EFCY_GROUP_NAME || items[0].efcyGroupName || null
      if (code) {
        if (!efcyGroups[code]) efcyGroups[code] = { groupName, drugs: [] }
        efcyGroups[code].drugs.push(name)
      }
    }
  }
  for (const [code, group] of Object.entries(efcyGroups)) {
    if (group.drugs.length >= 2) {
      warnings.push({
        type: '효능군중복', level: 'caution', drugs: group.drugs,
        reason: `${group.groupName || '동일 효능군'} 계열 약이 ${group.drugs.length}개예요. 중복 복용 주의!`,
        note: `효능군 코드: ${code}`,
      })
    }
  }
  return warnings
}

async function runDurCheck(pillResults, userProfile = {}) {
  const drugNames = pillResults.map(p => p.drugNameForSearch || p.summary).filter(Boolean)
  if (drugNames.length === 0) return []
  const checks = [checkDurCombination(drugNames)]
  if (userProfile.isPregnant) checks.push(checkDurPregnancy(drugNames))
  if (userProfile.isElderly)  checks.push(checkDurElderly(drugNames))
  if (drugNames.length >= 2)  checks.push(checkDurDuplicate(drugNames))
  const results = await Promise.all(checks)
  return results.flat()
}

// ─── 알약 종합 분석 ───────────────────────────────────────────────────────────
async function analyzePillsCombined(pillResults, symptom) {
  if (!GROQ_API_KEY || pillResults.length === 0) return null
  try {
    const pillSummary = pillResults.map((p, i) => `${i+1}. ${p.summary} - ${p.description}`).join('\n')
    const data = await safeFetchGroq({
      model: GROQ_MODEL,
      messages: [{
        role: 'system',
        content: '당신은 친절한 AI 약사입니다. 쉽고 짧게 설명하세요. 전문용어 금지.'
      }, {
        role: 'user',
        content: `다음 약들을 분석해주세요:\n${pillSummary}\n\n사용자 증상: ${symptom || '없음'}\n\n아래 JSON만 반환하세요 (마크다운 금지):\n{\n  "combinedUse": "이 약들을 함께 먹는 이유 1-2문장 (쉬운 말로)",\n  "matchScore": "증상과 일치도 (높음/보통/낮음/알수없음)",\n  "matchReason": "증상과 맞는지 이유 1문장",\n  "recommendation": "추천합니다 | 주의가 필요해요 | 확인이 필요해요",\n  "recommendCode": "safe | caution | danger",\n  "oneLineSummary": "20자 이내 핵심 한줄 요약"\n}`
      }],
      temperature: 0.3,
      max_tokens: 300,
    })
    const raw = data.choices?.[0]?.message?.content || '{}'
    return JSON.parse(raw.replace(/```json|```/g, '').trim())
  } catch (e) {
    console.warn('종합 분석 실패:', e.message)
    return null
  }
}

// ─── 알약 1개 전체 분석 ───────────────────────────────────────────────────────
async function analyzeSinglePill(pillFeature, symptomHint) {
  let pillData = null

  // 1단계: Vision이 약 이름 읽었으면 이름으로 먼저 검색
  if (pillFeature.drugName && pillFeature.drugName.trim().length > 0) {
    pillData = await fetchPillByName(pillFeature.drugName.trim())
    if (!pillData && pillFeature.imprint && pillFeature.imprint.trim().length > 0) {
      pillData = await fetchPillByName(pillFeature.imprint.trim())
    }
  }

  // 2단계: 각인 단독 검색
  if (!pillData && pillFeature.imprint && pillFeature.imprint.trim().length > 0) {
    pillData = await fetchPillByName(pillFeature.imprint.trim())
  }

  // 3단계: 색상/모양으로 fallback
  if (!pillData) {
    pillData = await fetchPillByFeature({
      color: pillFeature.color,
      shape: pillFeature.shape,
      imprint: pillFeature.imprint,
      form: pillFeature.form,
    })
  }

  // 4단계: 약품명으로 개요 + 제품허가 병렬 조회
  let drugInfo = null
  let permitInfo = null
  if (pillData?.itemName) {
    ;[drugInfo, permitInfo] = await Promise.all([
      fetchMfdsInfo(pillData.itemName),
      fetchDrugPermission(pillData.itemName),
    ])
  }

  if (pillData) {
    let efcySummary = drugInfo?.efcyQesitm || ''
    let atpnSummary = drugInfo?.atpnQesitm || ''
    let useSummary  = drugInfo?.useMethodQesitm || ''
    if (efcySummary.length > 100) efcySummary = await summarizeMfdsText('효능', efcySummary)
    if (atpnSummary.length > 100) atpnSummary = await summarizeMfdsText('주의사항', atpnSummary)
    if (useSummary.length  > 80)  useSummary  = await summarizeMfdsText('복용법', useSummary)
    const etcOtc = permitInfo?.etcOtcName || (drugInfo ? '처방약' : '-')
    return {
      statusCode: 'caution',
      statusText: '복용 전 확인하세요',
      summary: pillData.itemName || pillFeature.drugName || `${pillFeature.color} ${pillFeature.shape} 알약`,
      drugNameForSearch: pillData.itemName,
      description: efcySummary || `${pillFeature.color}색 ${pillFeature.shape} 알약이에요.`,
      warnings: atpnSummary || '복용 전 약사에게 확인하세요.',
      dosageGuide: useSummary || '-',
      interactions: drugInfo?.intrcQesitm ? [drugInfo.intrcQesitm.slice(0, 60)] : [],
      activeIngredients: permitInfo?.ingrName ? [permitInfo.ingrName] : pillData.itemName ? [pillData.itemName] : [],
      drugType:      etcOtc,
      confidence:    0.9,
      pillColor:     pillFeature.color,
      pillShape:     pillFeature.shape,
      pillImprint:   pillFeature.imprint,
      itemImage:     pillData?.itemImage    || null,
      entpName:      permitInfo?.entpName   || pillData?.entpName || null,
      permitDate:    permitInfo?.itemPermitDate || null,
      storageMethod: permitInfo?.storageMethod  || null,
      validTerm:     permitInfo?.validTerm      || null,
      packUnit:      permitInfo?.packUnit       || null,
      cancelName:    permitInfo?.cancelName     || null,
      mfdsFound:     true,
      permitFound:   !!permitInfo,
    }
  }

  return {
    statusCode:  'caution',
    statusText:  '식약처 DB 미등록',
    summary:     pillFeature.drugName || `${pillFeature.color} ${pillFeature.shape} 알약`,
    description: `${pillFeature.color}색 ${pillFeature.shape} 알약이에요. ${pillFeature.imprint ? `각인: ${pillFeature.imprint}` : '각인 없음'}`,
    warnings:    '식약처 DB에서 찾을 수 없어요. 처방한 의사/약사에게 확인하세요.',
    dosageGuide: '-',
    interactions: [],
    activeIngredients: [],
    drugType:    '-',
    confidence:  0.2,
    pillColor:   pillFeature.color,
    pillShape:   pillFeature.shape,
    pillImprint: pillFeature.imprint,
    mfdsFound:   false,
    permitFound: false,
  }
}

// ─── AI Vision 프롬프트 ───────────────────────────────────────────────────────
const buildVisionPrompt = (userConditions, symptom) => `
당신은 약학 전문가 + 이미지 분석 전문가입니다. 이미지 속 알약을 정밀 분석하세요.

## 분석 순서 (반드시 이 순서대로)

### STEP 1. 각인/표면 텍스트 읽기
알약 표면의 숫자, 영문, 한글을 정확히 읽으세요.
예시: TYLENOL, 500, ER, TL, 게보린, 펜잘 등

### STEP 2. 외형 특징 추출
색상, 모양, 제형, 크기를 아래 식약처 기준 단어로 추출하세요.
(색상/모양은 반드시 아래 허용 단어만 사용)

## 색상 (이 단어만 허용)
하양, 노랑, 주황, 분홍, 빨강, 갈색, 연두, 초록, 청록, 파랑, 남색, 보라, 회색, 검정, 투명

## 모양 (이 단어만 허용)
원형, 타원형, 장방형, 삼각형, 사각형, 마름모형, 오각형, 육각형, 팔각형, 기타

## 제형
정제, 경질캡슐, 연질캡슐, 필름코팅정

### STEP 3. 약 이름 종합 추론 (핵심!)
STEP1 + STEP2에서 수집한 모든 정보를 종합해서 이 알약이 어떤 약인지 추론하세요.

종합 근거:
- STEP1 각인 텍스트
- STEP2 색상 + 모양 + 제형 + 크기 조합
- 사용자 증상: ${symptom || '없음'}
- 기저질환: ${userConditions || '없음'}

추론 예시:
- 각인 "TYLENOL 500" + 하양 원형 → drugName: "타이레놀"
- 각인 "500" + 하양 원형 + 증상 두통 → drugName: "타이레놀500mg 추정"
- 각인 없음 + 분홍 타원형 + 증상 소화불량 → drugName: "소화제 계열 추정"
- 각인 없음 + 추론 불가 → drugName: ""

JSON만 반환 (마크다운/설명 절대 금지):
{
  "pills": [
    {
      "drugName": "STEP3에서 추론한 약 이름 (예: 타이레놀, 게보린, 소화제 계열 추정 / 추론 불가면 빈 문자열)",
      "color": "색상 (식약처 기준 단어만)",
      "shape": "모양 (식약처 기준 단어만)",
      "form": "제형",
      "imprint": "각인 문자 전체 (없으면 빈 문자열)",
      "size": "크기 (소/중/대)",
      "description": "외형 + 추론 근거 1문장 (예: 하양 원형 정제, 각인 TYLENOL 500으로 타이레놀로 추정)"
    }
  ],
  "totalCount": 알약_개수,
  "symptomHint": "증상 기반 예상 약품 종류 (예: 해열진통제, 소화제)"
}

알약이 안 보이면: {"pills": [], "totalCount": 0, "symptomHint": ""}
`

// ─── 채팅 시스템 프롬프트 (신뢰도 3케이스) ───────────────────────────────────
const buildChatSystemPrompt = (analysisResult, mfdsInfo, userConditions) => {
  const pct = Math.round((analysisResult?.confidence || 0) * 100)
  const drugName = analysisResult?.summary || '미분석'

  const highConfidencePrompt = `
당신은 식약처 공공 데이터를 기반으로 의약품 정보를 매칭해주는 '의약품 정보 분석 전문가'입니다.
모든 판단의 근거는 식약처 공식 허가 데이터에 기반합니다.

[응답 원칙]
1. "약사 상담 필요" 같은 유보적 표현을 쓰지 않습니다. 식약처에 등록된 정확한 용도와 효능을 즉시 답변합니다.
2. 사용자가 "이 약이 [질환명] 약이 맞나요?" 라고 물으면 "식약처 허가 기준: [질환명] 치료 및 완화용"이라고 명확히 답합니다.
3. 성분을 나열할 때 각 성분이 신체에서 어떤 작용(예: 기침 억제, 염증 완화)을 하는지 구체적으로 설명합니다.
4. 답변은 아래 템플릿 구조로 출력합니다:

[식약처 데이터 분석 결과]
의약품 명칭: {약 이름}
공식 허가 용도: 식약처에 '{효능군} 증상 완화' 효능으로 등록된 의약품입니다.
주요 작용:
  • [성분 A]: [신체 작용 설명]
  • [성분 B]: [신체 작용 설명]
현재 상태: 해당 증상에 대한 식약처의 공식적인 사용 승인 정보와 일치합니다.

현재 분석된 약품: ${drugName}
사용자 기저질환: ${userConditions || '없음'}
${mfdsInfo ? `\n식품의약품안전처 공식 정보:\n- 효능: ${mfdsInfo.efcyQesitm || '-'}\n- 복용법: ${mfdsInfo.useMethodQesitm || '-'}\n- 주의사항: ${mfdsInfo.atpnQesitm || '-'}\n- 부작용: ${mfdsInfo.seQesitm || '-'}` : ''}
`

  const midConfidencePrompt = `
당신은 식약처 공공 데이터를 기반으로 의약품 정보를 매칭해주는 '의약품 정보 분석 전문가'입니다.
현재 분석 일치율은 ${pct}%로 중간 수준입니다.

[응답 원칙]
1. 모든 답변에 "식약처 데이터베이스와 ${pct}% 일치하는 의약품 정보"임을 명시합니다.
2. 정보를 제공할 때 "데이터 매칭 결과"임을 강조하여 최종 판단은 사용자에게 있음을 안내합니다.
3. 답변 말미에 항상 아래 주의 문구를 포함합니다:
   "[주의] 분석 일치율이 80% 미만인 경우, 사진 상태에 따라 정보 왜곡이 발생할 수 있습니다. 본 앱은 데이터 대조 결과만을 제공하며, 최종 복용 결정에 따른 책임은 전적으로 사용자에게 있습니다. 실제 약품 외형을 반드시 대조한 후 복용하십시오."

현재 분석된 약품: ${drugName} (일치율 ${pct}%)
사용자 기저질환: ${userConditions || '없음'}
${mfdsInfo ? `\n식품의약품안전처 공식 정보:\n- 효능: ${mfdsInfo.efcyQesitm || '-'}\n- 복용법: ${mfdsInfo.useMethodQesitm || '-'}\n- 주의사항: ${mfdsInfo.atpnQesitm || '-'}` : ''}
`

  const lowConfidencePrompt = `
당신은 식약처 공공 데이터를 기반으로 의약품 정보를 매칭해주는 '의약품 정보 분석 전문가'입니다.
현재 분석 일치율은 ${pct}%로 안전한 정보 제공이 어렵습니다.

[응답 원칙]
1. 약품 정보를 직접 제공하지 않습니다.
2. 모든 질문에 대해 아래와 같이 안내합니다:
   "현재 데이터 일치율이 현저히 낮아(${pct}%), 잘못된 정보 제공으로 인한 약물 오남용 위험이 감지되었습니다. 사용자의 안전을 최우선으로 하여 분석 결과를 제공하지 않습니다."
3. 대신 아하 게시판 약사 상담을 권유합니다: https://www.a-ha.io/topic/%EC%95%BD%EC%98%81%EC%96%91%EC%A0%9C/%EC%95%BD%EB%B3%B5%EC%9A%A9?order=answerRegistration
`

  if (pct >= 80) return highConfidencePrompt
  if (pct >= 50) return midConfidencePrompt
  return lowConfidencePrompt
}

// ─── 상태 매핑 ────────────────────────────────────────────────────────────────
const STATUS_MAP = {
  safe: { icon: CheckCircle, bg: 'bg-green-50', border: 'border-emerald-200', text: 'text-emerald-700', badge: 'bg-green-100 text-emerald-800', label: '복용 가능' },
  caution: { icon: AlertTriangle, bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', badge: 'bg-amber-100 text-amber-800', label: '주의 필요' },
  danger: { icon: XCircle, bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-100 text-red-800', label: '복용 위험' },
  unidentified: { icon: XCircle, bg: 'bg-slate-50', border: 'border-slate-200', text: 'text-slate-600', badge: 'bg-slate-100 text-slate-700', label: '인식 불가' },
}

// ─── DUR 경고 카드 ────────────────────────────────────────────────────────────
function DurWarningCard({ warnings }) {
  if (!warnings || warnings.length === 0) return null
  const TYPE_STYLE = {
    병용금기:   { bg: 'bg-red-50',    border: 'border-red-200',    badge: 'bg-red-100 text-red-700',    icon: '🚫' },
    임부금기:   { bg: 'bg-pink-50',   border: 'border-pink-200',   badge: 'bg-pink-100 text-pink-700',  icon: '🤰' },
    노인주의:   { bg: 'bg-amber-50',  border: 'border-amber-200',  badge: 'bg-amber-100 text-amber-700',icon: '👴' },
    효능군중복: { bg: 'bg-orange-50', border: 'border-orange-200', badge: 'bg-orange-100 text-orange-700', icon: '⚠️' },
  }
  return (
    <div className="space-y-3 animate-slide-up">
      <p className="text-xs font-bold text-red-400 uppercase tracking-wide px-1 flex items-center gap-1">
        <span>🛡️</span> DUR 안전성 경고 {warnings.length}건
      </p>
      {warnings.map((w, i) => {
        const s = TYPE_STYLE[w.type] || TYPE_STYLE['효능군중복']
        return (
          <div key={i} className={`rounded-2xl border-2 ${s.border} ${s.bg} p-4 space-y-2`}>
            <div className="flex items-center gap-2">
              <span className="text-lg">{s.icon}</span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${s.badge}`}>{w.type}</span>
              <span className="text-xs text-slate-500">{w.drugs.join(' + ')}</span>
            </div>
            <p className="text-sm font-semibold text-slate-800">{w.reason}</p>
            {w.note ? <p className="text-xs text-slate-400">{w.note}</p> : null}
          </div>
        )
      })}
    </div>
  )
}

// ─── 알약 리스트 카드 ─────────────────────────────────────────────────────────
function PillListCard({ pillResults, onSelectPill, selectedIdx }) {
  if (!pillResults || pillResults.length === 0) return null
  return (
    <div className="space-y-3 animate-slide-up">
      <p className="text-xs font-bold text-slate-400 uppercase tracking-wide px-1">
        분석된 알약 {pillResults.length}개
      </p>
      {pillResults.map((pill, i) => {
        const s = STATUS_MAP[pill.statusCode] || STATUS_MAP.caution
        const StatusIcon = s.icon
        const isSelected = selectedIdx === i
        return (
          <button
            key={i}
            onClick={() => onSelectPill(i)}
            className={`w-full text-left rounded-2xl border-2 p-4 transition-all ${isSelected ? `${s.border} ${s.bg}` : 'border-slate-100 bg-white'}`}
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 text-lg"
                style={{ background: pill.pillColor === '하양' ? '#f1f5f9' : pill.pillColor === '분홍' ? '#fce7f3' : pill.pillColor === '파랑' ? '#dbeafe' : pill.pillColor === '노랑' ? '#fef9c3' : pill.pillColor === '연두' ? '#dcfce7' : '#f1f5f9' }}>
                💊
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="font-bold text-sm text-slate-800 truncate">{pill.summary}</p>
                  {pill.mfdsFound && <span className="text-[10px] bg-blue-100 text-blue-600 px-1.5 py-0.5 rounded-full font-semibold shrink-0">식약처</span>}
                </div>
                <p className="text-xs text-slate-400 mt-0.5">{pill.pillColor} · {pill.pillShape}{pill.pillImprint ? ` · 각인: ${pill.pillImprint}` : ''}{pill.entpName ? ` · ${pill.entpName}` : ''}</p>
                <p className="text-xs text-slate-500 mt-1 line-clamp-2">{pill.description}</p>
              </div>
              <div className={`text-xs font-bold px-2 py-1 rounded-full shrink-0 ${s.badge}`}>
                {pill.mfdsFound ? '확인됨' : '미확인'}
              </div>
            </div>
            {isSelected && pill.warnings && (
              <div className="mt-3 pt-3 border-t border-slate-100 space-y-2">
                <div className="flex gap-2">
                  <span className="text-xs font-bold text-slate-400 w-14 shrink-0">복용법</span>
                  <span className="text-xs text-slate-600">{pill.dosageGuide}</span>
                </div>
                <div className="flex gap-2">
                  <span className="text-xs font-bold text-slate-400 w-14 shrink-0">주의사항</span>
                  <span className="text-xs text-slate-600">{pill.warnings}</span>
                </div>
                {pill.entpName && (
                  <div className="flex gap-2">
                    <span className="text-xs font-bold text-slate-400 w-14 shrink-0">제조사</span>
                    <span className="text-xs text-slate-600">{pill.entpName}</span>
                  </div>
                )}
              </div>
            )}
          </button>
        )
      })}
    </div>
  )
}

function ResultCard({ result, mfdsInfo, onChat, onRetry }) {
  const statusCode = result?.statusCode || 'unidentified'
  const s = STATUS_MAP[statusCode] || STATUS_MAP.unidentified
  const StatusIcon = s.icon
  const [showMfds, setShowMfds] = useState(false)

  if (!result || result.statusCode === 'unidentified') {
    return (
      <div className={`rounded-3xl border-2 ${s.border} ${s.bg} p-6 space-y-4 animate-slide-up`}>
        <div className="flex items-center gap-3">
          <StatusIcon className={`${s.text} shrink-0`} size={28} />
          <div>
            <p className={`font-bold text-lg ${s.text}`}>{result?.summary || '약품 미인식'}</p>
            <p className="text-sm text-slate-500">{result?.description || '이미지를 다시 촬영해주세요.'}</p>
          </div>
        </div>
        <button onClick={onRetry} className="w-full py-3 rounded-2xl bg-slate-800 text-white font-semibold flex items-center justify-center gap-2 active:scale-95 transition-transform">
          <RefreshCw size={16} /> 다시 촬영하기
        </button>
      </div>
    )
  }

  const RECOMMEND_MAP = {
    safe:    { text: '추천합니다!',        bg: 'bg-green-500', emoji: '✅' },
    caution: { text: '주의가 필요해요!',   bg: 'bg-amber-500', emoji: '⚠️' },
    danger:  { text: '추천하지 않습니다!', bg: 'bg-red-500',   emoji: '❌' },
  }
  const rec = RECOMMEND_MAP[statusCode] || RECOMMEND_MAP.caution

  return (
    <div className={`rounded-3xl border-2 ${s.border} ${s.bg} overflow-hidden animate-slide-up`}>
      <div className={`${rec.bg} px-5 py-4 flex items-center justify-center gap-2`}>
        <span className="text-2xl">{rec.emoji}</span>
        <p className="text-white font-black text-2xl tracking-tight">{rec.text}</p>
      </div>

      {result.oneLineSummary && (
        <div className="px-5 py-3 bg-white border-b border-slate-100">
          <p className="text-slate-700 font-semibold text-sm text-center">{result.oneLineSummary}</p>
        </div>
      )}

      {mfdsInfo && (
        <div className="px-5 py-2 bg-blue-50 border-b border-blue-100 flex items-center gap-2">
          <Database size={13} className="text-[#0192F5]" />
          <p className="text-xs text-[#0192F5] font-semibold">식품의약품안전처 공식 정보 확인됨</p>
          <span className="ml-auto text-xs text-blue-300">{mfdsInfo.entpName}</span>
        </div>
      )}

      {result.permitFound && (
        <div className="px-5 py-2 bg-purple-50 border-b border-purple-100 flex items-center gap-2">
          <Shield size={13} className="text-purple-500" />
          <p className="text-xs text-purple-600 font-semibold">의약품 제품허가 정보 연동됨</p>
          {result.etcOtcName && (
            <span className="ml-auto text-xs bg-purple-100 text-purple-600 px-2 py-0.5 rounded-full font-semibold">{result.drugType}</span>
          )}
        </div>
      )}

      <div className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <StatusIcon className={`${s.text} shrink-0`} size={24} />
            <div className="min-w-0">
              <p className={`font-black text-lg leading-tight ${s.text} truncate`}>{result.summary}</p>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${s.badge}`}>{result.statusText || s.label}</span>
            </div>
          </div>
          <span className="text-xs bg-slate-100 text-slate-500 px-2 py-1 rounded-full shrink-0">{result.drugType}</span>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed">{result.description}</p>
      </div>

      <div className="mx-4 mb-4 bg-white rounded-2xl divide-y divide-slate-100 shadow-sm">
        <InfoRow icon={Clock} label="복용 방법" value={mfdsInfo?.useMethodQesitm || result.dosageGuide} source={mfdsInfo?.useMethodQesitm ? '식약처' : 'AI'} />
        <InfoRow icon={Shield} label="주의사항" value={mfdsInfo?.atpnQesitm || result.warnings} source={mfdsInfo?.atpnQesitm ? '식약처' : 'AI'} />
        {(mfdsInfo?.seQesitm) && <InfoRow icon={AlertTriangle} label="부작용" value={mfdsInfo.seQesitm} source="식약처" />}
      </div>

      {result.activeIngredients?.length > 0 && (
        <div className="px-4 pb-3 flex flex-wrap gap-1.5">
          {result.activeIngredients.map((ing, i) => (
            <span key={i} className="text-xs bg-white text-slate-600 px-2.5 py-1 rounded-full border border-slate-200 font-medium">{ing}</span>
          ))}
        </div>
      )}

      {mfdsInfo && (
        <div className="mx-4 mb-4">
          <button onClick={() => setShowMfds(!showMfds)} className="w-full py-2.5 rounded-2xl border border-blue-100 bg-blue-50 text-xs text-[#0192F5] font-semibold flex items-center justify-center gap-2">
            <Database size={13} />
            {showMfds ? '식약처 공식 정보 접기' : '식약처 공식 정보 더 보기'}
          </button>
          {showMfds && (
            <div className="mt-2 bg-white rounded-2xl border border-blue-100 divide-y divide-slate-50 overflow-hidden">
              {mfdsInfo.efcyQesitm && <MfdsRow label="효능" value={mfdsInfo.efcyQesitm} />}
              {mfdsInfo.atpnWarnQesitm && <MfdsRow label="경고" value={mfdsInfo.atpnWarnQesitm} highlight />}
              {mfdsInfo.intrcQesitm && <MfdsRow label="상호작용" value={mfdsInfo.intrcQesitm} />}
              {mfdsInfo.depositMethodQesitm && <MfdsRow label="보관법" value={mfdsInfo.depositMethodQesitm} />}
              {result.permitFound && (
                <>
                  {result.permitDate    && <MfdsRow label="허가일자"  value={result.permitDate} />}
                  {result.storageMethod && <MfdsRow label="저장방법"  value={result.storageMethod} />}
                  {result.validTerm     && <MfdsRow label="유효기간"  value={result.validTerm} />}
                  {result.packUnit      && <MfdsRow label="포장단위"  value={result.packUnit} />}
                  {result.cancelName    && result.cancelName !== '정상' && (
                    <MfdsRow label="허가상태" value={result.cancelName} highlight />
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}

      {result.interactions?.length > 0 && (
        <div className="mx-4 mb-4 p-3 bg-amber-50 rounded-2xl border border-amber-100">
          <p className="text-xs font-bold text-amber-700 mb-1 flex items-center gap-1">
            <AlertTriangle size={12} /> 병용 주의
          </p>
          <p className="text-xs text-amber-600">{result.interactions.join(', ')}</p>
        </div>
      )}

      {/* 신뢰도 — 3케이스 분기 */}
      {result.confidence !== undefined && (() => {
        const pct = Math.round((result.confidence || 0) * 100)

        if (pct >= 80) return (
          <div className="mx-4 mb-4 rounded-2xl p-4 space-y-3" style={{ background: '#eff6ff', border: '2px solid #bfdbfe' }}>
            <div className="flex items-center gap-3">
              <div className="text-center shrink-0">
                <p className="font-black text-4xl leading-none text-[#0192F5]">{pct}%</p>
                <p className="text-xs font-medium mt-1 text-[#0192F5]">데이터 일치율</p>
              </div>
              <div className="flex-1">
                <div className="h-3 bg-white rounded-full overflow-hidden">
                  <div className="h-full rounded-full bg-[#0192F5]" style={{ width: `${pct}%` }} />
                </div>
                <p className="text-xs mt-2 font-bold text-[#0192F5]">✅ 식약처 데이터베이스와 일치합니다</p>
              </div>
            </div>
            {result.description && (
              <div className="bg-white rounded-xl p-3 border border-blue-100">
                <p className="text-[10px] font-bold text-blue-400 uppercase tracking-wide mb-1">식약처 데이터 분석 결과</p>
                <p className="text-sm text-slate-700 leading-relaxed font-medium">{result.description}</p>
              </div>
            )}
          </div>
        )

        if (pct >= 50) return (
          <div className="mx-4 mb-4 rounded-2xl overflow-hidden" style={{ border: '2px solid #fde68a' }}>
            <div className="px-4 py-3 flex items-center gap-2" style={{ background: '#fffbeb' }}>
              <span className="text-base">⚠️</span>
              <p className="text-xs font-black text-amber-700 flex-1">데이터 정밀 분석 중</p>
              <span className="text-xs font-black text-amber-700 bg-amber-100 px-2 py-0.5 rounded-full">일치율 {pct}%</span>
            </div>
            <div className="px-4 py-3 bg-white space-y-2">
              <p className="text-sm text-slate-700 leading-relaxed">
                식약처 데이터베이스와 <span className="font-black text-amber-600">{pct}% 일치</span>하는 의약품 정보는{' '}
                <span className="font-black text-slate-800">{result.summary}</span>입니다.
                {result.description ? ` 해당 의약품은 주로 ${result.description}` : ''}
              </p>
              <div className="rounded-xl p-3" style={{ background: '#fffbeb', border: '1px solid #fde68a' }}>
                <p className="text-xs text-amber-700 leading-relaxed">
                  <span className="font-black">[주의]</span> 분석 일치율이 80% 미만인 경우, 사진 상태에 따라 정보 왜곡이 발생할 수 있습니다.
                  본 앱은 데이터 대조 결과만을 제공하며, <span className="font-bold">최종 복용 결정에 따른 책임은 전적으로 사용자에게 있습니다.</span>
                </p>
              </div>
              <a
                href="https://www.a-ha.io/topic/%EC%95%BD%EC%98%81%EC%96%91%EC%A0%9C/%EC%95%BD%EB%B3%B5%EC%9A%A9?order=answerRegistration"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 w-full py-3 rounded-xl font-bold text-sm text-white"
                style={{ background: 'linear-gradient(135deg, #f59e0b, #d97706)' }}
              >
                💬 아하 게시판에서 약사에게 질문하기 →
              </a>
            </div>
          </div>
        )

        return (
          <div className="mx-4 mb-4 rounded-2xl overflow-hidden" style={{ border: '2px solid #fecaca' }}>
            <div className="px-4 py-3 flex items-center gap-2 bg-red-500">
              <span className="text-base">🚫</span>
              <p className="text-sm font-black text-white flex-1">복용 위험 감지 — AI 분석 중단</p>
              <span className="text-xs font-bold text-red-200 bg-red-600 px-2 py-0.5 rounded-full">{pct}%</span>
            </div>
            <div className="px-4 py-4 bg-red-50 space-y-3">
              <p className="text-sm text-red-800 leading-relaxed font-medium">
                현재 데이터 일치율이 현저히 낮아({pct}% 미만), 잘못된 정보 제공으로 인한
                <span className="font-black"> 약물 오남용 위험이 감지</span>되었습니다.
              </p>
              <div className="bg-white rounded-xl p-3 border border-red-100">
                <p className="text-xs text-red-600 leading-relaxed">
                  사용자의 안전을 최우선으로 하여 AI 분석 결과를 표시하지 않습니다.
                  아래 게시판을 통해 전문 약사에게 질문하여 안전한 복용 안내를 받으십시오.
                </p>
              </div>
              <a
                href="https://www.a-ha.io/topic/%EC%95%BD%EC%98%81%EC%96%91%EC%A0%9C/%EC%95%BD%EB%B3%B5%EC%9A%A9?order=answerRegistration"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 w-full py-3.5 rounded-xl font-black text-sm text-white"
                style={{ background: 'linear-gradient(135deg, #ef4444, #dc2626)' }}
              >
                💬 아하 게시판에서 약사에게 질문하기 →
              </a>
            </div>
          </div>
        )
      })()}

      <div className="p-4 pt-0">
        <button onClick={onChat} className="w-full py-3.5 rounded-2xl bg-gradient-to-r from-[#0192F5] to-[#40BEFD] text-white font-bold flex items-center justify-center gap-2 shadow-md active:scale-95 transition-all">
          <MessageCircle size={18} /> AI 약사에게 더 물어보기
        </button>
      </div>
    </div>
  )
}

function InfoRow({ icon: Icon, label, value, source }) {
  if (!value) return null
  const displayValue = value.length > 100 ? value.slice(0, 100) + '...' : value
  return (
    <div className="flex gap-3 p-3">
      <div className="w-7 h-7 rounded-xl bg-blue-50 flex items-center justify-center shrink-0 mt-0.5">
        <Icon size={14} className="text-[#0192F5]" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 mb-0.5">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wide">{label}</p>
          {source && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${source === '식약처' ? 'bg-blue-100 text-blue-600' : 'bg-slate-100 text-slate-500'}`}>
              {source}
            </span>
          )}
        </div>
        <p className="text-sm text-slate-700 leading-snug">{displayValue}</p>
      </div>
    </div>
  )
}

function MfdsRow({ label, value, highlight }) {
  if (!value) return null
  const displayValue = value.length > 150 ? value.slice(0, 150) + '...' : value
  return (
    <div className={`p-3 ${highlight ? 'bg-red-50' : ''}`}>
      <p className={`text-xs font-bold mb-1 ${highlight ? 'text-red-600' : 'text-slate-400'}`}>{label}</p>
      <p className="text-xs text-slate-600 leading-relaxed">{displayValue}</p>
    </div>
  )
}

function AnalyzingSkeleton({ mfdsLoading }) {
  return (
    <div className="rounded-3xl border-2 border-blue-100 bg-blue-50 p-6 space-y-4 animate-pulse">
      <div className="flex items-center gap-3">
        <Loader2 size={28} className="text-[#40BEFD] animate-spin" />
        <div className="flex-1 space-y-2">
          <div className="h-5 bg-blue-200 rounded-lg w-3/4" />
          <div className="h-3 bg-blue-200 rounded w-1/2" />
        </div>
      </div>
      <div className="space-y-2">
        <div className="h-3 bg-blue-200 rounded w-full" />
        <div className="h-3 bg-blue-200 rounded w-5/6" />
      </div>
      <p className="text-center text-sm text-[#0192F5] font-medium">
        {mfdsLoading ? '🏥 식약처 DB 검색 중...' : '🔍 AI가 약품을 분석하고 있어요...'}
      </p>
    </div>
  )
}

// ─── 채팅 뷰 ─────────────────────────────────────────────────────────────────
function ChatView({ result, mfdsInfo, userConditions, onBack }) {
  const [messages, setMessages] = useState([{
    role: 'assistant',
    content: `안녕하세요! 👋 **${result?.summary || '분석된 약품'}**에 대해 무엇이든 물어보세요.\n\n복용 방법, 부작용, 다른 약과의 상호작용 등을 도와드릴 수 있어요.${mfdsInfo ? '\n\n✅ 식약처 공식 정보를 바탕으로 답변해드릴게요.' : ''}`,
    ts: Date.now(),
  }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return
    const userMsg = { role: 'user', content: text, ts: Date.now() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)
    try {
      const history = messages.slice(1).map(m => ({ role: m.role, content: m.content }))
      const data = await safeFetchGroq({
        model: GROQ_MODEL,
        messages: [
          { role: 'system', content: buildChatSystemPrompt(result, mfdsInfo, userConditions) },
          ...history,
          { role: 'user', content: text }
        ],
        temperature: 0.7,
        max_tokens: 600,
      })
      const reply = data.choices?.[0]?.message?.content || '죄송합니다, 응답을 가져오지 못했어요.'
      setMessages(prev => [...prev, { role: 'assistant', content: reply, ts: Date.now() }])
    } catch (e) {
      setMessages(prev => [...prev, { role: 'assistant', content: `⚠️ 오류: ${e.message}`, ts: Date.now() }])
    } finally { setLoading(false) }
  }

  return (
    <div className="flex flex-col h-[100dvh]">
      <div className="sticky top-0 z-10 px-4 pt-4 pb-3 border-b border-slate-100 bg-white flex items-center gap-3">
        <button onClick={onBack} className="w-9 h-9 rounded-2xl bg-slate-100 flex items-center justify-center">
          <ChevronLeft size={20} className="text-slate-600" />
        </button>
        <div className="flex-1 min-w-0">
          <p className="font-bold text-slate-800 text-sm truncate">AI 약사 상담</p>
          <p className="text-xs text-slate-400 truncate">{result?.summary}{mfdsInfo ? ' · 식약처 인증' : ''}</p>
        </div>
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#0192F5] to-[#40BEFD] flex items-center justify-center">
          <Sparkles size={15} className="text-white" />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' && (
              <div className="w-7 h-7 rounded-xl bg-gradient-to-br from-[#0192F5] to-[#40BEFD] flex items-center justify-center mr-2 mt-1 shrink-0">
                <Sparkles size={13} className="text-white" />
              </div>
            )}
            <div className={`max-w-[78%] px-4 py-3 rounded-3xl text-sm leading-relaxed whitespace-pre-wrap ${
              msg.role === 'user' ? 'bg-gradient-to-br from-[#0192F5] to-[#40BEFD] text-white rounded-br-lg' : 'bg-slate-100 text-slate-800 rounded-bl-lg'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex items-start gap-2">
            <div className="w-7 h-7 rounded-xl bg-gradient-to-br from-[#0192F5] to-[#40BEFD] flex items-center justify-center shrink-0">
              <Sparkles size={13} className="text-white" />
            </div>
            <div className="bg-slate-100 px-4 py-3 rounded-3xl rounded-bl-lg flex items-center gap-1.5">
              {[0,1,2].map(i => <span key={i} className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: `${i*0.15}s` }} />)}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {messages.length <= 2 && (
        <div className="px-4 pb-2">
          <div className="flex gap-2 overflow-x-auto pb-1">
            {['식전에 먹어도 돼요?', '어떤 효과가 있나요?', '다른 약과 같이 먹어도 되나요?', '부작용이 뭔가요?'].map(q => (
              <button key={q} onClick={() => { setInput(q); inputRef.current?.focus() }}
                className="shrink-0 text-xs bg-blue-50 text-[#0192F5] px-3 py-2 rounded-2xl border border-blue-100 font-medium whitespace-nowrap">
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="px-4 pb-8 pt-2 border-t border-slate-100 bg-white">
        <div className="flex items-end gap-2 bg-slate-100 rounded-3xl px-4 py-2">
          <textarea ref={inputRef} value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
            placeholder="약에 대해 질문하세요..." rows={1}
            className="flex-1 bg-transparent text-sm text-slate-800 placeholder-slate-400 resize-none outline-none max-h-24 py-1.5" />
          <button onClick={sendMessage} disabled={!input.trim() || loading}
            className="w-9 h-9 rounded-2xl bg-gradient-to-br from-[#0192F5] to-[#40BEFD] flex items-center justify-center shrink-0 disabled:opacity-30 active:scale-95 transition-all mb-0.5">
            <Send size={15} className="text-white" />
          </button>
        </div>
        <p className="text-center text-xs text-slate-300 mt-2">AI 정보는 참고용입니다 · 전문의 판단이 우선합니다</p>
      </div>
    </div>
  )
}

// ─── 히스토리 뷰 ─────────────────────────────────────────────────────────────
function HistoryView({ logs, onSelect, onBack }) {
  if (logs.length === 0) {
    return (
      <div className="flex flex-col h-[100dvh]">
        <div className="sticky top-0 z-10 px-4 pt-4 pb-3 border-b border-slate-100 bg-white flex items-center gap-3">
          <button onClick={onBack} className="w-9 h-9 rounded-2xl bg-slate-100 flex items-center justify-center"><ChevronLeft size={20} className="text-slate-600" /></button>
          <p className="flex-1 font-bold text-slate-800">분석 기록</p>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center text-slate-400 space-y-3 px-8">
          <div className="w-16 h-16 rounded-3xl bg-slate-100 flex items-center justify-center">
            <History size={32} className="text-slate-300" />
          </div>
          <p className="text-sm font-medium">아직 분석 기록이 없어요</p>
          <p className="text-xs text-center leading-relaxed">약품 사진을 촬영하면<br/>분석 결과가 여기에 저장됩니다.</p>
        </div>
      </div>
    )
  }
  return (
    <div className="flex flex-col h-[100dvh]">
      <div className="sticky top-0 z-10 px-4 pt-4 pb-3 border-b border-slate-100 bg-white flex items-center gap-3">
        <button onClick={onBack} className="w-9 h-9 rounded-2xl bg-slate-100 flex items-center justify-center"><ChevronLeft size={20} className="text-slate-600" /></button>
        <p className="flex-1 font-bold text-slate-800">분석 기록</p>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        {logs.map((log, i) => {
          const s = STATUS_MAP[log.statusCode] || STATUS_MAP.unidentified
          const StatusIcon = s.icon
          return (
            <button key={log.id || i} onClick={() => onSelect(log)}
              className={`w-full text-left p-4 rounded-2xl border ${s.border} ${s.bg} flex items-center gap-3 transition-all`}>
              <StatusIcon className={`${s.text} shrink-0`} size={22} />
              <div className="flex-1 min-w-0">
                <p className="font-bold text-slate-800 truncate text-sm">{log.summary || '약품명 없음'}</p>
                <p className="text-xs text-slate-400 mt-0.5">{log.statusText || s.label}</p>
                <p className="text-xs text-slate-300 mt-0.5">{log.createdAt?.toDate?.()?.toLocaleDateString('ko-KR') || '날짜 없음'}</p>
              </div>
              <ChevronRight size={16} className="text-slate-300 shrink-0" />
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ─── 관리자 뷰 ────────────────────────────────────────────────────────────────
function AdminView({ logs, onBack }) {
  const total = logs.length
  const trusted = logs.filter(l => (l.confidence || 0) >= 0.8).length
  const untrusted = total - trusted
  const avgConfidence = total > 0 ? Math.round(logs.reduce((sum, l) => sum + (l.confidence || 0), 0) / total * 100) : 0
  const safeCount = logs.filter(l => (l.confidence || 0) >= 0.8 && l.statusCode !== 'danger').length
  const cautionCount = logs.filter(l => (l.confidence || 0) < 0.8 && l.statusCode !== 'danger').length
  const dangerCount = logs.filter(l => l.statusCode === 'danger').length

  return (
    <div className="flex flex-col h-[100dvh] bg-slate-900">
      <div className="px-5 pt-6 pb-4 bg-slate-800 flex items-center gap-3 border-b border-slate-700">
        <button onClick={onBack} className="w-9 h-9 rounded-2xl bg-slate-700 flex items-center justify-center">
          <ChevronLeft size={20} className="text-white" />
        </button>
        <div>
          <p className="font-bold text-white text-sm">관리자 대시보드</p>
          <p className="text-xs text-slate-400">이거돼? 서비스 현황</p>
        </div>
        <div className="ml-auto w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
      </div>
      <div className="flex-1 overflow-y-auto px-5 py-5 space-y-4">
        <div className="bg-slate-800 rounded-3xl p-5 border border-slate-700">
          <p className="text-slate-400 text-xs font-medium mb-1">총 분석 횟수</p>
          <p className="text-4xl font-black text-white">{total}<span className="text-lg text-slate-400 ml-1">회</span></p>
        </div>
        <div className="bg-slate-800 rounded-3xl p-5 border border-slate-700">
          <p className="text-slate-400 text-xs font-medium mb-3">AI 인식 정확도</p>
          <p className="text-4xl font-black mb-3" style={{ color: avgConfidence >= 80 ? '#10b981' : '#f59e0b' }}>{avgConfidence}%</p>
          <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{ width: `${avgConfidence}%`, background: avgConfidence >= 80 ? '#10b981' : '#f59e0b' }} />
          </div>
          <div className="flex justify-between mt-3">
            <div className="text-center"><p className="text-emerald-400 font-bold text-lg">{trusted}</p><p className="text-slate-500 text-xs">신뢰 (80% 이상)</p></div>
            <div className="text-center"><p className="text-amber-400 font-bold text-lg">{untrusted}</p><p className="text-slate-500 text-xs">미신뢰 (80% 미만)</p></div>
          </div>
        </div>
        <div className="bg-slate-800 rounded-3xl p-5 border border-slate-700">
          <p className="text-slate-400 text-xs font-medium mb-3">사회 기여도</p>
          {[['#10b981', '안전 약품 안내', safeCount], ['#f59e0b', '주의 필요 경고', cautionCount], ['#ef4444', '위험 약품 차단', dangerCount]].map(([color, label, count]) => (
            <div key={label} className="flex items-center justify-between mb-3 last:mb-0">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ background: color }} />
                <p className="text-slate-300 text-sm">{label}</p>
              </div>
              <p className="font-bold" style={{ color }}>{count}건</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── 온보딩 슬라이드 ──────────────────────────────────────────────────────────
function OnboardingSlides({ onComplete }) {
  const [current, setCurrent] = useState(0)
  const slides = [
    {
      emoji: '💊',
      title: '이거 돼? 입니다',
      desc: '약 사진 한 장으로 성분, 효능, 주의사항을 바로 확인하세요',
      tips: ['AI가 알약을 직접 분석해드려요', '식약처 공식 데이터로 정확하게 알려드려요', '처방약도 일반약도 모두 확인 가능해요'],
      color: '#0192F5',
    },
    {
      emoji: '📸',
      title: '약을 찍어주세요',
      desc: '약 봉투 안 알약을 카메라로 찍거나 갤러리에서 사진을 올려주세요',
      tips: ['알약이 잘 보이게 가까이 찍어주세요', '각인 문자가 보이면 더 정확해요', '여러 알약이 있으면 한번에 찍어도 돼요'],
      color: '#0192F5',
    },
    {
      emoji: '🔍',
      title: '분석 결과 확인',
      desc: 'AI와 식약처 DB가 함께 각 알약의 성분과 효능을 알려드려요',
      tips: ['알약마다 성분과 효능을 따로 확인할 수 있어요', '식약처 공식 정보를 우선으로 보여드려요', '신뢰도 점수로 결과의 정확성을 알 수 있어요'],
      color: '#16A34A',
    },
    {
      emoji: '💬',
      title: 'AI 약사에게 물어보세요',
      desc: '분석 결과를 바탕으로 궁금한 점을 바로 물어볼 수 있어요',
      tips: ['식전/식후 복용 여부', '다른 약과 함께 먹어도 되는지', '부작용이 있는지'],
      color: '#7C3AED',
    },
  ]

  const next = () => {
    if (current < slides.length - 1) setCurrent(current + 1)
    else onComplete()
  }

  const s = slides[current]

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#fff', zIndex: 100, display: 'flex', flexDirection: 'column', maxWidth: 480, margin: '0 auto' }}>
      <div style={{ padding: '16px 20px', display: 'flex', justifyContent: 'flex-end' }}>
        <button onClick={onComplete} style={{ fontSize: 13, color: '#AAA', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit' }}>
          건너뛰기
        </button>
      </div>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '0 32px', gap: 24 }}>
        <div style={{ width: 100, height: 100, borderRadius: '50%', background: `${s.color}15`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 52 }}>
          {s.emoji}
        </div>
        <div style={{ textAlign: 'center' }}>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#111', letterSpacing: '-0.5px', marginBottom: 10 }}>{s.title}</h2>
          <p style={{ fontSize: 15, color: '#666', lineHeight: 1.7, whiteSpace: 'pre-line' }}>{s.desc}</p>
        </div>
        {current === 0 ? (
          <div style={{ alignSelf: 'stretch', display: 'flex', flexDirection: 'column', gap: 10 }}>
            {['AI 약품 분석', '식약처 공식 데이터 연동', '처방약 · 일반약 모두 가능'].map((feat, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, background: '#F0F7FF', borderRadius: 14, padding: '14px 18px', border: '1px solid #BDE0FF' }}>
                <span style={{ fontSize: 18 }}>{['🤖', '🏥', '💊'][i]}</span>
                <span style={{ fontSize: 14, color: '#0192F5', fontWeight: 600 }}>{feat}</span>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ alignSelf: 'stretch', display: 'flex', flexDirection: 'column', gap: 10 }}>
            {s.tips.map((tip, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, background: `${s.color}08`, borderRadius: 14, padding: '12px 16px', border: `1px solid ${s.color}20` }}>
                <div style={{ width: 24, height: 24, borderRadius: '50%', background: s.color, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <span style={{ color: '#fff', fontSize: 12, fontWeight: 700 }}>{i + 1}</span>
                </div>
                <span style={{ fontSize: 13, color: '#444', lineHeight: 1.5 }}>{tip}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div style={{ padding: '20px 24px 40px', display: 'flex', flexDirection: 'column', gap: 16, alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: 6 }}>
          {slides.map((_, i) => (
            <div key={i} onClick={() => setCurrent(i)} style={{ width: i === current ? 20 : 6, height: 6, borderRadius: 3, background: i === current ? s.color : '#DDD', cursor: 'pointer', transition: 'all 0.3s' }} />
          ))}
        </div>
        <button onClick={next} style={{ width: '100%', padding: '17px 0', borderRadius: 16, background: `linear-gradient(135deg, ${s.color}, ${s.color}CC)`, color: '#fff', border: 'none', fontSize: 16, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', letterSpacing: '-0.2px' }}>
          {current < slides.length - 1 ? '다음' : '시작하기 →'}
        </button>
      </div>
    </div>
  )
}

// ─── 카메라 뷰 ────────────────────────────────────────────────────────────────
function CameraView({ onCapture, onCancel }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let mounted = true
    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        if (!mounted) { stream.getTracks().forEach(t => t.stop()); return }
        streamRef.current = stream
        if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); setReady(true) }
      } catch (e) { setError('카메라 접근 권한이 필요합니다.') }
    }
    start()
    return () => { mounted = false; streamRef.current?.getTracks().forEach(t => t.stop()) }
  }, [])

  const shoot = () => {
    if (!videoRef.current || !ready) return
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0)
    canvas.toBlob(blob => { streamRef.current?.getTracks().forEach(t => t.stop()); onCapture(blob) }, 'image/jpeg', 0.92)
  }

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      <div className="relative flex-1 overflow-hidden">
        <video ref={videoRef} playsInline muted className="absolute inset-0 w-full h-full object-cover" />
        {ready && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-72 h-48 rounded-3xl border-2 border-white/60 relative">
              <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-black/60 text-white text-xs px-3 py-1 rounded-full whitespace-nowrap">약품이 이 안에 들어오게 맞춰주세요</div>
            </div>
          </div>
        )}
        {!ready && !error && <div className="absolute inset-0 flex items-center justify-center"><Loader2 size={40} className="text-white animate-spin" /></div>}
        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center px-8 text-center space-y-4">
            <XCircle size={48} className="text-red-400" />
            <p className="text-white text-sm">{error}</p>
            <button onClick={onCancel} className="px-6 py-2 bg-white text-slate-800 rounded-full font-semibold">돌아가기</button>
          </div>
        )}
        <button onClick={onCancel} className="absolute top-4 left-4 w-10 h-10 rounded-full bg-black/50 flex items-center justify-center">
          <X size={20} className="text-white" />
        </button>
      </div>
      {ready && (
        <div className="bg-black pb-12 pt-6 flex items-center justify-center">
          <button onClick={shoot} className="w-20 h-20 rounded-full border-4 border-white bg-white/20 flex items-center justify-center active:scale-90 transition-transform">
            <div className="w-14 h-14 rounded-full bg-white" />
          </button>
        </div>
      )}
    </div>
  )
}

// ─── 홈 뷰 ───────────────────────────────────────────────────────────────────
function HomeView({ userConditions, analysisResult, mfdsInfo, pillResults, combinedAnalysis, durWarnings, analyzing, mfdsLoading, onCameraCapture, onGalleryUpload, onChat, onHistory, onRetry, previewUrl, logCount, symptom, onSymptomChange, onLogoTap }) {
  const [selectedPillIdx, setSelectedPillIdx] = useState(0)
  const fileInputRef = useRef(null)
  const [step, setStep] = useState(previewUrl || analysisResult ? 2 : 1)
  if ((previewUrl || analyzing || mfdsLoading) && step === 1) setStep(2)

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) { onGalleryUpload(file); setStep(2) }
    e.target.value = ''
  }

  const AppHeader = () => (
    <div className="px-5 pt-6 pb-5 bg-gradient-to-b from-[#0192F5] to-[#40BEFD]">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <img src="/logo.png" alt="이거돼?" onClick={onLogoTap} className="w-10 h-10 rounded-2xl object-cover cursor-pointer active:scale-90 transition-transform" />
          <div>
            <h1 className="text-white font-black text-lg leading-tight">이거 돼?</h1>
            <p className="text-white/70 text-xs">AI 약물 판독 서비스</p>
          </div>
        </div>
        <button onClick={onHistory} className="relative w-10 h-10 rounded-2xl bg-white/20 flex items-center justify-center">
          <History size={20} className="text-white" />
          {logCount > 0 && <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-400 text-white text-[10px] font-bold rounded-full flex items-center justify-center">{Math.min(logCount, 9)}</span>}
        </button>
      </div>
    </div>
  )

  if (step === 1) {
    return (
      <div className="flex flex-col h-[100dvh]">
        <AppHeader />
        <div className="flex-1 flex flex-col px-5 py-8 space-y-6">
          <div className="text-center space-y-2">
            <div className="text-5xl mb-2">🤒</div>
            <p className="font-black text-slate-800 text-xl">어떤 증상이 있으신가요?</p>
            <p className="text-slate-400 text-sm">증상을 입력하면 더 정확한 분석을 해드려요</p>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-3 bg-slate-50 border-2 border-slate-200 rounded-2xl px-4 py-4 focus-within:border-[#0192F5] transition-colors">
              <input type="text" value={symptom} onChange={e => onSymptomChange(e.target.value)}
                placeholder="예) 두통, 소화불량, 기침, 발열..."
                className="flex-1 bg-transparent text-slate-800 placeholder-slate-400 text-base outline-none"
                onKeyDown={e => e.key === 'Enter' && setStep(2)} autoFocus />
              {symptom && <button onClick={() => onSymptomChange('')} className="text-slate-400"><X size={16} /></button>}
            </div>
            <div className="flex flex-wrap gap-2">
              {['두통', '소화불량', '기침', '발열', '코막힘', '근육통', '복통'].map(s => (
                <button key={s} onClick={() => onSymptomChange(symptom ? symptom + ', ' + s : s)}
                  className="text-sm px-3 py-1.5 rounded-full border border-slate-200 text-slate-600 bg-white active:bg-blue-50 active:border-[#40BEFD] active:text-[#0192F5] transition-all">
                  {s}
                </button>
              ))}
            </div>
          </div>
          <div className="flex-1" />
          <button onClick={() => setStep(2)} className="w-full py-4 rounded-3xl bg-gradient-to-r from-[#0192F5] to-[#40BEFD] text-white font-bold text-base shadow-lg shadow-blue-200 active:scale-95 transition-all">
            {symptom ? '약 사진 찍으러 가기 →' : '증상 없이 바로 찍기 →'}
          </button>
          <p className="text-center text-xs text-slate-300">증상 입력은 선택사항이에요</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-[100dvh]">
      <AppHeader />
      {symptom && (
        <div className="px-5 py-3 bg-blue-50 border-b border-blue-100 flex items-center gap-2">
          <span className="text-xl">🤒</span>
          <p className="text-base text-[#0192F5] font-bold flex-1 truncate">{symptom}</p>
          <button onClick={() => { onSymptomChange(''); setStep(1) }} className="text-blue-300"><X size={16} /></button>
        </div>
      )}
      <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5 pb-28">
        {previewUrl && (
          <div className="relative rounded-3xl overflow-hidden bg-slate-100 aspect-video shadow-md">
            <img src={previewUrl} alt="약품 사진" className="w-full h-full object-cover" />
          </div>
        )}
        {(analyzing || mfdsLoading) && <AnalyzingSkeleton mfdsLoading={mfdsLoading} />}
        {!analyzing && !mfdsLoading && combinedAnalysis && (
          <div className={`rounded-2xl p-4 border-2 ${
            combinedAnalysis.recommendCode === 'safe' ? 'bg-green-50 border-emerald-200' :
            combinedAnalysis.recommendCode === 'danger' ? 'bg-red-50 border-red-200' :
            'bg-amber-50 border-amber-200'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xl">
                {combinedAnalysis.recommendCode === 'safe' ? '✅' : combinedAnalysis.recommendCode === 'danger' ? '❌' : '⚠️'}
              </span>
              <p className={`font-black text-lg ${
                combinedAnalysis.recommendCode === 'safe' ? 'text-emerald-700' :
                combinedAnalysis.recommendCode === 'danger' ? 'text-red-700' : 'text-amber-700'
              }`}>{combinedAnalysis.recommendation}</p>
            </div>
            <p className="text-sm text-slate-700 font-medium mb-1">{combinedAnalysis.oneLineSummary}</p>
            <p className="text-sm text-slate-600 leading-relaxed">{combinedAnalysis.combinedUse}</p>
            {symptom && (
              <div className="mt-2 pt-2 border-t border-slate-200 flex items-start gap-2">
                <span className="text-xs font-bold text-slate-400 shrink-0">증상 비교</span>
                <span className={`text-xs font-semibold ${
                  combinedAnalysis.matchScore === '높음' ? 'text-emerald-600' :
                  combinedAnalysis.matchScore === '낮음' ? 'text-red-500' : 'text-amber-600'
                }`}>{combinedAnalysis.matchScore}</span>
                <span className="text-xs text-slate-500">{combinedAnalysis.matchReason}</span>
              </div>
            )}
          </div>
        )}
        {!analyzing && !mfdsLoading && durWarnings?.length > 0 && (
          <DurWarningCard warnings={durWarnings} />
        )}
        {!analyzing && !mfdsLoading && pillResults.length > 0 && (
          <PillListCard
            pillResults={pillResults}
            selectedIdx={selectedPillIdx}
            onSelectPill={setSelectedPillIdx}
          />
        )}
        {!analyzing && !mfdsLoading && pillResults.length === 0 && analysisResult && analysisResult.statusCode === 'unidentified' && (
          <ResultCard result={analysisResult} mfdsInfo={null} onChat={onChat} onRetry={() => { onRetry(); setStep(2) }} />
        )}
        {!analyzing && !mfdsLoading && pillResults.length > 0 && (
          <button
            onClick={onChat}
            className="w-full py-3.5 rounded-2xl bg-gradient-to-r from-[#0192F5] to-[#40BEFD] text-white font-bold flex items-center justify-center gap-2 shadow-md active:scale-95 transition-all"
          >
            <MessageCircle size={18} /> AI 약사에게 더 물어보기
          </button>
        )}
        {!previewUrl && !analyzing && !analysisResult && (
          <div className="text-center py-8 space-y-4">
            <div className="w-24 h-24 rounded-full bg-blue-50 flex items-center justify-center mx-auto">
              <Camera size={40} className="text-[#40BEFD]" />
            </div>
            <div className="space-y-1.5">
              <p className="font-bold text-slate-700">약 사진을 찍어주세요</p>
              <p className="text-sm text-slate-400 leading-relaxed">약 봉투, 약통, 낱알 모두 가능해요<br />AI + 식약처 DB로 정확하게 분석해드려요</p>
            </div>
            <div className="mt-6 space-y-2 text-left">
              {[['💊', '약 이름이 보이게 찍으면 더 정확해요'], ['📋', '처방전이나 약 봉투도 인식 가능해요'], ['🔍', '흐리지 않게 가까이서 촬영해주세요'], ['🏥', 'AI 분석 후 식약처 DB에서 공식 정보도 확인해요']].map(([emoji, text], i) => (
                <div key={i} className="flex items-center gap-2.5 bg-slate-50 rounded-2xl px-4 py-2.5">
                  <span className="text-lg">{emoji}</span>
                  <p className="text-xs text-slate-500">{text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      {step === 2 && (
        <div className="fixed bottom-0 left-1/2 -translate-x-1/2 w-full max-w-[480px] px-5 pb-8 pt-4 bg-gradient-to-t from-white via-white to-transparent">
          {(analyzing || mfdsLoading || pillResults.length > 0 || (analysisResult && analysisResult.statusCode === 'unidentified')) ? (
            <button
              onClick={() => { onRetry(); setStep(2) }}
              disabled={analyzing || mfdsLoading}
              className="w-full py-4 rounded-2xl bg-slate-100 text-slate-600 font-bold flex items-center justify-center gap-2 disabled:opacity-40"
            >
              <RefreshCw size={18} /> 다시 촬영하기
            </button>
          ) : (
            <div className="flex gap-3">
              <button onClick={() => fileInputRef.current?.click()} className="flex-1 py-4 rounded-2xl bg-slate-100 text-slate-600 font-bold flex items-center justify-center gap-2">
                <ImagePlus size={20} /> 갤러리
              </button>
              <button onClick={onCameraCapture} className="flex-[2] py-4 rounded-2xl bg-gradient-to-r from-[#0192F5] to-[#40BEFD] text-white font-bold text-base flex items-center justify-center gap-2 shadow-lg shadow-blue-200 active:scale-95 transition-all">
                <Camera size={22} /> 약 촬영하기
              </button>
            </div>
          )}
          <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
        </div>
      )}
    </div>
  )
}

// ─── 메인 앱 ─────────────────────────────────────────────────────────────────
export default function App() {
  const [userConditions, setUserConditions] = useState('일반 사용자')
  const [view, setView] = useState('home')
  const [previewUrl, setPreviewUrl] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [mfdsLoading, setMfdsLoading] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [mfdsInfo, setMfdsInfo] = useState(null)
  const [pillResults, setPillResults] = useState([])
  const [combinedAnalysis, setCombinedAnalysis] = useState(null)
  const [durWarnings, setDurWarnings] = useState([])
  const [analysisLogs, setAnalysisLogs] = useState([])
  const [currentUser, setCurrentUser] = useState(null)
  const [authReady, setAuthReady] = useState(false)
  const [symptom, setSymptom] = useState('')
  const [showAdminPin, setShowAdminPin] = useState(false)
  const [adminPin, setAdminPin] = useState('')
  const [showOnboarding, setShowOnboarding] = useState(!localStorage.getItem('igodae_onboarding_done'))
  const [logoTapCount, setLogoTapCount] = useState(0)
  const logoTapTimer = useRef(null)

  useEffect(() => {
    if (!auth) { setAuthReady(true); return }
    const unsub = onAuthStateChanged(auth, async (user) => {
      if (user) { setCurrentUser(user); setAuthReady(true) }
      else {
        try { const cred = await signInAnonymously(auth); setCurrentUser(cred.user) }
        catch (e) { console.warn('익명 로그인 실패:', e.message) }
        finally { setAuthReady(true) }
      }
    })
    return unsub
  }, [])

  useEffect(() => {
    if (!db || !currentUser || !authReady) return
    const q = query(LOGS_PATH(), orderBy('createdAt', 'desc'), limit(20))
    const unsub = onSnapshot(q, snap => {
      setAnalysisLogs(snap.docs.map(d => ({ id: d.id, ...d.data() })))
    }, err => console.warn('Firestore 구독 에러:', err.message))
    return unsub
  }, [currentUser, authReady])

  const saveToFirestore = useCallback(async (result) => {
    if (!db || !currentUser) return
    try {
      await addDoc(LOGS_PATH(), {
        userId: currentUser.uid,
        statusCode: result.statusCode,
        statusText: result.statusText,
        summary: result.summary,
        confidence: result.confidence,
        userConditions,
        createdAt: serverTimestamp(),
      })
    } catch (e) { console.warn('Firestore 저장 실패:', e.message) }
  }, [currentUser, userConditions])

  const processImage = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = e => {
        const dataUrl = e.target.result
        resolve({ base64: dataUrl.split(',')[1], previewUrl: dataUrl })
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }, [])

  const runAnalysis = useCallback(async (base64, mimeType = 'image/jpeg') => {
    setAnalyzing(true)
    setMfdsInfo(null)
    setAnalysisResult(null)
    setPillResults([])
    setDurWarnings([])

    let aiResult
    if (!GROQ_API_KEY) {
      await new Promise(r => setTimeout(r, 1500))
      aiResult = {
        pills: [
          { drugName: '', color: '하양', shape: '원형', form: '정제', imprint: '', size: '중', description: '흰색 원형 알약' },
          { drugName: '', color: '분홍', shape: '타원형', form: '정제', imprint: '', size: '소', description: '분홍색 타원형 알약' },
        ],
        totalCount: 2,
        symptomHint: 'API 키 미설정'
      }
    } else {
      try {
        const data = await safeFetchGroq({
          model: GROQ_VISION_MODEL,
          messages: [{ role: 'user', content: [
            { type: 'text', text: buildVisionPrompt(userConditions, symptom) },
            { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}` } }
          ]}],
          temperature: 0.1,
          max_tokens: 1000,
        })
        const raw = data.choices?.[0]?.message?.content || '{}'
        aiResult = JSON.parse(raw.replace(/```json|```/g, '').trim())
      } catch (e) {
        setAnalysisResult({ statusCode: 'unidentified', summary: '분석 실패', description: e.message, confidence: 0 })
        setAnalyzing(false)
        return
      }
    }

    setAnalyzing(false)

    if (!aiResult.pills || aiResult.pills.length === 0) {
      setAnalysisResult({ statusCode: 'unidentified', summary: '알약 미인식', description: '알약이 잘 보이도록 다시 촬영해주세요.', confidence: 0 })
      return
    }

    setMfdsLoading(true)
    try {
      const results = await Promise.all(
        aiResult.pills.map(pill => analyzeSinglePill(pill, aiResult.symptomHint))
      )
      setPillResults(results)
      setAnalysisResult(results[0])

      const combined = await analyzePillsCombined(results, symptom)
      setCombinedAnalysis(combined)

      // DUR 체크
      const userProfile = {
        isPregnant: userConditions.includes('임신') || userConditions.includes('임부'),
        isElderly:  userConditions.includes('노인') || userConditions.includes('고령'),
      }
      const dur = await runDurCheck(results, userProfile)
      setDurWarnings(dur)

      if (results[0]?.statusCode !== 'unidentified') {
        await saveToFirestore(results[0])
      }
    } catch (e) {
      console.warn('분석 실패:', e.message)
    } finally {
      setMfdsLoading(false)
    }
  }, [userConditions, symptom, saveToFirestore])

  const handleCameraCapture = useCallback(async (blob) => {
    setView('home')
    const { base64, previewUrl } = await processImage(blob)
    setPreviewUrl(previewUrl)
    setAnalysisResult(null)
    await runAnalysis(base64, 'image/jpeg')
  }, [processImage, runAnalysis])

  const handleGalleryUpload = useCallback(async (file) => {
    const { base64, previewUrl } = await processImage(file)
    setPreviewUrl(previewUrl)
    setAnalysisResult(null)
    await runAnalysis(base64, file.type || 'image/jpeg')
  }, [processImage, runAnalysis])

  const handleOnboardingComplete = () => {
    localStorage.setItem('igodae_onboarding_done', 'true')
    setShowOnboarding(false)
  }

  const handleLogoTap = () => {
    const next = logoTapCount + 1
    setLogoTapCount(next)
    if (logoTapTimer.current) clearTimeout(logoTapTimer.current)
    logoTapTimer.current = setTimeout(() => setLogoTapCount(0), 2000)
    if (next >= 5) { setLogoTapCount(0); setShowAdminPin(true); setAdminPin('') }
  }

  const handleAdminPin = (pin) => {
    if (pin === '1234') { setShowAdminPin(false); setView('admin') }
    else if (pin.length === 4) setAdminPin('')
  }

  const handleHistorySelect = (log) => {
    setAnalysisResult({ ...log })
    setMfdsInfo(null)
    setPreviewUrl(null)
    setView('home')
  }

  if (showOnboarding) return <OnboardingSlides onComplete={handleOnboardingComplete} />
  if (view === 'admin') return <AdminView logs={analysisLogs} onBack={() => setView('home')} />
  if (view === 'camera') return <CameraView onCapture={handleCameraCapture} onCancel={() => setView('home')} />
  if (view === 'chat' && analysisResult) return <ChatView result={analysisResult} mfdsInfo={mfdsInfo} userConditions={userConditions} onBack={() => setView('home')} />
  if (view === 'history') return <HistoryView logs={analysisLogs} onSelect={handleHistorySelect} onBack={() => setView('home')} />

  return (
    <>
      <HomeView
        userConditions={userConditions} analysisResult={analysisResult} mfdsInfo={mfdsInfo}
        pillResults={pillResults} combinedAnalysis={combinedAnalysis} durWarnings={durWarnings}
        analyzing={analyzing} mfdsLoading={mfdsLoading}
        onCameraCapture={() => setView('camera')} onGalleryUpload={handleGalleryUpload}
        onChat={() => setView('chat')} onHistory={() => setView('history')}
        onRetry={() => { setPreviewUrl(null); setAnalysisResult(null); setMfdsInfo(null); setPillResults([]); setCombinedAnalysis(null); setDurWarnings([]) }}
        previewUrl={previewUrl} logCount={analysisLogs.length}
        symptom={symptom} onSymptomChange={setSymptom} onLogoTap={handleLogoTap}
      />
      {showAdminPin && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center px-6">
          <div className="bg-white rounded-3xl p-6 w-full max-w-xs space-y-4">
            <p className="font-black text-slate-800 text-center text-lg">🔐 관리자 인증</p>
            <p className="text-slate-400 text-xs text-center">4자리 비밀번호를 입력하세요</p>
            <div className="flex justify-center gap-3">
              {[0,1,2,3].map(i => (
                <div key={i} className="w-10 h-10 rounded-2xl border-2 border-slate-200 flex items-center justify-center">
                  <span className="text-lg">{adminPin[i] ? '●' : ''}</span>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-3 gap-2">
              {['1','2','3','4','5','6','7','8','9','','0','⌫'].map((k, i) => (
                <button key={i} onClick={() => {
                  if (k === '⌫') setAdminPin(p => p.slice(0,-1))
                  else if (k && adminPin.length < 4) {
                    const next = adminPin + k
                    setAdminPin(next)
                    if (next.length === 4) handleAdminPin(next)
                  }
                }} className={`py-3 rounded-2xl font-bold text-lg ${k ? 'bg-slate-100 text-slate-800 active:bg-slate-200' : ''}`}>
                  {k}
                </button>
              ))}
            </div>
            <button onClick={() => setShowAdminPin(false)} className="w-full py-2 text-slate-400 text-sm">취소</button>
          </div>
        </div>
      )}
    </>
  )
}
