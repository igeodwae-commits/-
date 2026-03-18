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
const MFDS_API_KEY = import.meta.env.VITE_MFDS_API_KEY // 식약처 API 키
const GROQ_MODEL = 'llama-3.3-70b-versatile'
const GROQ_VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'
const GROQ_BASE = 'https://api.groq.com/openai/v1'

// 식약처 API 엔드포인트
const MFDS_DRUG_INFO_URL = 'https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList'
const MFDS_PILL_INFO_URL = 'https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03'

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

// ─── 식약처 API: 의약품 개요정보 조회 ────────────────────────────────────────
async function fetchMfdsInfo(drugName) {
  if (!MFDS_API_KEY || !drugName) return null
  try {
    const params = new URLSearchParams({
      serviceKey: MFDS_API_KEY,
      itemName: drugName,
      type: 'json',
      numOfRows: '3',
      pageNo: '1',
    })
    const res = await fetch(`${MFDS_DRUG_INFO_URL}?${params}`)
    const data = await res.json()
    const items = data?.body?.items
    if (!items || items.length === 0) return null
    const item = items[0]
    return {
      itemName: item.itemName,
      entpName: item.entpName,
      efcyQesitm: item.efcyQesitm,       // 효능
      useMethodQesitm: item.useMethodQesitm, // 복용법
      atpnWarnQesitm: item.atpnWarnQesitm,   // 주의사항 경고
      atpnQesitm: item.atpnQesitm,           // 주의사항
      intrcQesitm: item.intrcQesitm,         // 상호작용
      seQesitm: item.seQesitm,               // 부작용
      depositMethodQesitm: item.depositMethodQesitm, // 보관법
      source: '식품의약품안전처',
    }
  } catch (e) {
    console.warn('식약처 API 오류:', e.message)
    return null
  }
}

// ─── 식약처 API: 낱알식별 정보 조회 ──────────────────────────────────────────
async function fetchPillInfo(drugName) {
  if (!MFDS_API_KEY || !drugName) return null
  try {
    const params = new URLSearchParams({
      serviceKey: MFDS_API_KEY,
      itemName: drugName,
      type: 'json',
      numOfRows: '3',
      pageNo: '1',
    })
    const res = await fetch(`${MFDS_PILL_INFO_URL}?${params}`)
    const data = await res.json()
    const items = data?.body?.items
    if (!items || items.length === 0) return null
    const item = items[0]
    return {
      itemName: item.itemName,
      entpName: item.entpName,
      chart: item.chart,         // 약 모양
      colorClass1: item.colorClass1, // 색상
      formCodeName: item.formCodeName, // 제형
      itemImage: item.itemImage, // 이미지 URL
    }
  } catch (e) {
    console.warn('낱알식별 API 오류:', e.message)
    return null
  }
}

// ─── AI Vision 프롬프트 ───────────────────────────────────────────────────────
const buildVisionPrompt = (userConditions, symptom) => `
당신은 대한민국 공인 약사입니다.
사용자의 기저질환: ${userConditions}${symptom ? `\n사용자의 현재 증상: ${symptom}` : ''}

이미지에서 다음을 활용하여 약품을 식별하세요:
1. 약 포장지/박스의 제품명, 브랜드명 텍스트
2. 약 봉투에 인쇄된 약품명, 성분명
3. 알약/캡슐의 색상, 모양, 각인 문자
4. 포장 디자인 및 색상 조합

분석 가능한 약품 범위:
감기약, 해열진통제, 소화제, 항생제, 혈압약, 당뇨약, 피부약, 안약,
비타민/영양제, 수면제, 항히스타민제, 위장약, 변비약, 근육이완제,
한약제제, 외용제 등 모든 종류의 의약품

출력 규칙:
- summary: 제품명(성분명) 형식. 예: "게보린(이부프로펜+아세트아미노펜)" / "훼스탈골드(소화효소제)" / "베아제(복합소화제)"
- drugNameForSearch: 식약처 DB 검색용 약품명 (한글 제품명만, 브랜드명 우선. 예: "게보린" / "훼스탈" / "베아제")
- description: 쉬운 말로. "~로 추정됩니다" 절대 금지.
- warnings: 구체적으로
- dosageGuide: "하루 3번, 식후 30분에 1정씩" 처럼 구체적으로

JSON 형식으로만 응답하세요. 마크다운 없이 순수 JSON만:

{
  "status": "✅안전 | ⚠️주의 | ❌위험",
  "statusCode": "safe | caution | danger",
  "statusText": "한 줄 분류 설명",
  "oneLineSummary": "비전문가를 위한 한줄 요약",
  "summary": "의약품 공식 명칭 (성분명 포함)",
  "drugNameForSearch": "식약처 검색용 약품명 (한글)",
  "description": "약의 주요 효능 및 작용 기전 (2-3문장)",
  "warnings": "주의사항",
  "dosageGuide": "복용 방법",
  "interactions": ["병용 주의 약물/음식"],
  "alternatives": "대체약 또는 보완 방법",
  "activeIngredients": ["주요 성분명"],
  "drugType": "전문의약품 | 일반의약품 | 한약제제",
  "confidence": 0.0
}

약품을 식별할 수 없으면:
{"status": "❌위험", "statusCode": "unidentified", "summary": "약품 미인식", "description": "이미지에서 약품을 인식할 수 없습니다. 더 가까이서 촬영해주세요.", "confidence": 0}
`

const buildChatSystemPrompt = (analysisResult, mfdsInfo, userConditions) => `
당신은 '이거돼?' 앱의 AI 약사입니다.
사용자는 의학 지식이 없는 일반 환자입니다. 쉽고 친근한 말투로 설명하세요.

답변 규칙:
1. 약품명/성분명을 먼저 말하고 설명하세요
2. 전문 용어는 쉬운 말로 풀어서 설명하세요
3. 답변은 3-4문장으로 짧고 명확하게
4. 불확실하면 "약사나 의사에게 꼭 확인해보세요"라고 안내

현재 분석된 약품 정보 (AI 분석):
- 약품명: ${analysisResult?.summary || '미분석'}
- 안전도: ${analysisResult?.status || '-'}
- 사용자 기저질환: ${userConditions}

${mfdsInfo ? `식품의약품안전처 공식 정보:
- 효능: ${mfdsInfo.efcyQesitm || '-'}
- 복용법: ${mfdsInfo.useMethodQesitm || '-'}
- 주의사항: ${mfdsInfo.atpnQesitm || '-'}
- 부작용: ${mfdsInfo.seQesitm || '-'}` : ''}
`

// ─── 상태 매핑 ────────────────────────────────────────────────────────────────
const STATUS_MAP = {
  safe: { icon: CheckCircle, bg: 'bg-green-50', border: 'border-emerald-200', text: 'text-emerald-700', badge: 'bg-green-100 text-emerald-800', label: '복용 가능' },
  caution: { icon: AlertTriangle, bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', badge: 'bg-amber-100 text-amber-800', label: '주의 필요' },
  danger: { icon: XCircle, bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-100 text-red-800', label: '복용 위험' },
  unidentified: { icon: XCircle, bg: 'bg-slate-50', border: 'border-slate-200', text: 'text-slate-600', badge: 'bg-slate-100 text-slate-700', label: '인식 불가' },
}

// ─── 결과 카드 ────────────────────────────────────────────────────────────────
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
      {/* 추천 배너 */}
      <div className={`${rec.bg} px-5 py-4 flex items-center justify-center gap-2`}>
        <span className="text-2xl">{rec.emoji}</span>
        <p className="text-white font-black text-2xl tracking-tight">{rec.text}</p>
      </div>

      {/* 한줄 요약 */}
      {result.oneLineSummary && (
        <div className="px-5 py-3 bg-white border-b border-slate-100">
          <p className="text-slate-700 font-semibold text-sm text-center">{result.oneLineSummary}</p>
        </div>
      )}

      {/* 식약처 인증 뱃지 */}
      {mfdsInfo && (
        <div className="px-5 py-2 bg-blue-50 border-b border-blue-100 flex items-center gap-2">
          <Database size={13} className="text-[#0192F5]" />
          <p className="text-xs text-[#0192F5] font-semibold">식품의약품안전처 공식 정보 확인됨</p>
          <span className="ml-auto text-xs text-blue-300">{mfdsInfo.entpName}</span>
        </div>
      )}

      {/* 약품 헤더 */}
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

      {/* 핵심 정보 — 식약처 우선, 없으면 AI */}
      <div className="mx-4 mb-4 bg-white rounded-2xl divide-y divide-slate-100 shadow-sm">
        <InfoRow icon={Clock} label="복용 방법" value={mfdsInfo?.useMethodQesitm || result.dosageGuide} source={mfdsInfo?.useMethodQesitm ? '식약처' : 'AI'} />
        <InfoRow icon={Shield} label="주의사항" value={mfdsInfo?.atpnQesitm || result.warnings} source={mfdsInfo?.atpnQesitm ? '식약처' : 'AI'} />
        {(mfdsInfo?.seQesitm) && <InfoRow icon={AlertTriangle} label="부작용" value={mfdsInfo.seQesitm} source="식약처" />}
        {result.alternatives && <InfoRow icon={Zap} label="대체약" value={result.alternatives} source="AI" />}
      </div>

      {/* 성분 태그 */}
      {result.activeIngredients?.length > 0 && (
        <div className="px-4 pb-3 flex flex-wrap gap-1.5">
          {result.activeIngredients.map((ing, i) => (
            <span key={i} className="text-xs bg-white text-slate-600 px-2.5 py-1 rounded-full border border-slate-200 font-medium">{ing}</span>
          ))}
        </div>
      )}

      {/* 식약처 상세 정보 토글 */}
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
            </div>
          )}
        </div>
      )}

      {/* 병용 주의 */}
      {result.interactions?.length > 0 && (
        <div className="mx-4 mb-4 p-3 bg-amber-50 rounded-2xl border border-amber-100">
          <p className="text-xs font-bold text-amber-700 mb-1 flex items-center gap-1">
            <AlertTriangle size={12} /> 병용 주의
          </p>
          <p className="text-xs text-amber-600">{result.interactions.join(', ')}</p>
        </div>
      )}

      {/* 신뢰도 */}
      {result.confidence !== undefined && (
        <div className="mx-4 mb-4">
          {(() => {
            const pct = Math.round((result.confidence || 0) * 100)
            const color = pct >= 80 ? '#0192F5' : pct >= 60 ? '#f59e0b' : '#ef4444'
            const bg = pct >= 80 ? '#eff6ff' : pct >= 60 ? '#fffbeb' : '#fef2f2'
            const border = pct >= 80 ? '#bfdbfe' : pct >= 60 ? '#fde68a' : '#fecaca'
            return (
              <div className="rounded-2xl p-4 flex items-center gap-4" style={{ background: bg, border: `2px solid ${border}` }}>
                <div className="text-center shrink-0">
                  <p className="font-black text-4xl leading-none" style={{ color }}>{pct}%</p>
                  <p className="text-xs font-medium mt-1" style={{ color }}>인식 신뢰도</p>
                </div>
                <div className="flex-1">
                  <div className="h-3 bg-white rounded-full overflow-hidden">
                    <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
                  </div>
                  <p className="text-xs mt-2 font-medium" style={{ color }}>
                    {pct >= 80 ? '✅ 신뢰할 수 있는 결과예요' : pct >= 60 ? '⚠️ 참고용으로만 활용하세요' : '❌ 다시 촬영해보세요'}
                  </p>
                </div>
              </div>
            )
          })()}
        </div>
      )}

      {/* AI 상담 버튼 */}
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
        <p className="text-sm text-slate-700 leading-snug">{value}</p>
      </div>
    </div>
  )
}

function MfdsRow({ label, value, highlight }) {
  if (!value) return null
  return (
    <div className={`p-3 ${highlight ? 'bg-red-50' : ''}`}>
      <p className={`text-xs font-bold mb-1 ${highlight ? 'text-red-600' : 'text-slate-400'}`}>{label}</p>
      <p className="text-xs text-slate-600 leading-relaxed">{value}</p>
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
  const safeCount = logs.filter(l => l.statusCode === 'safe').length
  const cautionCount = logs.filter(l => l.statusCode === 'caution').length
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
function HomeView({ userConditions, analysisResult, mfdsInfo, analyzing, mfdsLoading, onCameraCapture, onGalleryUpload, onChat, onHistory, onRetry, previewUrl, logCount, symptom, onSymptomChange, onLogoTap }) {
  const fileInputRef = useRef(null)
  const [step, setStep] = useState(previewUrl || analysisResult ? 2 : 1)

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
        {!analyzing && !mfdsLoading && analysisResult && (
          <ResultCard result={analysisResult} mfdsInfo={mfdsInfo} onChat={onChat} onRetry={() => { onRetry(); setStep(2) }} />
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
      <div className="fixed bottom-0 left-1/2 -translate-x-1/2 w-full max-w-[480px] px-5 pb-8 pt-4 bg-gradient-to-t from-white via-white to-transparent">
        <div className="flex gap-3">
          <button onClick={() => fileInputRef.current?.click()} className="flex-1 py-4 rounded-2xl bg-slate-100 text-slate-600 font-bold flex items-center justify-center gap-2">
            <ImagePlus size={20} /> 갤러리
          </button>
          <button onClick={onCameraCapture} className="flex-[2] py-4 rounded-2xl bg-gradient-to-r from-[#0192F5] to-[#40BEFD] text-white font-bold text-base flex items-center justify-center gap-2 shadow-lg shadow-blue-200 active:scale-95 transition-all">
            <Camera size={22} /> 약 촬영하기
          </button>
        </div>
        <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
      </div>
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
  const [analysisLogs, setAnalysisLogs] = useState([])
  const [currentUser, setCurrentUser] = useState(null)
  const [authReady, setAuthReady] = useState(false)
  const [symptom, setSymptom] = useState('')
  const [showAdminPin, setShowAdminPin] = useState(false)
  const [adminPin, setAdminPin] = useState('')
  const [logoTapCount, setLogoTapCount] = useState(0)
  const logoTapTimer = useRef(null)

  // Firebase 익명 인증
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

  // Firestore 구독
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

  // AI 분석 + 식약처 API 연동
  const runAnalysis = useCallback(async (base64, mimeType = 'image/jpeg') => {
    // 1단계: Groq Vision AI 분석
    setAnalyzing(true)
    setMfdsInfo(null)

    let aiResult
    if (!GROQ_API_KEY) {
      await new Promise(r => setTimeout(r, 1500))
      aiResult = {
        status: '⚠️주의', statusCode: 'caution', statusText: '데모 모드',
        oneLineSummary: 'API 키를 설정하면 실제 분석이 가능합니다',
        summary: 'API 키 미설정', drugNameForSearch: '',
        description: '.env 파일에 VITE_GROQ_API_KEY를 설정해주세요.',
        warnings: '이것은 데모 결과입니다.', dosageGuide: '하루 3번, 식후 30분',
        interactions: [], alternatives: '',
        activeIngredients: [], drugType: '일반의약품', confidence: 0,
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
          max_tokens: 1200,
        })
        const raw = data.choices?.[0]?.message?.content || '{}'
        aiResult = JSON.parse(raw.replace(/```json|```/g, '').trim())
      } catch (e) {
        aiResult = { status: '❌위험', statusCode: 'unidentified', summary: '분석 실패', description: e.message, confidence: 0 }
      }
    }

    setAnalysisResult(aiResult)
    setAnalyzing(false)

    // 2단계: 식약처 API 조회 (AI가 약품명 추출한 경우)
    if (aiResult.statusCode !== 'unidentified' && (aiResult.drugNameForSearch || aiResult.summary)) {
      setMfdsLoading(true)
      const searchName = aiResult.drugNameForSearch || aiResult.summary?.split('(')[0]?.trim()
      try {
        const [drugInfo, pillInfo] = await Promise.all([
          fetchMfdsInfo(searchName),
          fetchPillInfo(searchName),
        ])
        if (drugInfo) setMfdsInfo({ ...drugInfo, ...pillInfo })
      } catch (e) {
        console.warn('식약처 API 조회 실패:', e.message)
      } finally {
        setMfdsLoading(false)
      }
    }

    // Firestore 저장
    if (aiResult.statusCode !== 'unidentified') {
      await saveToFirestore(aiResult)
    }

    return aiResult
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

  if (view === 'admin') return <AdminView logs={analysisLogs} onBack={() => setView('home')} />
  if (view === 'camera') return <CameraView onCapture={handleCameraCapture} onCancel={() => setView('home')} />
  if (view === 'chat' && analysisResult) return <ChatView result={analysisResult} mfdsInfo={mfdsInfo} userConditions={userConditions} onBack={() => setView('home')} />
  if (view === 'history') return <HistoryView logs={analysisLogs} onSelect={handleHistorySelect} onBack={() => setView('home')} />

  return (
    <>
      <HomeView
        userConditions={userConditions} analysisResult={analysisResult} mfdsInfo={mfdsInfo}
        analyzing={analyzing} mfdsLoading={mfdsLoading}
        onCameraCapture={() => setView('camera')} onGalleryUpload={handleGalleryUpload}
        onChat={() => setView('chat')} onHistory={() => setView('history')}
        onRetry={() => { setPreviewUrl(null); setAnalysisResult(null); setMfdsInfo(null) }}
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
