/**
 * 이거돼? — AI 기반 약물 판독 & 복약 가이드
 * 
 * Stack: React + Vite + Tailwind CSS + Firebase + Gemini 2.5 Flash
 * Author: 이거돼 Team
 * 
 * 환경변수 설정 필요 (.env 파일 참조)
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Camera, ImagePlus, Send, ChevronRight, Clock, AlertTriangle,
  CheckCircle, XCircle, Pill, MessageCircle, History, Home,
  Loader2, Sparkles, RefreshCw, ChevronLeft, Info, Star,
  Shield, Zap, X
} from 'lucide-react'

// ─── Firebase SDK (동적 임포트로 번들 최적화) ─────────────────────────────────
import { initializeApp, getApps } from 'firebase/app'
import {
  getAuth, signInAnonymously, onAuthStateChanged
} from 'firebase/auth'
import {
  getFirestore, collection, addDoc, query, orderBy,
  limit, onSnapshot, serverTimestamp, doc, getDoc
} from 'firebase/firestore'

// ─── Firebase 설정 ────────────────────────────────────────────────────────────
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
const GROQ_MODEL = 'llama-3.3-70b-versatile'
const GROQ_VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'
const GROQ_BASE = 'https://api.groq.com/openai/v1'

// Firebase 초기화 (중복 방지)
let app, auth, db
try {
  app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig)
  auth = getAuth(app)
  db = getFirestore(app)
} catch (e) {
  console.warn('Firebase 초기화 실패 (환경변수 미설정):', e.message)
}

// ─── Firestore 경로 헬퍼 ──────────────────────────────────────────────────────
const LOGS_PATH = () =>
  collection(db, `artifacts/${APP_ID}/public/data/analysis_logs`)

// ─── OpenAI API 유틸 (지수 백오프 재시도) ────────────────────────────────────
async function safeFetchGroq(body, retries = 3, delay = 1000) {
  if (!GROQ_API_KEY) throw new Error('VITE_GROQ_API_KEY 환경변수가 설정되지 않았습니다.')

  for (let i = 0; i < retries; i++) {
    try {
      const res = await fetch(
        `${GROQ_BASE}/chat/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${GROQ_API_KEY}`,
          },
          body: JSON.stringify(body),
        }
      )

      if (res.status === 401) {
        throw new Error('API 키가 유효하지 않습니다. .env 파일을 확인하세요.')
      }

      if (res.status === 429 || res.status >= 500) {
        if (i < retries - 1) {
          await new Promise(r => setTimeout(r, delay * Math.pow(2, i)))
          continue
        }
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.error?.message || `HTTP ${res.status}`)
      }

      return await res.json()
    } catch (e) {
      if (i === retries - 1) throw e
      await new Promise(r => setTimeout(r, delay * Math.pow(2, i)))
    }
  }
}

// ─── 약물 분석 프롬프트 ───────────────────────────────────────────────────────
const buildVisionPrompt = (userConditions) => `
당신은 대한민국 공인 약사이자 위장내과 전문의입니다.
사용자의 기저질환: ${userConditions}

## 이미지 분석 지침
이미지에서 다음을 모두 활용하여 약품을 식별하세요:
1. 약 포장지/박스의 제품명, 브랜드명 텍스트
2. 약 봉투에 인쇄된 약품명, 성분명
3. 알약/캡슐의 색상, 모양, 각인 문자
4. 포장 디자인 및 색상 조합
5. 한글/영문 혼합 표기 모두 인식

## 한국 주요 약품 참고
- 타이레놀(아세트아미노펜): 해열진통, 위 자극 적음
- 부루펜/이부프로펜: 소염진통, 위 자극 강함 주의
- 게보린: 복합진통제, 위염 주의
- 판콜/판피린: 종합감기약
- 훼스탈/베아제: 소화효소제, 위염에 안전
- 겔포스/개비스콘: 위산중화제, 위염에 유익
- 잔탁/오메프라졸: 위산억제제
- 비타민C/종합비타민: 일반적으로 안전

## 출력 말투 규칙 (매우 중요)
- summary: 반드시 제품명(성분명) 형식으로. 예: "타이레놀(아세트아미노펜)" / "판콜에이(복합감기약)"
- description: 환자가 이해할 수 있는 쉬운 말로. "~로 추정됩니다" 절대 금지. 약 이름 먼저, 효능 설명 후.
  좋은 예: "타이레놀은 열을 내리고 통증을 줄여주는 해열진통제예요. 위에 자극이 적어서 위염 환자도 비교적 안전하게 드실 수 있어요."
  나쁜 예: "이 약 조합을 보니 감기약인 것으로 추정됩니다."
- warnings: 쉬운 말로. "식후 30분에 드세요", "공복에는 피하세요" 처럼 구체적으로
- dosageGuide: "하루 3번, 식후 30분에 1정씩" 처럼 구체적으로

JSON 형식으로만 응답하세요. 마크다운 없이 순수 JSON만:

{
  "status": "✅안전 | ⚠️주의 | ❌위험",
  "statusCode": "safe | caution | danger",
  "statusText": "한 줄 분류 설명",
  "oneLineSummary": "비전문가를 위한 한줄 요약. 예: 감기에 도움이 돼요! / 위에 자극이 강해서 주의가 필요해요! / 위염 환자에게 위험할 수 있어요!",
  "summary": "의약품 공식 명칭 (성분명 포함)",
  "description": "약의 주요 효능 및 작용 기전 (2-3문장)",
  "warnings": "위염 환자 특화 주의사항",
  "dosageGuide": "복용 방법 (식전/식후/취침전, 용량, 빈도)",
  "interactions": ["병용 주의 약물/음식"],
  "alternatives": "대체약 또는 보완 방법",
  "activeIngredients": ["주요 성분명"],
  "drugType": "전문의약품 | 일반의약품 | 한약제제",
  "confidence": 인식 신뢰도 (0.0~1.0)
}

약품을 식별할 수 없으면:
{"status": "❌위험", "statusCode": "unidentified", "summary": "약품 미인식", "description": "이미지에서 약품을 인식할 수 없습니다. 약 이름이 보이도록 더 가까이서 촬영해주세요.", "confidence": 0}
`

const buildChatSystemPrompt = (analysisResult, userConditions) => `
당신은 '이거돼?' 앱의 AI 약사입니다.
사용자는 의학 지식이 없는 일반 환자입니다. 쉽고 친근한 말투로 설명하세요.

## 답변 규칙
1. 반드시 약품명/성분명을 먼저 말하고 설명하세요
   좋은 예: "타이레놀(아세트아미노펜)은 해열진통제예요. 위에 자극이 적어서 위염 환자도 비교적 안전하게 드실 수 있어요."
   나쁜 예: "이 약 조합을 보니 감기약인 것으로 추정됩니다."
2. 전문 용어는 반드시 쉬운 말로 풀어서 설명하세요
   좋은 예: "아세트아미노펜(타이레놀 성분)이 포함되어 있어요"
   나쁜 예: "NSAIDs 계열 약물로 COX-2를 억제합니다"
3. 답변은 3-4문장으로 짧고 명확하게
4. 위험하거나 확실하지 않으면 "약사나 의사에게 꼭 확인해보세요"라고 안내

현재 분석된 약품 정보:
- 약품명: \${analysisResult?.summary || '미분석'}
- 안전도: \${analysisResult?.status || '-'}
- 사용자 기저질환: \${userConditions}
`

// ─── 상태 색상 / 아이콘 매핑 ──────────────────────────────────────────────────
const STATUS_MAP = {
  safe: {
    icon: CheckCircle,
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    text: 'text-emerald-700',
    badge: 'bg-emerald-100 text-emerald-800',
    bar: 'bg-emerald-500',
    label: '복용 가능',
  },
  caution: {
    icon: AlertTriangle,
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    text: 'text-amber-700',
    badge: 'bg-amber-100 text-amber-800',
    bar: 'bg-amber-500',
    label: '주의 필요',
  },
  danger: {
    icon: XCircle,
    bg: 'bg-red-50',
    border: 'border-red-200',
    text: 'text-red-700',
    badge: 'bg-red-100 text-red-800',
    bar: 'bg-red-500',
    label: '복용 위험',
  },
  unidentified: {
    icon: XCircle,
    bg: 'bg-slate-50',
    border: 'border-slate-200',
    text: 'text-slate-600',
    badge: 'bg-slate-100 text-slate-700',
    bar: 'bg-slate-400',
    label: '인식 불가',
  },
}

// ─── 위험도 게이지 컴포넌트 ───────────────────────────────────────────────────


// ─── 분석 결과 카드 ───────────────────────────────────────────────────────────
function ResultCard({ result, onChat, onRetry }) {
  const statusCode = result?.statusCode || 'unidentified'
  const s = STATUS_MAP[statusCode] || STATUS_MAP.unidentified
  const StatusIcon = s.icon
  const [expanded, setExpanded] = useState(false)

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
        <button
          onClick={onRetry}
          className="w-full py-3 rounded-2xl bg-slate-800 text-white font-semibold flex items-center justify-center gap-2 active:scale-95 transition-transform"
        >
          <RefreshCw size={16} /> 다시 촬영하기
        </button>
      </div>
    )
  }

  // 추천 배너 설정
  const RECOMMEND_MAP = {
    safe:    { text: '추천합니다!',        bg: 'bg-emerald-500', emoji: '✅' },
    caution: { text: '주의가 필요해요!',   bg: 'bg-amber-500',   emoji: '⚠️' },
    danger:  { text: '추천하지 않습니다!', bg: 'bg-red-500',     emoji: '❌' },
  }
  const rec = RECOMMEND_MAP[statusCode] || RECOMMEND_MAP.caution

  return (
    <div className={`rounded-3xl border-2 ${s.border} ${s.bg} overflow-hidden animate-slide-up`}>

      {/* 크고 굵은 추천 배너 */}
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

      {/* 헤더 */}
      <div className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <StatusIcon className={`${s.text} shrink-0`} size={24} />
            <div className="min-w-0">
              <p className={`font-black text-lg leading-tight ${s.text} truncate`}>
                {result.summary}
              </p>
              <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${s.badge}`}>
                {result.statusText || s.label}
              </span>
            </div>
          </div>
        </div>

        <p className="text-sm text-slate-600 leading-relaxed">{result.description}</p>


      </div>

      {/* 핵심 정보 */}
      <div className="mx-4 mb-4 bg-white rounded-2xl divide-y divide-slate-100 shadow-sm">
        <InfoRow icon={Clock} label="복용 방법" value={result.dosageGuide} />
        <InfoRow icon={Shield} label="주의사항" value={result.warnings} />
        {result.alternatives && (
          <InfoRow icon={Zap} label="대체약" value={result.alternatives} />
        )}
      </div>

      {/* 성분 태그 */}
      {result.activeIngredients?.length > 0 && (
        <div className="px-4 pb-3 flex flex-wrap gap-1.5">
          {result.activeIngredients.map((ing, i) => (
            <span key={i} className="text-xs bg-white text-slate-600 px-2.5 py-1 rounded-full border border-slate-200 font-medium">
              {ing}
            </span>
          ))}
        </div>
      )}

      {/* 상호작용 경고 */}
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
        <div className="px-4 pb-2">
          <p className="text-xs text-slate-400 text-right">
            인식 신뢰도: {Math.round(result.confidence * 100)}%
          </p>
        </div>
      )}

      {/* AI 상담 버튼 */}
      <div className="p-4 pt-0">
        <button
          onClick={onChat}
          className="w-full py-3.5 rounded-2xl bg-gradient-to-r from-teal-600 to-cyan-600 text-white font-bold flex items-center justify-center gap-2 shadow-md active:scale-95 transition-all"
        >
          <MessageCircle size={18} /> AI 약사에게 더 물어보기
        </button>
      </div>
    </div>
  )
}

function InfoRow({ icon: Icon, label, value }) {
  if (!value) return null
  return (
    <div className="flex gap-3 p-3">
      <div className="w-7 h-7 rounded-xl bg-teal-50 flex items-center justify-center shrink-0 mt-0.5">
        <Icon size={14} className="text-teal-600" />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs font-bold text-slate-400 uppercase tracking-wide">{label}</p>
        <p className="text-sm text-slate-700 leading-snug mt-0.5">{value}</p>
      </div>
    </div>
  )
}

// ─── 분석 로딩 스켈레톤 ───────────────────────────────────────────────────────
function AnalyzingSkeleton() {
  return (
    <div className="rounded-3xl border-2 border-teal-100 bg-teal-50 p-6 space-y-4 animate-pulse">
      <div className="flex items-center gap-3">
        <Loader2 size={28} className="text-teal-500 animate-spin" />
        <div className="flex-1 space-y-2">
          <div className="h-5 shimmer rounded-lg w-3/4" />
          <div className="h-3 shimmer rounded w-1/2" />
        </div>
      </div>
      <div className="space-y-2">
        <div className="h-3 shimmer rounded w-full" />
        <div className="h-3 shimmer rounded w-5/6" />
        <div className="h-3 shimmer rounded w-4/6" />
      </div>
      <div className="h-10 shimmer rounded-2xl" />
      <p className="text-center text-sm text-teal-600 font-medium animate-pulse">
        🔍 AI가 약품을 분석하고 있어요...
      </p>
    </div>
  )
}

// ─── 채팅 뷰 ─────────────────────────────────────────────────────────────────
function ChatView({ result, userConditions, onBack }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: `안녕하세요! 👋 **${result?.summary || '분석된 약품'}**에 대해 무엇이든 물어보세요.\n\n복용 방법, 부작용, 다른 약과의 상호작용 등을 도와드릴 수 있어요.`,
      ts: Date.now(),
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg = { role: 'user', content: text, ts: Date.now() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const history = messages
        .slice(1)
        .map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content }))

      const data = await safeFetchGroq({
        model: GROQ_MODEL,
        messages: [
          { role: 'system', content: buildChatSystemPrompt(result, userConditions) },
          ...history,
          { role: 'user', content: text }
        ],
        temperature: 0.7,
        max_tokens: 600,
      })

      const reply = data.choices?.[0]?.message?.content || '죄송합니다, 응답을 가져오지 못했어요.'
      setMessages(prev => [...prev, { role: 'assistant', content: reply, ts: Date.now() }])
    } catch (e) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `⚠️ 오류가 발생했어요: ${e.message}`,
        ts: Date.now(),
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex flex-col h-[100dvh]">
      {/* 채팅 헤더 */}
      <div className="glass sticky top-0 z-10 px-4 pt-4 pb-3 border-b border-slate-100 flex items-center gap-3">
        <button
          onClick={onBack}
          className="w-9 h-9 rounded-2xl bg-slate-100 flex items-center justify-center active:bg-slate-200 transition-colors"
        >
          <ChevronLeft size={20} className="text-slate-600" />
        </button>
        <div className="flex-1 min-w-0">
          <p className="font-bold text-slate-800 text-sm truncate">AI 약사 상담</p>
          <p className="text-xs text-slate-400 truncate">{result?.summary}</p>
        </div>
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center">
          <Sparkles size={15} className="text-white" />
        </div>
      </div>

      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3 scrollbar-hide">
        {messages.map((msg, i) => (
          <div key={i} className={`flex bubble-in ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'assistant' && (
              <div className="w-7 h-7 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center mr-2 mt-1 shrink-0">
                <Sparkles size={13} className="text-white" />
              </div>
            )}
            <div
              className={`max-w-[78%] px-4 py-3 rounded-3xl text-sm leading-relaxed whitespace-pre-wrap ${
                msg.role === 'user'
                  ? 'bg-gradient-to-br from-teal-600 to-cyan-600 text-white rounded-br-lg'
                  : 'bg-slate-100 text-slate-800 rounded-bl-lg'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex items-start gap-2">
            <div className="w-7 h-7 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-500 flex items-center justify-center shrink-0">
              <Sparkles size={13} className="text-white" />
            </div>
            <div className="bg-slate-100 px-4 py-3 rounded-3xl rounded-bl-lg flex items-center gap-1.5">
              {[0, 1, 2].map(i => (
                <span
                  key={i}
                  className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* 빠른 질문 */}
      {messages.length <= 2 && (
        <div className="px-4 pb-2">
          <div className="flex gap-2 overflow-x-auto scrollbar-hide pb-1">
            {[
              '식전에 먹어도 돼요?',
              '어떤 효과가 있나요?',
              '다른 약과 같이 먹어도 되나요?',
              '부작용이 뭔가요?',
            ].map(q => (
              <button
                key={q}
                onClick={() => { setInput(q); inputRef.current?.focus() }}
                className="shrink-0 text-xs bg-teal-50 text-teal-700 px-3 py-2 rounded-2xl border border-teal-100 font-medium active:bg-teal-100 transition-colors whitespace-nowrap"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 입력창 */}
      <div className="px-4 pb-safe-bottom pb-4 pt-2 border-t border-slate-100 bg-white">
        <div className="flex items-end gap-2 bg-slate-100 rounded-3xl px-4 py-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="약에 대해 질문하세요..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-slate-800 placeholder-slate-400 resize-none outline-none max-h-24 scrollbar-hide py-1.5"
            style={{ minHeight: '24px' }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            className="w-9 h-9 rounded-2xl bg-gradient-to-br from-teal-600 to-cyan-600 flex items-center justify-center shrink-0 disabled:opacity-30 active:scale-95 transition-all mb-0.5"
          >
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
        <Header title="분석 기록" onBack={onBack} />
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
      <Header title="분석 기록" onBack={onBack} />
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3 scrollbar-hide">
        {logs.map((log, i) => {
          const s = STATUS_MAP[log.statusCode] || STATUS_MAP.unidentified
          const StatusIcon = s.icon
          return (
            <button
              key={log.id || i}
              onClick={() => onSelect(log)}
              className={`w-full text-left p-4 rounded-2xl border ${s.border} ${s.bg} flex items-center gap-3 active:scale-98 transition-all`}
            >
              <StatusIcon className={`${s.text} shrink-0`} size={22} />
              <div className="flex-1 min-w-0">
                <p className="font-bold text-slate-800 truncate text-sm">{log.summary || '약품명 없음'}</p>
                <p className="text-xs text-slate-400 mt-0.5 truncate">{log.statusText || s.label}</p>
                <p className="text-xs text-slate-300 mt-0.5">
                  {log.createdAt?.toDate?.()?.toLocaleDateString('ko-KR') || '날짜 없음'}
                </p>
              </div>
              <ChevronRight size={16} className="text-slate-300 shrink-0" />
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ─── 간단한 헤더 컴포넌트 ─────────────────────────────────────────────────────
function Header({ title, onBack, action }) {
  return (
    <div className="glass sticky top-0 z-10 px-4 pt-4 pb-3 border-b border-slate-100 flex items-center gap-3">
      <button
        onClick={onBack}
        className="w-9 h-9 rounded-2xl bg-slate-100 flex items-center justify-center active:bg-slate-200 transition-colors"
      >
        <ChevronLeft size={20} className="text-slate-600" />
      </button>
      <p className="flex-1 font-bold text-slate-800">{title}</p>
      {action}
    </div>
  )
}

// ─── 온보딩 / 설정 뷰 (기저질환 입력) ────────────────────────────────────────
function OnboardingView({ onComplete }) {
  const CONDITIONS = [
    { id: 'gastritis', label: '위염', emoji: '🔥', desc: '위 점막 염증' },
    { id: 'gerd', label: '역류성 식도염', emoji: '⚡', desc: '위산 역류' },
    { id: 'ulcer', label: '위궤양', emoji: '🫥', desc: '위 점막 손상' },
    { id: 'ibs', label: '과민성 대장증후군', emoji: '🌀', desc: '장 과민 반응' },
    { id: 'hypertension', label: '고혈압', emoji: '❤️', desc: '혈압 관리' },
    { id: 'diabetes', label: '당뇨', emoji: '💉', desc: '혈당 관리' },
    { id: 'none', label: '해당 없음', emoji: '✨', desc: '기저질환 없음' },
  ]

  const [selected, setSelected] = useState(new Set(['gastritis']))

  const toggle = (id) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (id === 'none') return new Set(['none'])
      next.delete('none')
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next.size === 0 ? new Set(['none']) : next
    })
  }

  const confirm = () => {
    const labels = CONDITIONS
      .filter(c => selected.has(c.id))
      .map(c => c.label)
      .join(', ')
    onComplete(labels || '해당 없음')
  }

  return (
    <div className="flex flex-col h-[100dvh] px-6 pt-16 pb-8 bg-gradient-to-b from-teal-50 to-white">
      <div className="flex-1 space-y-8">
        {/* 앱 로고 */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-gradient-to-br from-teal-500 to-cyan-500 shadow-xl shadow-teal-200">
            <Pill size={36} className="text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-black text-slate-800">이거돼?</h1>
            <p className="text-slate-500 text-sm mt-1">AI 기반 약물 판독 & 복약 가이드</p>
          </div>
        </div>

        {/* 기저질환 선택 */}
        <div className="space-y-3">
          <div>
            <p className="font-bold text-slate-800">기저질환을 선택해주세요</p>
            <p className="text-xs text-slate-400 mt-1">선택한 정보로 맞춤 약물 분석을 제공해요</p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {CONDITIONS.map(c => (
              <button
                key={c.id}
                onClick={() => toggle(c.id)}
                className={`p-3 rounded-2xl border-2 text-left transition-all active:scale-95 ${
                  selected.has(c.id)
                    ? 'border-teal-400 bg-teal-50'
                    : 'border-slate-100 bg-white'
                }`}
              >
                <span className="text-xl">{c.emoji}</span>
                <p className={`text-sm font-bold mt-1 ${selected.has(c.id) ? 'text-teal-700' : 'text-slate-700'}`}>
                  {c.label}
                </p>
                <p className="text-xs text-slate-400">{c.desc}</p>
              </button>
            ))}
          </div>
        </div>
      </div>

      <button
        onClick={confirm}
        className="w-full py-4 rounded-3xl bg-gradient-to-r from-teal-600 to-cyan-600 text-white font-bold text-base shadow-lg shadow-teal-200 active:scale-95 transition-all"
      >
        시작하기 →
      </button>
      <p className="text-center text-xs text-slate-300 mt-3">개인정보는 기기에만 저장됩니다</p>
    </div>
  )
}

// ─── 메인 홈 뷰 ───────────────────────────────────────────────────────────────
function HomeView({
  userConditions, analysisResult, analyzing,
  onCameraCapture, onGalleryUpload, onChat, onHistory, onRetry,
  previewUrl, logCount, currentUser,
}) {
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) onGalleryUpload(file)
    e.target.value = ''
  }

  return (
    <div className="flex flex-col h-[100dvh]">
      {/* 상단 헤더 */}
      <div className="px-5 pt-6 pb-4 bg-gradient-to-b from-teal-600 to-teal-500">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="flex items-center gap-1.5 mb-0.5">
              <Pill size={18} className="text-white/80" />
              <span className="text-white/80 text-xs font-medium">이거돼?</span>
            </div>
            <h2 className="text-white font-black text-xl">약 사진을 찍어보세요</h2>
          </div>
          <button onClick={onHistory} className="relative">
            <div className="w-10 h-10 rounded-2xl bg-white/20 flex items-center justify-center active:bg-white/30 transition-colors">
              <History size={20} className="text-white" />
            </div>
            {logCount > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-400 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                {Math.min(logCount, 9)}
              </span>
            )}
          </button>
        </div>


      </div>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5 scrollbar-hide pb-28">

        {/* 촬영 미리보기 */}
        {previewUrl && (
          <div className="relative rounded-3xl overflow-hidden bg-slate-100 aspect-video animate-fade-in shadow-md">
            <img src={previewUrl} alt="약품 사진" className="w-full h-full object-cover" />
            {!analysisResult && !analyzing && (
              <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent flex items-end p-4">
                <p className="text-white text-sm font-medium">분석 대기중...</p>
              </div>
            )}
          </div>
        )}

        {/* 분석 중 스켈레톤 */}
        {analyzing && <AnalyzingSkeleton />}

        {/* 분석 결과 */}
        {!analyzing && analysisResult && (
          <ResultCard
            result={analysisResult}
            onChat={onChat}
            onRetry={onRetry}
          />
        )}

        {/* 빈 상태 가이드 */}
        {!previewUrl && !analyzing && !analysisResult && (
          <div className="text-center py-8 space-y-4">
            <div className="relative inline-flex">
              <div className="w-24 h-24 rounded-full bg-teal-50 flex items-center justify-center ring-pulse">
                <Camera size={40} className="text-teal-400" />
              </div>
            </div>
            <div className="space-y-1.5">
              <p className="font-bold text-slate-700">약 사진을 찍어주세요</p>
              <p className="text-sm text-slate-400 leading-relaxed">
                약 봉투, 약통, 낱알 모두 가능해요<br />
                AI가 즉시 성분과 복용법을 알려드려요
              </p>
            </div>

            {/* 사용 팁 */}
            <div className="mt-6 space-y-2 text-left">
              {[
                { emoji: '💊', text: '약 이름이 보이게 찍으면 더 정확해요' },
                { emoji: '📋', text: '처방전이나 약 봉투도 인식 가능해요' },
                { emoji: '🔍', text: '흐리지 않게 가까이서 촬영해주세요' },
              ].map((tip, i) => (
                <div key={i} className="flex items-center gap-2.5 bg-slate-50 rounded-2xl px-4 py-2.5">
                  <span className="text-lg">{tip.emoji}</span>
                  <p className="text-xs text-slate-500">{tip.text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 하단 촬영 버튼 */}
      <div className="fixed bottom-0 left-1/2 -translate-x-1/2 w-full max-w-[480px] px-5 pb-8 pt-4 bg-gradient-to-t from-white via-white to-transparent">
        <div className="flex gap-3">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-1 py-4 rounded-2xl bg-slate-100 text-slate-600 font-bold flex items-center justify-center gap-2 active:bg-slate-200 transition-colors"
          >
            <ImagePlus size={20} /> 갤러리
          </button>
          <button
            onClick={onCameraCapture}
            className="flex-[2] py-4 rounded-2xl bg-gradient-to-r from-teal-600 to-cyan-600 text-white font-bold text-base flex items-center justify-center gap-2 shadow-lg shadow-teal-200 active:scale-95 transition-all"
          >
            <Camera size={22} /> 약 촬영하기
          </button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />
      </div>
    </div>
  )
}

// ─── 카메라 캡처 뷰 ───────────────────────────────────────────────────────────
function CameraView({ onCapture, onCancel }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let mounted = true
    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }
        })
        if (!mounted) { stream.getTracks().forEach(t => t.stop()); return }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
          setReady(true)
        }
      } catch (e) {
        setError('카메라 접근 권한이 필요합니다. 설정에서 허용해주세요.')
      }
    }
    start()
    return () => {
      mounted = false
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [])

  const shoot = () => {
    if (!videoRef.current || !ready) return
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0)
    canvas.toBlob(blob => {
      streamRef.current?.getTracks().forEach(t => t.stop())
      onCapture(blob)
    }, 'image/jpeg', 0.92)
  }

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      <div className="relative flex-1 overflow-hidden">
        <video ref={videoRef} playsInline muted className="absolute inset-0 w-full h-full object-cover" />

        {/* 오버레이 가이드 */}
        {ready && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="w-72 h-48 rounded-3xl border-2 border-white/60 shadow-2xl">
              <div className="absolute -top-5 left-1/2 -translate-x-1/2 bg-black/60 text-white text-xs px-3 py-1 rounded-full whitespace-nowrap">
                약품이 이 안에 들어오게 맞춰주세요
              </div>
            </div>
          </div>
        )}

        {!ready && !error && (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 size={40} className="text-white animate-spin" />
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center px-8 text-center space-y-4">
            <XCircle size={48} className="text-red-400" />
            <p className="text-white text-sm">{error}</p>
            <button onClick={onCancel} className="px-6 py-2 bg-white text-slate-800 rounded-full font-semibold">
              돌아가기
            </button>
          </div>
        )}

        {/* 닫기 */}
        <button
          onClick={onCancel}
          className="absolute top-4 left-4 w-10 h-10 rounded-full bg-black/50 flex items-center justify-center"
        >
          <X size={20} className="text-white" />
        </button>
      </div>

      {/* 촬영 버튼 */}
      {ready && (
        <div className="bg-black pb-12 pt-6 flex items-center justify-center">
          <button
            onClick={shoot}
            className="w-20 h-20 rounded-full border-4 border-white bg-white/20 flex items-center justify-center active:scale-90 transition-transform"
          >
            <div className="w-14 h-14 rounded-full bg-white" />
          </button>
        </div>
      )}
    </div>
  )
}

// ─── 메인 앱 컴포넌트 ─────────────────────────────────────────────────────────
export default function App() {
  // 사용자 설정
  const [userConditions, setUserConditions] = useState('일반 사용자')

  // 앱 상태
  const [view, setView] = useState('home') // onboarding | home | camera | chat | history
  const [previewUrl, setPreviewUrl] = useState(null)
  const [imageBase64, setImageBase64] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [analysisLogs, setAnalysisLogs] = useState([])
  const [currentUser, setCurrentUser] = useState(null)
  const [authReady, setAuthReady] = useState(false)

  // ─ Firebase 익명 인증 ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!auth) { setAuthReady(true); return }
    const unsub = onAuthStateChanged(auth, async (user) => {
      if (user) {
        setCurrentUser(user)
        setAuthReady(true)
      } else {
        try {
          const cred = await signInAnonymously(auth)
          setCurrentUser(cred.user)
        } catch (e) {
          console.warn('익명 로그인 실패:', e.message)
        } finally {
          setAuthReady(true)
        }
      }
    })
    return unsub
  }, [])

  // ─ Firestore 로그 구독 (Auth Guard 포함) ──────────────────────────────────
  useEffect(() => {
    if (!db || !currentUser || !authReady) return

    const q = query(LOGS_PATH(), orderBy('createdAt', 'desc'), limit(20))
    const unsub = onSnapshot(q, snap => {
      setAnalysisLogs(snap.docs.map(d => ({ id: d.id, ...d.data() })))
    }, err => {
      console.warn('Firestore 구독 에러:', err.message)
    })
    return unsub
  }, [currentUser, authReady])

  // ─ Firestore 저장 ─────────────────────────────────────────────────────────
  const saveToFirestore = useCallback(async (result) => {
    if (!db || !currentUser) return
    try {
      await addDoc(LOGS_PATH(), {
        userId: currentUser.uid,
        statusCode: result.statusCode,
        statusText: result.statusText,
        summary: result.summary,
        gastritisImpact: result.gastritisImpact,
        userConditions,
        createdAt: serverTimestamp(),
      })
    } catch (e) {
      console.warn('Firestore 저장 실패:', e.message)
    }
  }, [currentUser, userConditions])

  // ─ 이미지 → Base64 변환 ───────────────────────────────────────────────────
  const processImage = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = e => {
        const dataUrl = e.target.result
        const base64 = dataUrl.split(',')[1]
        const previewUrl = dataUrl
        resolve({ base64, previewUrl })
      }
      reader.onerror = reject
      reader.readAsDataURL(file instanceof Blob ? file : file)
    })
  }, [])

  // ─ GPT-4o Vision 분석 ───────────────────────────────────────────────────
  const analyzeImage = useCallback(async (base64, mimeType = 'image/jpeg') => {
    if (!GROQ_API_KEY) {
      await new Promise(r => setTimeout(r, 2000))
      return {
        status: '⚠️주의',
        statusCode: 'caution',
        statusText: '데모 모드 (API 키 미설정)',
        summary: 'API 키를 설정하면 실제 분석이 가능합니다',
        description: '.env 파일에 VITE_GROQ_API_KEY를 설정해주세요.',
        warnings: 'API 키 없이는 실제 분석을 수행할 수 없습니다.',
        dosageGuide: '.env.example 파일을 참고하여 환경변수를 설정해주세요.',
        gastritisImpact: 5,
        interactions: ['데모 모드'],
        alternatives: 'README.md의 설치 가이드를 참고하세요.',
        activeIngredients: ['데모'],
        drugType: '일반의약품',
        confidence: 0,
      }
    }

    const data = await safeFetchGroq({
      model: GROQ_VISION_MODEL,
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: buildVisionPrompt(userConditions) },
          { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}` } }
        ]
      }],
      temperature: 0.1,
      max_tokens: 1200,
      response_format: { type: 'json_object' },
    })

    const raw = data.choices?.[0]?.message?.content || '{}'
    try {
      const clean = raw.replace(/```json|```/g, '').trim()
      return JSON.parse(clean)
    } catch {
      return {
        status: '❌위험',
        statusCode: 'unidentified',
        summary: 'JSON 파싱 오류',
        description: '분석 결과를 읽을 수 없습니다. 다시 시도해주세요.',
        confidence: 0,
      }
    }
  }, [userConditions])

  // ─ 카메라 캡처 핸들러 ─────────────────────────────────────────────────────
  const handleCameraCapture = useCallback(async (blob) => {
    setView('home')
    const { base64, previewUrl } = await processImage(blob)
    setPreviewUrl(previewUrl)
    setImageBase64(base64)
    setAnalysisResult(null)
    setAnalyzing(true)

    try {
      const result = await analyzeImage(base64, 'image/jpeg')
      setAnalysisResult(result)
      if (result.statusCode !== 'unidentified') {
        await saveToFirestore(result)
      }
    } catch (e) {
      setAnalysisResult({
        status: '❌위험',
        statusCode: 'unidentified',
        summary: '분석 실패',
        description: e.message,
        confidence: 0,
      })
    } finally {
      setAnalyzing(false)
    }
  }, [processImage, analyzeImage, saveToFirestore])

  // ─ 갤러리 업로드 핸들러 ───────────────────────────────────────────────────
  const handleGalleryUpload = useCallback(async (file) => {
    const { base64, previewUrl } = await processImage(file)
    setPreviewUrl(previewUrl)
    setImageBase64(base64)
    setAnalysisResult(null)
    setAnalyzing(true)

    try {
      const mimeType = file.type || 'image/jpeg'
      const result = await analyzeImage(base64, mimeType)
      setAnalysisResult(result)
      if (result.statusCode !== 'unidentified') {
        await saveToFirestore(result)
      }
    } catch (e) {
      setAnalysisResult({
        status: '❌위험',
        statusCode: 'unidentified',
        summary: '분석 실패',
        description: e.message,
        confidence: 0,
      })
    } finally {
      setAnalyzing(false)
    }
  }, [processImage, analyzeImage, saveToFirestore])

  // ─ 온보딩 완료 ────────────────────────────────────────────────────────────
  const handleOnboardingComplete = (conditions) => {
    localStorage.setItem('igeordwae_conditions', conditions)
    setUserConditions(conditions)
    setView('home')
  }

  // ─ 히스토리 항목 선택 ─────────────────────────────────────────────────────
  const handleHistorySelect = (log) => {
    setAnalysisResult({
      ...log,
      status: log.status || (STATUS_MAP[log.statusCode]?.label || ''),
    })
    setPreviewUrl(null)
    setView('home')
  }

  // ─ 렌더링 ─────────────────────────────────────────────────────────────────
  if (view === 'camera') {
    return <CameraView onCapture={handleCameraCapture} onCancel={() => setView('home')} />
  }

  if (view === 'chat' && analysisResult) {
    return (
      <ChatView
        result={analysisResult}
        userConditions={userConditions}
        onBack={() => setView('home')}
      />
    )
  }

  if (view === 'history') {
    return (
      <HistoryView
        logs={analysisLogs}
        onSelect={handleHistorySelect}
        onBack={() => setView('home')}
      />
    )
  }

  return (
    <HomeView
      userConditions={userConditions}
      analysisResult={analysisResult}
      analyzing={analyzing}
      onCameraCapture={() => setView('camera')}
      onGalleryUpload={handleGalleryUpload}
      onChat={() => setView('chat')}
      onHistory={() => setView('history')}
      onRetry={() => {
        setPreviewUrl(null)
        setAnalysisResult(null)
        setImageBase64(null)
      }}
      previewUrl={previewUrl}
      logCount={analysisLogs.length}
      currentUser={currentUser}
    />
  )
}
