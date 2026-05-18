import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

function groqDevProxy(env) {
  return {
    name: 'groq-dev-proxy',
    configureServer(server) {
      server.middlewares.use('/api/groq-proxy', async (req, res) => {
        res.setHeader('Access-Control-Allow-Origin', '*')
        res.setHeader('Content-Type', 'application/json; charset=utf-8')

        if (req.method === 'OPTIONS') {
          res.statusCode = 200
          res.end()
          return
        }

        if (req.method !== 'POST') {
          res.statusCode = 405
          res.end(JSON.stringify({ error: 'Method not allowed' }))
          return
        }

        const apiKey = env.GROQ_API_KEY || env.VITE_GROQ_API_KEY
        if (!apiKey) {
          res.statusCode = 500
          res.end(JSON.stringify({ error: 'GROQ_API_KEY not configured' }))
          return
        }

        try {
          const chunks = []
          for await (const chunk of req) chunks.push(chunk)
          const body = Buffer.concat(chunks).toString('utf8')

          const upstream = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${apiKey}`,
            },
            body,
          })
          const text = await upstream.text()
          res.statusCode = upstream.status
          res.end(text)
        } catch (error) {
          res.statusCode = 502
          res.end(JSON.stringify({ error: error.message }))
        }
      })
    },
  }
}

function mfdsDevProxy(env) {
  const endpoints = {
    drugInfo: 'https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList',
    pillInfo: 'https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03',
    permission: 'https://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnDtlInq05',
  }

  return {
    name: 'mfds-dev-proxy',
    configureServer(server) {
      server.middlewares.use('/api/mfds-proxy', async (req, res) => {
        try {
          const url = new URL(req.url || '', 'http://localhost')
          const endpoint = url.searchParams.get('endpoint')
          const targetBase = endpoints[endpoint]
          const serviceKey = env.MFDS_API_KEY || env.VITE_MFDS_API_KEY

          res.setHeader('Access-Control-Allow-Origin', '*')
          res.setHeader('Content-Type', 'application/json; charset=utf-8')

          if (!targetBase) {
            res.statusCode = 400
            res.end(JSON.stringify({ error: `Unknown endpoint: ${endpoint}` }))
            return
          }

          if (!serviceKey) {
            res.statusCode = 500
            res.end(JSON.stringify({ error: 'MFDS_API_KEY not configured' }))
            return
          }

          url.searchParams.delete('endpoint')
          url.searchParams.set('serviceKey', serviceKey)
          url.searchParams.set('type', 'json')

          const response = await fetch(`${targetBase}?${url.searchParams}`)
          const text = await response.text()
          res.statusCode = response.status
          res.end(text)
        } catch (error) {
          res.statusCode = 502
          res.end(JSON.stringify({ error: error.message }))
        }
      })
    },
  }
}

// DL 모델 추론 서버 프록시 (ml/server.py → localhost:5001)
function modelDevProxy() {
  return {
    name: 'model-dev-proxy',
    configureServer(server) {
      server.middlewares.use('/api/model-inference', async (req, res) => {
        res.setHeader('Access-Control-Allow-Origin', '*')
        res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
        res.setHeader('Content-Type', 'application/json; charset=utf-8')

        if (req.method === 'OPTIONS') { res.statusCode = 200; res.end(); return }

        try {
          // health 체크는 GET
          const subPath = req.url?.replace(/\?.*$/, '') || ''
          const targetUrl = `http://localhost:5001/api/model-inference${subPath}`

          if (req.method === 'GET') {
            const upstream = await fetch(targetUrl)
            const text = await upstream.text()
            res.statusCode = upstream.status
            res.end(text)
            return
          }

          // POST — 이미지 추론
          const chunks = []
          for await (const chunk of req) chunks.push(chunk)
          const body = Buffer.concat(chunks).toString('utf8')

          const upstream = await fetch(targetUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
          })
          const text = await upstream.text()
          res.statusCode = upstream.status
          res.end(text)
        } catch (error) {
          // 모델 서버 미실행 → 503 (앱에서 Groq 폴백)
          res.statusCode = 503
          res.end(JSON.stringify({ error: 'DL 모델 서버 미연결', fallback: true }))
        }
      })
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  return {
  plugins: [react(), mfdsDevProxy(env), groqDevProxy(env), modelDevProxy()],
  server: {
    port: 3000,
    host: true
  }
  }
})
