import { useState } from 'react'
import './App.css'

// ── 后端接口类型（与 src/shiwen/api/main.py 对齐） ──────────────────────────

interface Citation {
  book: string
  chapter: string
  version: string
  text: string
}

interface ChatResponse {
  answer: string
  citations: Citation[]
  rounds: number
  grounding_pass: boolean
  grounding_reason?: string | null
}

interface WriteResponse {
  topic: string
  sections: { title: string; text: string; citations_count: number }[]
  article: string
  citations: Citation[]
}

interface RoleplayResponse {
  sage_id: string
  sage_name: string
  school: string
  response: string
  citations: Citation[]
}

interface DebateSpeech {
  sage_id: string
  name: string
  school: string
  text: string
  citations: Citation[]
  urgency_rank: number
}

interface DebateResponse {
  topic: string
  speeches: DebateSpeech[]
  summary: string
  drift_events: unknown[]
}

const SAGES = [
  { id: 'kongzi', name: '孔子', school: '儒家' },
  { id: 'mengzi', name: '孟子', school: '儒家' },
  { id: 'laozi', name: '老子', school: '道家' },
  { id: 'hanfei', name: '韩非子', school: '法家' },
]

function genSessionId(): string {
  return 'sess-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
}

type Tab = 'chat' | 'write' | 'roleplay' | 'debate'

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')
  return (
    <main className="container">
      <h1>识文新裁</h1>
      <p className="tagline">贯穿古籍「数字化整理 → 学术化研究 → 大众化传播」全生命周期的 AI 平台</p>

      <nav className="tabs">
        <button className={tab === 'chat' ? 'active' : ''} onClick={() => setTab('chat')}>研微·问答</button>
        <button className={tab === 'write' ? 'active' : ''} onClick={() => setTab('write')}>研微·写作</button>
        <button className={tab === 'roleplay' ? 'active' : ''} onClick={() => setTab('roleplay')}>新裁·角色扮演</button>
        <button className={tab === 'debate' ? 'active' : ''} onClick={() => setTab('debate')}>先贤辩论</button>
      </nav>

      {tab === 'chat' && <ChatTab />}
      {tab === 'write' && <WriteTab />}
      {tab === 'roleplay' && <RoleplayTab />}
      {tab === 'debate' && <DebateTab />}
    </main>
  )
}

// ── 研微·问答 ────────────────────────────────────────────────────────────────

function ChatTab() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [reply, setReply] = useState<ChatResponse | null>(null)
  const [error, setError] = useState('')

  async function ask() {
    if (!query.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()
      setReply(data)
    } catch (e) {
      setError('请求失败：' + String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <div className="chat">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && ask()}
          placeholder="问一句古籍，如：学而时习之出自哪一篇？"
        />
        <button onClick={ask} disabled={loading}>{loading ? '思考中…' : '提问'}</button>
      </div>
      {error && <p className="error">{error}</p>}
      {reply && (
        <div className="answer">
          <p className="answer-text">{reply.answer}</p>
          {reply.citations.length > 0 && (
            <div className="citations">
              <h3>引据（{reply.citations.length}）</h3>
              <ul>
                {reply.citations.map((c, i) => (
                  <li key={i}>
                    <b>「{c.book}·{c.chapter}（{c.version}）」</b>
                    <span>{c.text}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

// ── 研微·写作 ────────────────────────────────────────────────────────────────

function WriteTab() {
  const [topic, setTopic] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<WriteResponse | null>(null)
  const [error, setError] = useState('')

  async function run() {
    if (!topic.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/write', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, max_sections: 3 }),
      })
      setResult(await res.json())
    } catch (e) {
      setError('请求失败：' + String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <div className="chat">
        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="写作选题，如：论语中的仁学思想"
        />
        <button onClick={run} disabled={loading}>{loading ? '写作中…' : '生成文章'}</button>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <article className="article">
          <div className="article-body">
            {result.article.split('\n').map((line, i) =>
              line.startsWith('#') ? <h2 key={i}>{line.replace(/^#+\s*/, '')}</h2> : <p key={i}>{line}</p>
            )}
          </div>
        </article>
      )}
    </section>
  )
}

// ── 新裁·角色扮演 ────────────────────────────────────────────────────────────

function RoleplayTab() {
  const [sageId, setSageId] = useState('kongzi')
  const [sessionId, setSessionId] = useState(() => genSessionId())
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<{ role: string; content: string }[]>([])
  const [error, setError] = useState('')

  async function send() {
    if (!message.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/roleplay', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sage_id: sageId, message, session_id: sessionId }),
      })
      const data = await res.json()
      setHistory([...history, { role: 'user', content: message }, { role: 'assistant', content: data.response }])
      setMessage('')
    } catch (e) {
      setError('请求失败：' + String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <div className="sage-picker">
        {SAGES.map((s) => (
          <button key={s.id} className={sageId === s.id ? 'active' : ''} onClick={() => { setSageId(s.id); setHistory([]); setSessionId(genSessionId()) }}>
            {s.name}（{s.school}）
          </button>
        ))}
      </div>
      <div className="conversation">
        {history.map((h, i) => (
          <div key={i} className={h.role === 'user' ? 'msg-user' : 'msg-sage'}>
            {h.content}
          </div>
        ))}
      </div>
      <div className="chat">
        <input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder={`向${SAGES.find((s) => s.id === sageId)?.name}请教…`}
        />
        <button onClick={send} disabled={loading}>{loading ? '回答中…' : '发送'}</button>
      </div>
      {error && <p className="error">{error}</p>}
    </section>
  )
}

// ── 先贤辩论 ────────────────────────────────────────────────────────────────

function DebateTab() {
  const [topic, setTopic] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<DebateResponse | null>(null)
  const [error, setError] = useState('')

  async function run() {
    if (!topic.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch('/api/debate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, max_speeches: 8 }),
      })
      setResult(await res.json())
    } catch (e) {
      setError('请求失败：' + String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <div className="chat">
        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="辩题，如：德治与法治哪个更适合治国？"
        />
        <button onClick={run} disabled={loading}>{loading ? '辩论中…' : '开始辩论'}</button>
      </div>
      {error && <p className="error">{error}</p>}
      {result && (
        <div className="debate">
          {result.speeches.map((s, i) => (
            <div key={i} className="speech">
              <div className="speech-head">
                <b>{s.name}</b>（{s.school}）· 第{s.urgency_rank}顺位
              </div>
              <p>{s.text}</p>
              {s.citations.length > 0 && (
                <div className="speech-citations">
                  {s.citations.map((c, j) => (
                    <span key={j}>「{c.book}·{c.chapter}」</span>
                  ))}
                </div>
              )}
            </div>
          ))}
          {result.summary && (
            <div className="debate-summary">
              <h3>主持总结</h3>
              <p>{result.summary}</p>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
