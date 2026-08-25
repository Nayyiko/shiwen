import { useRef, useState } from 'react'
import './App.css'

// ── 类型 ──────────────────────────────────────────────────────────────────────

interface Citation {
  book: string
  chapter: string
  version: string
  text: string
}

interface Speech {
  sage_id: string
  name: string
  school: string
  text: string
  urgency_rank: number
  citations: Citation[]
}

const SAGES = [
  { id: 'kongzi', name: '孔子', school: '儒家' },
  { id: 'mengzi', name: '孟子', school: '儒家' },
  { id: 'laozi', name: '老子', school: '道家' },
  { id: 'hanfei', name: '韩非子', school: '法家' },
]

type Tab = 'chat' | 'write' | 'roleplay' | 'debate'

// ── SSE 读取（fetch stream，兼容 POST）─────────────────────────────────────

async function streamSSE(url: string, body: object, onEvent: (e: Record<string, any>) => void) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok || !res.body) throw new Error('HTTP ' + res.status)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      for (const line of raw.split('\n')) {
        if (line.startsWith('data: ')) {
          try { onEvent(JSON.parse(line.slice(6))) } catch { /* 忽略解析错误 */ }
        }
      }
    }
  }
}

function genId(prefix: string): string {
  return prefix + Date.now().toString(36) + Math.random().toString(36).slice(2, 6)
}

// ── 主组件 ────────────────────────────────────────────────────────────────────

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

// ── 研微·问答（流式）─────────────────────────────────────────────────────────

function ChatTab() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [answer, setAnswer] = useState('')
  const [citations, setCitations] = useState<Citation[]>([])
  const [error, setError] = useState('')

  async function ask() {
    if (!query.trim() || loading) return
    setLoading(true)
    setError('')
    setAnswer('')
    setCitations([])
    try {
      await streamSSE('/api/chat/stream', { query }, (e) => {
        if (e.type === 'citations') setCitations(e.citations || [])
        else if (e.type === 'token') setAnswer((a) => a + e.token)
      })
    } catch (err) {
      setError('请求失败：' + String(err))
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
      {answer && (
        <div className="answer">
          <p className="answer-text">{answer}</p>
          {citations.length > 0 && (
            <div className="citations">
              <h3>引据（{citations.length}）</h3>
              <ul>
                {citations.map((c, i) => (
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

// ── 研微·写作（流式）─────────────────────────────────────────────────────────

function WriteTab() {
  const [topic, setTopic] = useState('')
  const [loading, setLoading] = useState(false)
  const [outline, setOutline] = useState<string[]>([])
  const [sections, setSections] = useState<{ title: string; text: string }[]>([])
  const [article, setArticle] = useState('')
  const [error, setError] = useState('')

  async function run() {
    if (!topic.trim() || loading) return
    setLoading(true)
    setError('')
    setOutline([])
    setSections([])
    setArticle('')
    try {
      await streamSSE('/api/write/stream', { topic, max_sections: 3 }, (e) => {
        if (e.type === 'outline') setOutline((e.sections || []).map((s: any) => s.title))
        else if (e.type === 'section') setSections((prev) => [...prev, { title: e.title, text: e.text }])
        else if (e.type === 'article') setArticle(e.article || '')
      })
    } catch (err) {
      setError('请求失败：' + String(err))
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

      {outline.length > 0 && (
        <div className="outline">
          <h3>大纲（{outline.length} 节）</h3>
          <ol>{outline.map((t, i) => <li key={i}>{t}</li>)}</ol>
        </div>
      )}

      {sections.map((s, i) => (
        <div className="section" key={i}>
          <h3>{s.title}</h3>
          <p>{s.text}</p>
        </div>
      ))}

      {article && (
        <div className="article">
          <div className="article-body">
            {article.split('\n').map((line, i) =>
              line.startsWith('#') ? <h2 key={i}>{line.replace(/^#+\s*/, '')}</h2> : <p key={i}>{line}</p>
            )}
          </div>
        </div>
      )}
    </section>
  )
}

// ── 新裁·角色扮演（流式 + Redis 记忆）────────────────────────────────────────

function RoleplayTab() {
  const [sageId, setSageId] = useState('kongzi')
  const [sessionId, setSessionId] = useState(() => genId('sess-'))
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<{ role: string; content: string }[]>([])
  const [error, setError] = useState('')
  const convRef = useRef<HTMLDivElement>(null)

  async function send() {
    if (!message.trim() || loading) return
    setLoading(true)
    setError('')
    const userMsg = message
    setMessage('')
    setHistory((h) => [...h, { role: 'user', content: userMsg }])
    try {
      await streamSSE('/api/roleplay/stream', { sage_id: sageId, message: userMsg, session_id: sessionId }, (e) => {
        if (e.type === 'token') {
          setHistory((h) => {
            const last = h[h.length - 1]
            if (last && last.role === 'assistant') {
              return [...h.slice(0, -1), { ...last, content: last.content + e.token }]
            }
            return [...h, { role: 'assistant', content: e.token }]
          })
        }
      })
    } catch (err) {
      setError('请求失败：' + String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section>
      <div className="sage-picker">
        {SAGES.map((s) => (
          <button key={s.id} className={sageId === s.id ? 'active' : ''}
            onClick={() => { setSageId(s.id); setHistory([]); setSessionId(genId('sess-')) }}>
            {s.name}（{s.school}）
          </button>
        ))}
      </div>
      <div className="conversation" ref={convRef}>
        {history.map((h, i) => (
          <div key={i} className={h.role === 'user' ? 'msg-user' : 'msg-sage'}>{h.content}</div>
        ))}
        {loading && <div className="msg-sage">思考中…</div>}
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

// ── 先贤辩论（流式 + 插话）───────────────────────────────────────────────────

function DebateTab() {
  const [topic, setTopic] = useState('')
  const [loading, setLoading] = useState(false)
  const [speeches, setSpeeches] = useState<Speech[]>([])
  const [summary, setSummary] = useState('')
  const [interjections, setInterjections] = useState<string[]>([])
  const [interject, setInterject] = useState('')
  const [topicId, setTopicId] = useState('')
  const [error, setError] = useState('')

  async function run() {
    if (!topic.trim() || loading) return
    setLoading(true)
    setError('')
    setSpeeches([])
    setSummary('')
    setInterjections([])
    const tid = genId('deb-')
    setTopicId(tid)
    try {
      await streamSSE(`/api/debate/stream?topic_id=${tid}`, { topic, max_speeches: 8 }, (e) => {
        if (e.type === 'speech') setSpeeches((prev) => [...prev, e.speech as Speech])
        else if (e.type === 'summary') setSummary(e.summary || '')
        else if (e.type === 'interjection') setInterjections((prev) => [...prev, e.message])
      })
    } catch (err) {
      setError('请求失败：' + String(err))
    } finally {
      setLoading(false)
    }
  }

  async function sendInterject() {
    if (!interject.trim() || !topicId) return
    const msg = interject
    setInterject('')
    setInterjections((prev) => [...prev, `你：${msg}`])
    try {
      await fetch('/api/debate/interject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic_id: topicId, message: msg }),
      })
    } catch { /* 插话失败不阻塞 */ }
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

      {loading && (
        <div className="chat interject-bar">
          <input
            value={interject}
            onChange={(e) => setInterject(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendInterject()}
            placeholder="辩论进行中，可插话追问…"
          />
          <button onClick={sendInterject}>插话</button>
        </div>
      )}

      {interjections.map((msg, i) => (
        <div key={i} className="interjection">{msg}</div>
      ))}

      <div className="debate">
        {speeches.map((s, i) => (
          <div key={i} className="speech">
            <div className="speech-head"><b>{s.name}</b>（{s.school}）· 第{s.urgency_rank}顺位</div>
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
        {summary && (
          <div className="debate-summary">
            <h3>主持总结</h3>
            <p>{summary}</p>
          </div>
        )}
      </div>
    </section>
  )
}
