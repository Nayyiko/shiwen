import { useState } from 'react'
import './App.css'

interface ChatResponse {
  message: string
}

const modules = [
  { name: '研微', desc: '学术 RAG 问答（带引据）· 研究写作' },
  { name: '新裁', desc: '历史人物角色扮演 · 沉浸式叙事' },
  { name: '识文', desc: '古籍 OCR / 整理入口' },
]

export default function App() {
  const [query, setQuery] = useState('')
  const [reply, setReply] = useState('')

  async function ask() {
    if (!query.trim()) return
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    })
    const data: ChatResponse = await res.json()
    setReply(data.message)
  }

  return (
    <main className="container">
      <h1>识文新裁</h1>
      <p className="tagline">
        贯穿古籍「数字化整理 → 学术化研究 → 大众化传播」全生命周期的 AI 平台
      </p>

      <section className="modules">
        {modules.map((m) => (
          <div className="card" key={m.name}>
            <h2>{m.name}</h2>
            <p>{m.desc}</p>
          </div>
        ))}
      </section>

      <section className="chat">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && ask()}
          placeholder="问一句古籍，如：学而时习之出自哪一篇？"
        />
        <button onClick={ask}>提问</button>
        {reply && <p className="reply">{reply}</p>}
      </section>
    </main>
  )
}
