import { useEffect, useMemo, useState } from "react";
import type { OrganismState } from "./types";

const EMPTY: OrganismState = {
  tick: 0,
  time: 0,
  name: "01",
  lineage: "",
  mode: "awake",
  stage: "",
  prediction_error: 0,
  uncertainty: 0,
  workspace: { kinds: [], scores: [], ignition: 0, broadcast: [], sources: [] },
  motivation: {},
  intrinsic: { total: 0, progress: 0, novelty: 0, weights: {} },
  goals: [],
  entities: [],
  self: { identity: [], capability: 0, controllability: 0, agency: 0, continuity: 0 },
  meta: {},
  lexicon_size: 0,
  lexicon_words: [],
  episodic_count: 0,
  semantic_count: 0,
  beliefs: [],
  last_utterance: "",
  last_intent: null,
  dialogue: { partner: null, turns: [] },
  imagination: [],
  causal: {},
  development: { stage: "", metrics: {}, history: [] },
  milestones: [],
  relationships: {},
  preferences: {},
  spontaneous_utterances: 0,
  attention: null,
  working_memory: [],
  core_norm: 0,
};

function Heat({ values }: { values: number[] }) {
  const slice = values.slice(0, 64);
  const max = Math.max(...slice.map(Math.abs), 1e-6);
  return (
    <div className="heat">
      {slice.map((v, i) => {
        const n = (v / max + 1) / 2;
        const c = `hsl(${170 + n * 40} 70% ${20 + n * 45}%)`;
        return <i key={i} style={{ background: c }} />;
      })}
    </div>
  );
}

function Bar({ value, kind }: { value: number; kind?: string }) {
  return (
    <div className={`bar ${kind ?? ""}`}>
      <span style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
    </div>
  );
}

export function App() {
  const [state, setState] = useState<OrganismState>(EMPTY);
  const [text, setText] = useState("");
  const [tab, setTab] = useState("memory");
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let alive = true;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (alive) setTimeout(connect, 1200);
      };
      ws.onmessage = (ev) => setState(JSON.parse(ev.data));
    };
    const poll = async () => {
      try {
        const r = await fetch("/api/state");
        if (r.ok) setState(await r.json());
      } catch {
        /* runtime may still be starting */
      }
    };
    poll();
    connect();
    const id = setInterval(poll, 900);
    return () => {
      alive = false;
      ws?.close();
      clearInterval(id);
    };
  }, []);

  const say = async () => {
    const t = text.trim();
    if (!t) return;
    await fetch("/api/say", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker: "human", text: t }),
    });
    setText("");
  };

  const world = state.world;
  const tabs = ["memory", "beliefs", "self", "imagination", "language", "development", "metrics"];

  const identityHeat = useMemo(() => state.self.identity ?? [], [state.self.identity]);

  return (
    <div className="app">
      <header className="top">
        <div className="brand">01</div>
        <div className={`pill ${connected ? "on" : ""}`}>{connected ? "live" : "seeking runtime"}</div>
        <div className="pill">tick {state.tick}</div>
        <div className="pill">mode {state.mode}</div>
        <div className="pill">stage {state.stage.replaceAll("_", " ")}</div>
        <div className="pill">lineage {state.lineage}</div>
        <div className="tools">
          <button onClick={() => fetch("/api/checkpoint", { method: "POST" })}>checkpoint</button>
          <button onClick={() => fetch("/api/step", { method: "POST" })}>step</button>
        </div>
      </header>
      <main className="grid">
        <section className="col">
          <div className="panel chat" style={{ flex: 1 }}>
            <h2>Conversation — environmental events</h2>
            <div className="turns">
              {(state.dialogue.turns || []).map((t, i) => {
                const origin = t.act || (t.speaker === "01" ? "organism" : "human");
                const cls = origin === "spontaneous" ? "spontaneous" : t.speaker === "01" ? "o1" : "human";
                return (
                  <div key={i} className={`turn ${cls}`}>
                    <div className="who">
                      {t.speaker} · tick {t.tick}
                      {` · ${origin}`}
                    </div>
                    <div>{t.text}</div>
                  </div>
                );
              })}
            </div>
            <div className="composer">
              <input
                value={text}
                placeholder="Speak into the environment"
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && say()}
              />
              <button onClick={say}>send</button>
            </div>
            <p className="note">
              Human turns are environmental events. Organism turns require a learned communicative state. Spontaneous turns are organism-originated without a human prompt.
            </p>
            {state.last_intent && (
              <p className="note">
                Last intent: {state.last_intent.act} · urgency {state.last_intent.urgency.toFixed(2)} · prior uncertainty{" "}
                {state.last_intent.uncertainty.toFixed(2)} · retrievals {state.last_intent.retrievals.length}
              </p>
            )}
          </div>
        </section>
        <section className="col">
          <div className="panel">
            <h2>World — researcher view, not organism pixels</h2>
            <div className="world">
              {(world?.objects || []).map((o) => (
                <div
                  key={o.id}
                  className="obj"
                  title={o.id}
                  style={{
                    left: `${(o.x / (world?.width || 12)) * 100}%`,
                    top: `${(o.y / (world?.height || 12)) * 100}%`,
                    width: `${Math.max(8, o.r * 18)}px`,
                    height: `${Math.max(8, o.r * 18)}px`,
                    background: `rgb(${o.color.map((c) => Math.round(c * 255)).join(",")})`,
                    opacity: o.hidden ? 0.15 : 1,
                    boxShadow: o.kind === "light" && o.light > 0.5 ? "0 0 18px #ffd978" : undefined,
                  }}
                />
              ))}
            </div>
          </div>
          <div className="panel">
            <h2>Global workspace vectors (not thought bubbles)</h2>
            <div className="row">
              <span>ignition {state.workspace.ignition.toFixed(3)}</span>
              <span>{state.workspace.kinds.join(" · ") || "empty"}</span>
            </div>
            <Heat values={state.workspace.broadcast} />
            <p className="note">Labels below are researcher interpretations of candidate kinds, not organism-generated English.</p>
            <ul>
              {state.workspace.kinds.map((k, i) => (
                <li key={i}>
                  {k} · score {(state.workspace.scores[i] ?? 0).toFixed(3)} · {state.workspace.sources[i]}
                </li>
              ))}
            </ul>
          </div>
          <div className="panel">
            <h2>Entities</h2>
            <div className="kv">
              {state.entities.map((e) => (
                <div key={e.id} style={{ display: "contents" }}>
                  <span>
                    {e.id}
                    {e.self ? " (self)" : ""}
                    {e.agent ? " agent" : ""} {e.visible ? "" : " occluded"}
                  </span>
                  <span>u {e.uncertainty.toFixed(2)} p {e.permanence.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
        <section className="col">
          <div className="panel">
            <h2>Homeostasis / motivation</h2>
            {Object.entries(state.motivation).map(([k, v]) => (
              <div key={k}>
                <div className="row">
                  <span>{k}</span>
                  <span>{v.toFixed(3)}</span>
                </div>
                <Bar value={v} kind={k.includes("error") || k.includes("rest") ? "rose" : undefined} />
              </div>
            ))}
            <div className="row">
              <span>intrinsic {state.intrinsic.total.toFixed(3)}</span>
              <span>progress {state.intrinsic.progress.toFixed(3)}</span>
            </div>
          </div>
          <div className="panel" style={{ flex: 1 }}>
            <div className="tabs">
              {tabs.map((t) => (
                <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
                  {t}
                </button>
              ))}
            </div>
            {tab === "memory" && (
              <div>
                <p className="note">
                  episodic {state.episodic_count} · semantic {state.semantic_count} · lexicon {state.lexicon_size}
                </p>
                <div className="note">words: {state.lexicon_words.join(" ") || "none grounded yet"}</div>
                <h2>Working memory</h2>
                <ul>
                  {state.working_memory.map((w, i) => (
                    <li key={i}>
                      {w.source} · {w.act.toFixed(2)} · {w.entity ?? ""}
                    </li>
                  ))}
                </ul>
                <h2>Goals</h2>
                <ul>
                  {state.goals.map((g) => (
                    <li key={g.id}>
                      {g.origin} {g.action ?? ""} p={g.priority.toFixed(2)} {g.status}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {tab === "beliefs" && (
              <ul>
                {state.beliefs.map((b) => (
                  <li key={b.id}>
                    {b.prop} · conf {b.conf.toFixed(2)} · contra {b.contra}
                  </li>
                ))}
              </ul>
            )}
            {tab === "self" && (
              <div>
                <div className="row">
                  <span>capability {state.self.capability.toFixed(3)}</span>
                  <span>agency {state.self.agency.toFixed(3)}</span>
                </div>
                <Bar value={state.self.capability} />
                <div className="row">
                  <span>controllability {state.self.controllability.toFixed(3)}</span>
                  <span>continuity {state.self.continuity.toFixed(3)}</span>
                </div>
                <Heat values={identityHeat} />
                <p className="note">Identity vector. Not a biography and not English.</p>
              </div>
            )}
            {tab === "imagination" && (
              <ul>
                {state.imagination.map((t, i) => (
                  <li key={i}>
                    {t.kind} · V {t.value.toFixed(3)} · info {t.info.toFixed(3)} · pe {t.pe.toFixed(3)}
                  </li>
                ))}
              </ul>
            )}
            {tab === "language" && (
              <div>
                <p className="note">Semantic intent exists before characters. Vectors are not English thoughts.</p>
                {state.last_intent && (
                  <div className="kv">
                    <span>urgency</span>
                    <span>{state.last_intent.urgency.toFixed(3)}</span>
                    <span>prior uncertainty</span>
                    <span>{state.last_intent.uncertainty.toFixed(3)}</span>
                    <span>prior PE</span>
                    <span>{state.last_intent.pe.toFixed(3)}</span>
                    <span>retrievals</span>
                    <span>{state.last_intent.retrievals.join(",") || "none"}</span>
                  </div>
                )}
                <Heat values={state.workspace.broadcast} />
              </div>
            )}
            {tab === "development" && (
              <div>
                <ul>
                  {state.milestones.map((m, i) => (
                    <li key={i}>
                      t{m.tick} {m.name} — {m.evidence}
                    </li>
                  ))}
                </ul>
                <div className="kv">
                  {Object.entries(state.development.metrics).map(([k, v]) => (
                    <div key={k} style={{ display: "contents" }}>
                      <span>{k}</span>
                      <span>{v.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {tab === "metrics" && (
              <div className="kv">
                <span>prediction error</span>
                <span>{state.prediction_error.toFixed(4)}</span>
                <span>uncertainty</span>
                <span>{state.uncertainty.toFixed(4)}</span>
                <span>core norm</span>
                <span>{state.core_norm.toFixed(3)}</span>
                <span>spontaneous speech</span>
                <span>{state.spontaneous_utterances}</span>
                <span>attention</span>
                <span>{state.attention ?? "—"}</span>
              </div>
            )}
          </div>
        </section>
      </main>
      <footer className="foot">
        <span className="warn">Observatory only — this UI is not the organism.</span>
        <span style={{ marginLeft: 12 }}>No LLM is used for thought or speech.</span>
      </footer>
    </div>
  );
}
