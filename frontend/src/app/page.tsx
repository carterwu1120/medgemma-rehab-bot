"use client";

import { FormEvent, useMemo, useState } from "react";
import { ChatResponse, chatWithBackend } from "@/lib/api";

type ChatTurn = {
  id: number;
  query: string;
  response?: ChatResponse;
  error?: string;
  isLoading: boolean;
};

function chipClass(tone: "neutral" | "warn" = "neutral"): string {
  if (tone === "warn") {
    return "rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-900";
  }
  return "rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700";
}

function scoreLabel(score: number): string {
  return score.toFixed(3);
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(8);
  const [videoLimit, setVideoLimit] = useState(3);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const latestResponse = useMemo(
    () => turns.find((turn) => turn.response)?.response,
    [turns],
  );

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isSubmitting) return;

    setIsSubmitting(true);
    const id = Date.now();
    setTurns((prev) => [{ id, query: trimmed, isLoading: true }, ...prev]);
    setQuery("");

    try {
      const response = await chatWithBackend({
        query: trimmed,
        top_k: topK,
        video_limit: videoLimit,
        temperature: 0,
        max_tokens: 450,
      });

      setTurns((prev) =>
        prev.map((turn) =>
          turn.id === id ? { ...turn, response, isLoading: false } : turn,
        ),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Request failed";
      setTurns((prev) =>
        prev.map((turn) =>
          turn.id === id ? { ...turn, error: message, isLoading: false } : turn,
        ),
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,#dbeafe,transparent_50%),linear-gradient(180deg,#f8fafc,#eef2ff)] px-4 py-8 text-slate-900 md:px-8">
      <div className="mx-auto w-full max-w-7xl">
        <header className="mb-6 rounded-2xl border border-slate-200 bg-white/90 p-6 shadow-sm backdrop-blur">
          <p className="text-xs uppercase tracking-[0.16em] text-slate-500">RehabCompass 居家復健助手</p>
          <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">Exercise, Rehab &amp; Stretch Guide</h1>
          <p className="mt-2 text-sm text-slate-600">
            描述你的不適與情境，我會提供安全優先的分步建議、停止/就醫警訊，並推薦可跟做的復健影片與參考來源。
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <form className="space-y-4" onSubmit={onSubmit}>
              <label className="block space-y-2">
                <span className="text-sm font-medium">Question</span>
                <textarea
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="例如：我久坐後下背痠痛，請給我10分鐘安全流程"
                  className="h-28 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none ring-sky-300 transition focus:border-sky-400 focus:ring"
                />
              </label>

              <div className="grid gap-3 sm:grid-cols-3">
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Top K</span>
                  <input
                    type="number"
                    min={1}
                    max={20}
                    value={topK}
                    onChange={(event) => setTopK(Number(event.target.value) || 8)}
                    className="w-full rounded-lg border border-slate-300 px-2 py-2 text-sm"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Video Limit</span>
                  <input
                    type="number"
                    min={0}
                    max={10}
                    value={videoLimit}
                    onChange={(event) => setVideoLimit(Number(event.target.value) || 0)}
                    className="w-full rounded-lg border border-slate-300 px-2 py-2 text-sm"
                  />
                </label>
                <div className="flex items-end">
                  <button
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
                  >
                    {isSubmitting ? "Generating..." : "Ask"}
                  </button>
                </div>
              </div>
            </form>

            <div className="mt-6 space-y-4">
              {turns.length === 0 && (
                <p className="rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-sm text-slate-500">
                  No messages yet.
                </p>
              )}

              {turns.map((turn) => (
                <article key={turn.id} className="rounded-xl border border-slate-200 p-4">
                  <p className="text-xs uppercase tracking-wide text-slate-500">User</p>
                  <p className="mt-1 whitespace-pre-wrap text-sm">{turn.query}</p>

                  <div className="mt-4 border-t border-slate-100 pt-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">Assistant</p>
                    {turn.isLoading && (
                      <p className="mt-1 text-sm text-slate-500">Processing...</p>
                    )}
                    {turn.error && (
                      <p className="mt-1 rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-700">
                        {turn.error}
                      </p>
                    )}
                    {turn.response && (
                      <>
                        <p className="mt-2 whitespace-pre-wrap text-sm leading-6">{turn.response.answer}</p>
                        {turn.response.references.length > 0 && (
                          <div className="mt-3 flex flex-wrap gap-2">
                            {turn.response.references.map((ref) => (
                              <span key={ref} className={chipClass("neutral")}>
                                {ref}
                              </span>
                            ))}
                          </div>
                        )}
                        {turn.response.policy_notes.length > 0 && (
                          <div className="mt-3 flex flex-wrap gap-2">
                            {turn.response.policy_notes.map((note) => (
                              <span key={note} className={chipClass("warn")}>
                                {note}
                              </span>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="space-y-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="text-lg font-semibold">Evidence Chunks</h2>
              <p className="mt-1 text-xs text-slate-500">From latest successful response</p>

              <div className="mt-4 space-y-3">
                {!latestResponse && (
                  <p className="text-sm text-slate-500">No chunks yet.</p>
                )}
                {latestResponse?.retrieved_chunks.map((chunk) => (
                  <div key={chunk.chunk_id} className="rounded-lg border border-slate-200 p-3">
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-xs font-medium text-slate-700">{chunk.source_name} · p{chunk.page}</p>
                      <span className="text-xs text-slate-500">{scoreLabel(chunk.score)}</span>
                    </div>
                    <p className="mt-2 text-xs text-slate-500">{chunk.chunk_id}</p>
                    <p className="mt-2 line-clamp-4 text-sm text-slate-700">{chunk.snippet}</p>
                    {chunk.tags.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {chunk.tags.map((tag) => (
                          <span key={tag} className={chipClass("neutral")}>
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="text-lg font-semibold">Recommended Videos</h2>
              <p className="mt-1 text-xs text-slate-500">From latest successful response</p>

              <div className="mt-4 space-y-3">
                {!latestResponse && (
                  <p className="text-sm text-slate-500">No videos yet.</p>
                )}
                {latestResponse?.videos.map((video) => (
                  <div key={`${video.provider}-${video.video_id}`} className="rounded-lg border border-slate-200 p-3">
                    <a
                      href={video.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-sm font-semibold text-sky-700 underline-offset-2 hover:underline"
                    >
                      {video.title}
                    </a>
                    <p className="mt-1 text-xs text-slate-500">
                      {video.provider} · score {scoreLabel(video.score)}
                    </p>
                    <p className="mt-2 text-sm text-slate-700">{video.summary}</p>
                    {video.why.length > 0 && (
                      <p className="mt-2 text-xs text-slate-600">Why: {video.why.join("; ")}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
