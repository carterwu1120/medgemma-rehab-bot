export type RetrievedChunk = {
  chunk_id: string;
  source_name: string;
  page: number;
  score: number;
  tags: string[];
  snippet: string;
};

export type VideoCandidate = {
  video_id: string;
  title: string;
  url: string;
  provider: string;
  score: number;
  summary: string;
  tags: string[];
  intent_tags: string[];
  why: string[];
  notes: string;
};

export type ChatResponse = {
  user_id: string;
  session_id: string;
  episode_id?: string | null;
  history_turns_used: number;
  effective_query: string;
  answer: string;
  language: string;
  policy_notes: string[];
  references: string[];
  episode_slots: Record<string, string>;
  body_tags: string[];
  intent_tags: string[];
  retrieved_chunks: RetrievedChunk[];
  videos: VideoCandidate[];
};

export type ChatPayload = {
  query: string;
  top_k?: number;
  video_limit?: number;
  temperature?: number;
  max_tokens?: number;
  user_id?: string;
  session_id?: string;
};

function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:9000";
}

export async function chatWithBackend(
  payload: ChatPayload,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch(`${getApiBaseUrl()}/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    let detail = "Unknown backend error";
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // Keep default error detail.
    }
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }

  return (await response.json()) as ChatResponse;
}
