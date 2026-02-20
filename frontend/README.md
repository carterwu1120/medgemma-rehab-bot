# MedGemma Frontend

Next.js frontend for the MedGemma rehab backend.

## Prerequisites

- Backend API running at `http://127.0.0.1:9000`
- Node.js 20+

## Local Run

1. Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:9000
```

2. Start frontend:

```bash
npm run dev
```

3. Open `http://localhost:3000`

## Features

- Send rehab query to `POST /v1/chat`
- Show answer with policy notes and chunk references
- Show retrieved evidence chunks
- Show recommended videos from backend
