# Trace / Signal frontend

Next.js App Router frontend for the spiral and wave screening prototype.

## Local development

```powershell
cd frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

With the default environment, the app uses a deterministic local demo adapter. Set `NEXT_PUBLIC_API_URL` to the FastAPI base URL and `NEXT_PUBLIC_USE_MOCK_API=false` to use the API contract from Phase 7.

## Verification

```powershell
npm run typecheck
npm run lint
npm test
npm run build
```

The drawing canvas records pointer coordinates, timing, pressure, pointer type, and pen-lift markers in memory. Only result metadata is retained in browser storage for the history view. Raw images and raw stroke points are not persisted by this frontend.
