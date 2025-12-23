# Migration from Next.js to Vite

This project has been migrated from Next.js to Vite with an Express backend server.

## Architecture Changes

- **Frontend**: Vite + React (replaces Next.js)
- **Backend**: Express server (replaces Next.js API routes)
- **Agent**: LangGraph agent (unchanged)

## Project Structure

```
├── index.html              # Vite entry point
├── vite.config.ts          # Vite configuration
├── server/
│   └── index.ts           # Express backend server
├── src/
│   ├── main.tsx           # React entry point
│   ├── App.tsx            # Main app component
│   └── globals.css        # Global styles
└── agent/                 # LangGraph agent (unchanged)
```

## Installation

1. Install dependencies:
```bash
npm install
```

2. Install agent dependencies:
```bash
npm run install:agent
```

## Running the Application

### Development Mode

Start all services (UI, server, and agent):
```bash
npm run dev
```

This will start:
- **Vite dev server** on `http://localhost:3000` (frontend)
- **Express server** on `http://localhost:3001` (API backend)
- **LangGraph agent** on `http://localhost:8123` (agent)

### Individual Services

- Frontend only: `npm run dev:ui`
- Backend only: `npm run dev:server`
- Agent only: `npm run dev:agent`

## Key Changes

### 1. API Endpoint

The CopilotKit runtime endpoint is now served by Express instead of Next.js API routes:
- **Before**: `/api/copilotkit` (Next.js API route)
- **After**: `http://localhost:3001/api/copilotkit` (Express server)

Vite proxies `/api/*` requests to the Express server automatically.

### 2. Frontend Entry Point

- **Before**: Next.js App Router (`src/app/layout.tsx` + `src/app/page.tsx`)
- **After**: Vite + React (`index.html` → `src/main.tsx` → `src/App.tsx`)

### 3. Build Process

- **Before**: `next build` → `.next/` directory
- **After**: `vite build` → `dist/` directory

## Environment Variables

Create a `.env` file in the root directory:
```env
AGENT_URL=http://localhost:8123
PORT=3001
```

## Production Deployment

1. Build the frontend:
```bash
npm run build
```

2. Start the Express server (which can also serve the built frontend):
```bash
npm start
```

Note: You may need to configure the Express server to serve static files from the `dist` directory in production.

## Troubleshooting

### Port Conflicts

If ports 3000, 3001, or 8123 are already in use, you can change them:
- Vite: Update `vite.config.ts` `server.port`
- Express: Set `PORT` environment variable
- Agent: Update `agent/sample_agent/demo.py`

### CORS Issues

CORS is enabled in the Express server. If you encounter issues, check the `cors()` middleware in `server/index.ts`.

### Agent Connection

Ensure the agent is running before starting the frontend. The Express server will log connection errors if it can't reach the agent.

