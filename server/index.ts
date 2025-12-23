import http from 'http';
import { URL } from 'url';
import { Readable } from 'stream';
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangGraphHttpAgent } from "@ag-ui/langgraph";

// 1. Service adapter for multi-agent support
const serviceAdapter = new ExperimentalEmptyAdapter();

// 2. Create the CopilotRuntime instance and utilize the LangGraph AG-UI
//    integration to setup the connection.
const runtime = new CopilotRuntime({
  agents: {
    "sample_agent": new LangGraphHttpAgent({
      url: process.env.AGENT_URL || "http://localhost:8123",
    }),
  }
});

// 3. Get the request handler from CopilotKit
const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
  runtime,
  serviceAdapter,
  endpoint: "/api/copilotkit",
});

const PORT = process.env.PORT || 3001;

const server = http.createServer(async (req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // Only handle POST requests to /api/copilotkit
  if (req.method === 'POST' && req.url === '/api/copilotkit') {
    try {
      // Convert Node.js request to Web API Request
      const url = new URL(req.url || '/', `http://${req.headers.host}`);
      const body = await readRequestBody(req);
      
      const webRequest = new Request(url, {
        method: req.method,
        headers: new Headers(req.headers as Record<string, string>),
        body: body || undefined,
      });

      // Use CopilotKit's request handler
      const webResponse = await handleRequest(webRequest as any);

      // Convert Web API Response to Node.js response (supports streaming)
      const headers: Record<string, string | string[]> = {};
      webResponse.headers.forEach((value, key) => {
        const existing = headers[key];
        if (existing === undefined) headers[key] = value;
        else if (Array.isArray(existing)) existing.push(value);
        else headers[key] = [existing, value];
      });

      res.writeHead(webResponse.status, headers);

      if (webResponse.body) {
        // Pipe the response stream to the client (required for @stream/@defer)
        Readable.fromWeb(webResponse.body as any).pipe(res);
        return;
      }

      res.end(await webResponse.text());
    } catch (error) {
      console.error("Error in CopilotKit API route:", error);
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: "Internal server error",
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined
      }));
    }
  } else if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: "ok", service: "copilotkit-runtime" }));
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: "Not found" }));
  }
});

function readRequestBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', chunk => {
      body += chunk.toString();
    });
    req.on('end', () => {
      resolve(body);
    });
    req.on('error', reject);
  });
}

server.listen(PORT, () => {
  console.log(`🚀 CopilotKit runtime server running on http://localhost:${PORT}`);
  console.log(`📡 Agent URL: ${process.env.AGENT_URL || "http://localhost:8123"}`);
});

