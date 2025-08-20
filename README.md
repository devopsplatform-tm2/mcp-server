# Context Hub MCP

Context Hub용 MCP 서버(STDIO). 다른 AI 에이전트가 STDIO 방식으로 실행해 사용할 수 있습니다.

## 실행 (uvx 원격 실행)

```bash
uvx --from git+https://github.com/devopsplatform-tm2/mcp-server.git@master context-hub-mcp
```

백엔드 FastAPI 주소 지정:

```bash
uvx --from git+https://github.com/devopsplatform-tm2/mcp-server.git@master context-hub-mcp
```

## 로컬 실행

```bash
uv run --with fastmcp --with httpx --with python-dotenv context_hub_mcp/mcp_server.py
```

엔트리포인트 스크립트:
- `context-hub-mcp` → `context_hub_mcp.mcp_server:main`

필수 환경변수:
- `FASTAPI_BASE_URL` (기본: `http://localhost:8000`)

라이선스: MIT
