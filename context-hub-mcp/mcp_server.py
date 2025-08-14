#!/usr/bin/env python3
"""
Standalone MCP server for Context Hub.

This script runs the MCP server independently from the FastAPI application,
suitable for integration with Cursor, Claude Desktop, or other MCP clients.

Follows fastMCP guide patterns for maximum compatibility.
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path (상대 경로 기반)
backend_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(backend_dir))

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = backend_dir / '.env'
load_dotenv(env_path)

from fastmcp import FastMCP
import time
import httpx
from typing import Any

# FastAPI server configuration
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")

# Timeout constants for different operations
SEARCH_TIMEOUT = 2.0  # Fast search operations
CRUD_TIMEOUT = 5.0    # Create, Read, Update, Delete operations
RULE_INIT_TIMEOUT = 10.0  # Rule initialization (involves file I/O)

# Response limits
MAX_SEARCH_LIMIT = 100
DEFAULT_SEARCH_LIMIT = 20

# Create FastMCP server instance
mcp = FastMCP("Context Hub")

def format_content_for_api(content: str | dict[str, Any]) -> dict[str, Any]:
    """
    Content를 백엔드 API가 기대하는 표준 형식으로 변환합니다.
    
    Args:
        content: 원본 content (문자열 또는 딕셔너리)
        
    Returns:
        백엔드 API 호환 형식의 content 딕셔너리
    """
    if isinstance(content, str):
        # 문자열인 경우 백엔드가 기대하는 형식으로 구조화
        return {
            "content_type": "markdown",
            "raw_content": content,
            "text": content,  # 프론트엔드 호환성을 위해 추가
            "format": "markdown",
            "size": len(content.encode('utf-8'))
        }
    else:
        # 이미 딕셔너리인 경우 content_type이 없으면 추가
        formatted_content = content.copy() if isinstance(content, dict) else content
        if isinstance(formatted_content, dict):
            if "content_type" not in formatted_content:
                formatted_content["content_type"] = "markdown"
            # 프론트엔드 호환성을 위해 text 필드가 없으면 raw_content에서 복사
            if "text" not in formatted_content and "raw_content" in formatted_content:
                formatted_content["text"] = formatted_content["raw_content"]
        return formatted_content

def create_error_response(start_time: float, operation: str, error_type: str, message: str, **kwargs) -> dict[str, Any]:
    """공통 에러 응답 생성 함수"""
    response_time_ms = int((time.time() - start_time) * 1000)
    return {
        "success": False,
        "operation": operation,
        "response_time_ms": response_time_ms,
        "error": error_type,
        "message": message,
        **kwargs
    }

@mcp.tool(name="get-context")
async def get_context(
    context_id: str | None = None,
) -> dict[str, Any]:
    """
    컨텍스트 허브에서 `context_id`로 단일 컨텍스트를 조회합니다.
    
    토큰 사용과 불필요한 데이터 전송을 최소화하기 위해, 이 도구는
    ID 기반의 단일 컨텍스트 조회만 지원합니다.
    
    매개변수
    - context_id: 조회할 컨텍스트의 ID (UUID 형식)
    
    반환
    - contexts: 컨텍스트 목록(길이 0 또는 1)
    - total_count: 조회된 컨텍스트 개수
    - response_time_ms: 총 응답 시간(ms)
    - success: 성공 여부
    - error: 오류 메시지(실패 시)
    """
    start_time = time.time()

    try:
        # Validate input
        if not context_id:
            response_time_ms = int((time.time() - start_time) * 1000)
            return {
                "contexts": [],
                "total_count": 0,
                "response_time_ms": response_time_ms,
                "success": False,
                "error": "context_id는 필수 입력값입니다 (get-context)"
            }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{FASTAPI_BASE_URL}/api/v1/contexts/{context_id}",
                timeout=2.0
            )
            
            if response.status_code == 200:
                context_data = response.json()
                contexts = [context_data]
                total_count = 1
            elif response.status_code == 404:
                contexts = []
                total_count = 0
            else:
                response.raise_for_status()

        response_time_ms = int((time.time() - start_time) * 1000)

        # Return MCP-compatible response format
        return {
            "contexts": contexts,
            "total_count": total_count,
            "response_time_ms": response_time_ms,
            "success": True
        }

    except httpx.TimeoutException:
        return create_error_response(start_time, None, "timeout", "FastAPI server timeout (>2 seconds)", 
                                   contexts=[], total_count=0)
    except httpx.ConnectError:
        return create_error_response(start_time, None, "connection_error", 
                                   f"Cannot connect to FastAPI server at {FASTAPI_BASE_URL}",
                                   contexts=[], total_count=0)
    except Exception as e:
        return create_error_response(start_time, None, "api_error", f"API call failed: {str(e)}",
                                   contexts=[], total_count=0)

@mcp.tool(name="add-context")
async def upsert_context_hub(
    name: str,
    content: str | dict[str, Any],
    context_type: str = "document",
    description: str | None = None,
    version: str | None = None,
    document_ids: list[str] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    is_active: bool = True,
    created_by: str | None = None
) -> dict[str, Any]:
    """
    컨텍스트 허브에 새로운 컨텍스트를 생성합니다.
    
    이 도구는 FastAPI의 POST /api/v1/contexts/ 엔드포인트를 호출합니다.
    서버에서 중복 검증·비즈니스 규칙·버전 관리를 수행합니다.
    
    매개변수
    - name: 컨텍스트 이름 (필수)
    - content: 컨텍스트 내용 (문자열 또는 JSON)
    - context_type: 컨텍스트 타입 (기본값: document)
    - description: 컨텍스트 설명
    - version: 버전 (기본값: "1.0.0")
    - document_ids: 연결된 문서 ID 목록
    - session_id: 세션 ID
    - user_id: 사용자 ID
    - metadata: 추가 메타데이터
    - is_active: 활성화 상태
    - created_by: 생성자 ID
    - updated_by: 수정자 ID (사용되지 않음)
    
    반환
    - success: 성공 여부
    - operation: 수행된 작업("created")
    - context: 생성된 컨텍스트 객체
    - context_id: 생성된 컨텍스트 ID
    - final_name: 최종 이름
    - response_time_ms: 총 응답 시간(ms)
    - message: 사용자 친화적 메시지
    """
    start_time = time.time()
    
    try:
        # API 요청 데이터 구성
        # content를 백엔드 API 호환 형식으로 변환
        formatted_content = format_content_for_api(content)
            
        create_data = {
            "name": name,
            "content": formatted_content,
            "context_type": context_type,
            "version": version or "1.0.0",
            "is_active": is_active
        }
        
        # 선택적 필드들 추가
        if description is not None:
            create_data["description"] = description
        if document_ids is not None:
            create_data["document_ids"] = document_ids
        if session_id is not None:
            create_data["session_id"] = session_id
        if user_id is not None:
            create_data["user_id"] = user_id
        if metadata is not None:
            create_data["metadata"] = metadata
        if created_by is not None:
            create_data["created_by"] = created_by
                
        # FastAPI 엔드포인트 호출
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/api/v1/contexts/",
                json=create_data,
                timeout=CRUD_TIMEOUT
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 201:
                created_context = response.json()
                return {
                    "success": True,
                    "operation": "created",
                    "context": created_context,
                    "context_id": created_context.get("id"),
                    "final_name": created_context.get("name", name),
                    "response_time_ms": response_time_ms,
                    "message": f"Context '{created_context.get('name', name)}' was successfully created"
                }
            else:
                # API 오류 처리
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", str(response.text))
                except Exception:
                    error_detail = response.text
                
                return {
                    "success": False,
                    "operation": None,
                    "context": None,
                    "response_time_ms": response_time_ms,
                    "error": f"API error {response.status_code}: {error_detail}",
                    "message": f"컨텍스트 생성 중 오류가 발생했습니다: {error_detail}"
                }
    
    except httpx.TimeoutException:
        return create_error_response(start_time, None, "timeout", "요청 시간이 초과되었습니다", 
                                   context=None)
    except httpx.ConnectError:
        return create_error_response(start_time, None, "connection_error", 
                                   f"FastAPI 서버({FASTAPI_BASE_URL})에 연결할 수 없습니다",
                                   context=None)
    except Exception as e:
        return create_error_response(start_time, None, "creation_failed", 
                                   f"컨텍스트 생성 중 오류가 발생했습니다: {str(e)}",
                                   context=None)

@mcp.tool(name="update-context")
async def update_context_hub(
    context_id: str,
    name: str | None = None,
    content: str | dict[str, Any] | None = None,
    context_type: str | None = None,
    description: str | None = None,
    version: str | None = None,
    document_ids: list[str] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    is_active: bool | None = None,
    updated_by: str | None = None
) -> dict[str, Any]:
    """
    컨텍스트 허브에서 기존 컨텍스트를 수정합니다.
    
    FastAPI의 PUT /api/v1/contexts/{context_id} 엔드포인트를 호출합니다.
    서버에서 존재 여부 확인, 유효성 검증, 버전 관리를 처리합니다.
    
    매개변수
    - context_id: 수정할 컨텍스트 ID (필수)
    - name: 새로운 컨텍스트 이름
    - content: 새로운 컨텍스트 내용
    - context_type: 새로운 컨텍스트 타입
    - description: 새로운 설명
    - version: 새로운 버전
    - document_ids: 새로운 문서 ID 목록
    - session_id: 새로운 세션 ID
    - user_id: 새로운 사용자 ID
    - metadata: 새로운 메타데이터
    - is_active: 새로운 활성화 상태
    - updated_by: 수정자 ID
    
    반환
    - success: 성공 여부
    - operation: 수행된 작업("updated")
    - context: 수정된 컨텍스트 객체
    - context_id: 컨텍스트 ID
    - final_name: 최종 이름
    - response_time_ms: 총 응답 시간(ms)
    - message: 사용자 친화적 메시지
    """
    start_time = time.time()
    
    try:
        # 수정할 데이터만 포함한 요청 데이터 구성
        update_data = {}
        
        if name is not None:
            update_data["name"] = name
            
        if content is not None:
            # content를 백엔드 API 호환 형식으로 변환
            formatted_content = format_content_for_api(content)
            update_data["content"] = formatted_content
            
        if context_type is not None:
            update_data["context_type"] = context_type
        if description is not None:
            update_data["description"] = description
        if version is not None:
            update_data["version"] = version
        if document_ids is not None:
            update_data["document_ids"] = document_ids
        if session_id is not None:
            update_data["session_id"] = session_id
        if user_id is not None:
            update_data["user_id"] = user_id
        if metadata is not None:
            update_data["metadata"] = metadata
        if is_active is not None:
            update_data["is_active"] = is_active
        if updated_by is not None:
            update_data["updated_by"] = updated_by
        
        # FastAPI 엔드포인트 호출
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{FASTAPI_BASE_URL}/api/v1/contexts/{context_id}",
                json=update_data,
                timeout=CRUD_TIMEOUT
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                updated_context = response.json()
                return {
                    "success": True,
                    "operation": "updated",
                    "context": updated_context,
                    "context_id": updated_context.get("id"),
                    "final_name": updated_context.get("name"),
                    "response_time_ms": response_time_ms,
                    "message": f"Context '{updated_context.get('name')}' was successfully updated"
                }
            elif response.status_code == 404:
                return {
                    "success": False,
                    "operation": None,
                    "context": None,
                    "response_time_ms": response_time_ms,
                    "error": f"Context with ID '{context_id}' not found",
                    "message": f"컨텍스트 ID '{context_id}'를 찾을 수 없습니다."
                }
            else:
                # API 오류 처리
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", str(response.text))
                except Exception:
                    error_detail = response.text
                
                return {
                    "success": False,
                    "operation": None,
                    "context": None,
                "response_time_ms": response_time_ms,
                    "error": f"API error {response.status_code}: {error_detail}",
                    "message": f"컨텍스트 수정 중 오류가 발생했습니다: {error_detail}"
            }
    
    except httpx.TimeoutException:
        return create_error_response(start_time, None, "timeout", "요청 시간이 초과되었습니다", 
                                   context=None)
    except httpx.ConnectError:
        return create_error_response(start_time, None, "connection_error", 
                                   f"FastAPI 서버({FASTAPI_BASE_URL})에 연결할 수 없습니다",
                                   context=None)
    except Exception as e:
        return create_error_response(start_time, None, "update_failed", 
                                   f"컨텍스트 수정 중 오류가 발생했습니다: {str(e)}",
                                   context=None)

@mcp.tool(name="init-rule")
async def init_rule(
    file_name: str = "contexthub__mcp_rule.mdc",
    custom_content: str | None = None,
    overwrite: bool = False,
    project_root: str | None = None
) -> dict[str, Any]:
    """
    프로젝트의 .cursor/rules 디렉토리에 Context Hub MCP 룰 파일을 자동 생성합니다.
    
    이 도구는 Cursor IDE에서 Context Hub MCP 도구가 자동 호출되도록 하는 룰 파일을
    생성하여 개발 효율을 높입니다.
    
    매개변수
    - file_name: 생성할 룰 파일명 (기본값: contexthub__mcp_rule.mdc)
    - custom_content: 사용자 정의 룰 내용 (기본 템플릿을 사용하지 않을 때)
    - overwrite: 기존 파일 덮어쓰기 허용 (기본값: False)
    - project_root: 프로젝트 루트 디렉토리 경로 (기본값: 현재 작업 디렉토리)
    
    반환
    - success, operation, message, file_path, project_root, response_time_ms, metadata
    
    예시
    - 기본 룰 생성: await init_rule()
    - 특정 경로 생성: await init_rule(project_root="/path/to/project")
    - 파일명 지정: await init_rule(file_name="custom_rule.mdc")
    """
    start_time = time.time()
    
    try:
        # 프로젝트 루트 디렉토리 결정
        if project_root is None:
            # MCP 클라이언트의 현재 작업 디렉토리 사용
            project_root = os.getcwd()
        
        # 상대 경로를 정규화된 경로로 변환
        project_root = Path(project_root).resolve()
        
        # 컨텐츠 결정: 사용자 정의 또는 Context Hub MCP 템플릿
        if custom_content is None:
            # FastAPI에서 Context Hub MCP 템플릿 조회
            try:
                async with httpx.AsyncClient() as client:
                    template_response = await client.get(
                        f"{FASTAPI_BASE_URL}/api/v1/rules/template/contexthub_mcp",
                        timeout=CRUD_TIMEOUT
                    )
                
                if template_response.status_code == 200:
                    template_data = template_response.json()
                    content = template_data["content"]
                else:
                    # 템플릿 조회 실패 시 파일에서 로드 시도
                    raise Exception("Template not found via API")
                        
            except Exception as e:
                # 네트워크 오류 등으로 템플릿 조회 실패 시 파일에서 기본 내용 로드
                try:
                    # 백엔드 app 디렉토리의 템플릿 파일 경로
                    backend_dir = Path(__file__).parent  # mcp_server.py가 있는 디렉토리
                    template_path = backend_dir / "app" / "templates" / "rules" / "contexthub_mcp_rule.mdc"
                    content = template_path.read_text(encoding='utf-8')
                except Exception:
                    # 파일 읽기도 실패하면 최소한의 기본 내용 사용
                    content = "# Context Hub MCP Rule - Template not available"
        else:
            content = custom_content
        
        # FastAPI rules/init 엔드포인트 호출
        rules_init_request = {
            "file_name": file_name,
            "content": content,
            "path": ".cursor/rules",
            "overwrite": overwrite
        }
        
        # MCP 클라이언트의 작업 디렉토리를 항상 전달
        # (FastAPI 서버의 cwd와 MCP 클라이언트의 cwd가 다를 수 있음)
        rules_init_request["project_root"] = str(project_root)
        
        # 프로젝트 루트를 현재 작업 디렉토리로 설정하여 FastAPI 호출
        # 실제로는 FastAPI가 실행되는 환경에서 상대 경로가 처리되므로
        # 여기서는 요청 데이터만 전달
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{FASTAPI_BASE_URL}/api/v1/rules/init",
                json=rules_init_request,
                timeout=RULE_INIT_TIMEOUT
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "operation": "cursor_rule_created",
                    "message": f"✅ Cursor 룰 파일이 성공적으로 생성되었습니다: {file_name}",
                    "file_path": result.get("file_path", f".cursor/rules/{file_name}"),
                    "project_root": str(project_root),
                    "response_time_ms": response_time_ms,
                    "metadata": {
                        "file_name": file_name,
                        "content_length": len(content),
                        "template_used": "contexthub_mcp" if custom_content is None else "custom",
                        "fastapi_response": result
                    }
                }
            elif response.status_code == 409:
                # 파일이 이미 존재하는 경우
                return {
                    "success": False,
                    "operation": "cursor_rule_exists",
                    "message": f"⚠️  Cursor 룰 파일이 이미 존재합니다: {file_name}",
                    "file_path": f".cursor/rules/{file_name}",
                    "project_root": str(project_root),
                    "response_time_ms": response_time_ms,
                    "error": "File already exists",
                    "suggestion": "기존 파일을 삭제하거나 다른 파일명을 사용해 주세요."
                }
            else:
                # 기타 HTTP 오류
                error_detail = "Unknown error"
                try:
                    error_response = response.json()
                    error_detail = error_response.get("detail", str(response.text))
                except Exception:
                    error_detail = response.text
                
                return {
                    "success": False,
                    "operation": "cursor_rule_failed",
                    "message": f"❌ Cursor 룰 파일 생성 실패: {file_name}",
                    "project_root": str(project_root),
                    "response_time_ms": response_time_ms,
                    "error": f"HTTP {response.status_code}: {error_detail}",
                    "suggestion": "FastAPI 서버 상태를 확인하고 다시 시도해 주세요."
                }
        
    except httpx.TimeoutException:
        return create_error_response(start_time, "rule_timeout", "timeout", 
                                   "❌ 룰 파일 생성 시간 초과",
                                   project_root=str(project_root) if project_root else "unknown")
    except httpx.ConnectError:
        return create_error_response(start_time, "rule_connection_error", "connection_error",
                                   "❌ FastAPI 서버에 연결할 수 없습니다",
                                   project_root=str(project_root) if project_root else "unknown")
    except Exception as e:
        return create_error_response(start_time, "rule_error", "unexpected_error",
                                   f"❌ 룰 파일 생성 중 오류 발생: {str(e)}",
                                   project_root=str(project_root) if project_root else "unknown")

@mcp.tool(name="search-context-id")
async def search_context_id(
    query: str,
    limit: int = 20,
    context_type: str | None = None,
    is_active: bool | None = None
) -> dict[str, Any]:
    """
    키워드(자연어)로 컨텍스트를 검색합니다. (퍼지 검색)
    
    이름, 설명, 내용 전반을 대상으로 키워드 추출 기반의 퍼지 매칭을 수행합니다.
    관련도 기준으로 정렬되며, 하이라이트 정보가 포함됩니다.
    
    매개변수
    - query: 검색 문자열
    - limit: 최대 반환 개수 (기본 20, 최대 100)
    - context_type: 컨텍스트 타입 필터 ('document' | 'session' | 'user')
    - is_active: 활성 상태 필터 (true/false)
    
    반환
    - results: 검색 결과 목록
    - total_count: 총 결과 수
    - query: 실제 검색 쿼리
    - response_time_ms: 총 응답 시간(ms)
    - success: 성공 여부
    - error_message: 오류 메시지(실패 시)
    - highlighted_matches: 하이라이트 정보
    - search_metadata: 세부 메타데이터
    """
    start_time = time.time()

    try:
        # Validate query
        if not query or not query.strip():
            return {
                "results": [],
                "total_count": 0,
                "query": query,
                "response_time_ms": 0,
                "success": True,
                "error_message": "Empty query provided"
            }

        # Build query parameters
        params = {
            "q": query.strip(),
            "limit": min(max(1, limit), MAX_SEARCH_LIMIT)  # Ensure limit is between 1-100
        }
        
        # Add optional filters
        if context_type:
            params["context_type"] = context_type
        if is_active is not None:
            params["is_active"] = is_active

        # Call the fuzzy search API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{FASTAPI_BASE_URL}/api/v1/search",
                params=params,
                timeout=SEARCH_TIMEOUT
            )
            response.raise_for_status()
            
            search_data = response.json()
            response_time_ms = int((time.time() - start_time) * 1000)

            # Return MCP-compatible response format
            return {
                "results": search_data.get("results", []),
                "total_count": search_data.get("total_count", 0),
                "query": search_data.get("query", query),
                "response_time_ms": response_time_ms,
                "success": search_data.get("success", True),
                "error_message": search_data.get("error_message"),
                "highlighted_matches": search_data.get("highlighted_matches", []),
                "search_metadata": {
                    "api_response_time_ms": search_data.get("response_time_ms", 0),
                    "total_response_time_ms": response_time_ms,
                    "keyword_extraction": "API handles automatic keyword splitting",
                    "fuzzy_matching_enabled": True
                }
            }

    except httpx.TimeoutException:
        return create_error_response(start_time, None, "timeout", "FastAPI server timeout (>2 seconds)",
                                   results=[], total_count=0, query=query, error_message="Timeout occurred")
    except httpx.ConnectError:
        return create_error_response(start_time, None, "connection_error", 
                                   f"Cannot connect to FastAPI server at {FASTAPI_BASE_URL}",
                                   results=[], total_count=0, query=query, error_message="Connection failed")
    except Exception as e:
        return create_error_response(start_time, None, "search_failed", f"Search API call failed: {str(e)}",
                                   results=[], total_count=0, query=query, error_message=str(e))

if __name__ == "__main__":
    mcp.run()