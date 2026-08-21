"""_creds_to_client_kwargs — 새 provider 들의 자격증명 매핑 계약.

Bedrock/Vertex 의 다중 필드 자격증명은 ``extras`` 로 나른다 — else 분기로
떨어지면 region/project 가 조용히 소실된다 (감사 함정 #3).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from xgen_agent_runtime.core.pipeline import _creds_to_client_kwargs
from xgen_agent_runtime.llm_client.credentials import ProviderCredentials


def test_bedrock_extras_reach_the_constructor():
    creds = ProviderCredentials(
        api_key="",
        extras={
            "aws_region": "ap-northeast-2",
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "s3",
        },
    )
    kwargs = _creds_to_client_kwargs("bedrock", creds)
    assert kwargs["aws_region"] == "ap-northeast-2"
    assert kwargs["aws_access_key_id"] == "AKIA"
    assert "api_key" not in kwargs


def test_bedrock_region_only_is_valid_config():
    """키 없는(IAM role) 배포 — region 만으로 구성이 성립한다."""
    creds = ProviderCredentials(api_key="", extras={"aws_region": "us-east-1"})
    kwargs = _creds_to_client_kwargs("bedrock", creds)
    assert kwargs == {"aws_region": "us-east-1"}


def test_vertex_project_location_and_sa_json():
    creds = ProviderCredentials(
        api_key="",
        extras={
            "project": "p-1",
            "location": "asia-northeast3",
            "credentials_json": "{}",
        },
    )
    kwargs = _creds_to_client_kwargs("vertex", creds)
    assert kwargs == {
        "project": "p-1",
        "location": "asia-northeast3",
        "credentials_json": "{}",
    }


def test_vertex_express_key_travels_as_api_key():
    creds = ProviderCredentials(api_key="express")
    kwargs = _creds_to_client_kwargs("vertex", creds)
    assert kwargs == {"api_key": "express"}


def test_codex_cli_mirrors_the_cli_extras_contract():
    creds = ProviderCredentials(
        api_key="sk-o",
        auth_mode="api_key",
        extras={
            "workspace_root": "/ws",
            "sandbox_mode": "read-only",
            "bypass_sandbox": True,
            "mcp_config": {"mcpServers": {}},
            "timeout_s": 60.0,
        },
    )
    kwargs = _creds_to_client_kwargs("codex_cli", creds)
    assert kwargs["api_key"] == "sk-o"
    assert kwargs["auth_mode"] == "api_key"
    assert kwargs["workspace_dir"] == "/ws"   # settings 명 → 생성자 명
    assert kwargs["sandbox_mode"] == "read-only"
    assert kwargs["bypass_sandbox"] is True
