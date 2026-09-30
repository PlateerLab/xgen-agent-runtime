# Cross-Platform Session 통합 브랜치

- 기준: `main` (`35b7ec422ae85234c98e12c89794b74fdcc658cf`)
- 상태: Draft. 서버 계약과 격리·회귀 검증이 끝날 때까지 `main`에 병합하지 않는다.

## Runtime 작업 경계

1. Runtime 내부 실행 `session_id`와 서버가 발급하는 Agent Session ID를 구분하고, 실행·재시작·도구 호출 경계에서 사용자 및 Agent Session 문맥을 명시적으로 전달한다.
2. 개인 credential은 사용자·실행별 참조로만 전달한다. 로그, trace, prompt, 도구 결과와 subprocess 환경으로 원문이 새지 않는지 검증한다.
3. 도구 실행의 capability 대상, 승인과 취소 신호를 서버 계약에 맞춰 처리하고 다른 사용자·기기 문맥이 섞이지 않도록 격리 테스트를 추가한다.

첫 하위 작업은 검증된 호스트가 제공하는 `CanonicalSessionIdentity`를 `ToolContext`에 보관하고 Stage 10의 호출별 문맥 재구성에서 유지한다. `PipelineState.session_id`는 로컬 실행 ID로 남으며, canonical Agent Session ID와 혼동하지 않는다. 기존 호스트의 기본값은 `None`이다.

이 타입은 전달 용도이며 인증을 수행하지 않는다. Workflow는 Gateway principal과 Agent Session 소유권을 검증한 뒤에만 값을 주입해야 한다. 개인 credential, secret lease, capability 권한 판단과 실제 Workflow 주입은 후속 작업으로 남아 있다.
