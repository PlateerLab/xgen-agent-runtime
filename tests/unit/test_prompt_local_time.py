"""모델에게 알려 주는 "지금" 은 배포가 정한 시간대여야 한다 (4.50.0).

고정 UTC 였을 때 생기는 구멍: 한국(UTC+9) 사용자가 **자정~오전 9시**에 "오늘" 을
물으면 모델은 어제 날짜를 들고 일한다 — 하루 중 9시간. 시차가 있는 어느 배포에서나
오프셋만큼 같은 구멍이 생긴다.

시간대 **값**은 엔진이 정하지 않는다(배포마다 다르다). 엔진은 프로세스 지역 시각을
읽고 어느 시간대인지 이름·오프셋을 함께 말한다.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pytest

from xgen_agent_runtime.stages.s03_system.artifact.default.builders import DateTimeBlock


def _render_with_tz(tz: str | None) -> str:
    old = os.environ.get("TZ")
    try:
        if tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = tz
        time.tzset()
        return DateTimeBlock().render(None)
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="POSIX only")
class TestDateTimeBlock:
    def test_names_the_zone_and_the_offset(self):
        out = _render_with_tz("Asia/Seoul")
        assert "KST" in out and "UTC+09:00" in out

    def test_utc_is_not_said_twice(self):
        out = _render_with_tz("UTC")
        assert out.count("UTC") == 1, out

    def test_the_date_follows_the_deployment_zone(self):
        """같은 순간이라도 시간대가 다르면 날짜가 다를 수 있다 — 그게 요점이다."""
        seoul = _render_with_tz("Asia/Seoul")
        honolulu = _render_with_tz("Pacific/Honolulu")  # UTC-10, 서울과 19시간 차
        assert seoul != honolulu
        # 어느 쪽도 UTC 벽시계를 그대로 쓰지 않는다.
        utc_stamp = datetime.now(timezone.utc).strftime("%H:%M")
        assert not (utc_stamp in seoul and utc_stamp in honolulu)

    def test_still_works_without_tz_set(self):
        out = _render_with_tz(None)
        assert out.startswith("Current date: ")
        assert len(out) > len("Current date: ")

    def test_block_stays_volatile(self):
        """캐시된 프리픽스에 들어가면 1분마다 전체 재프리필이 된다."""
        assert DateTimeBlock().volatile is True
