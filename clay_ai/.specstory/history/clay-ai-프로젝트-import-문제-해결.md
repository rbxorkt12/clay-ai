## SpecStory

## clay_ai 프로젝트 import 문제 해결 (2025. 1. 25. 오후 7:26:23)

_**User**_

지금 플젝이 clay_ai 인데 이렇게 하면 위치를 못잡아. 어떻게 고칠수 있어?

---

_**Assistant**_

먼저 folder_structure.yaml을 확인해야 할 것 같은데요, 현재는 보이지 않네요. 하지만 일반적으로 이런 import 문제는 프로젝트 구조와 관련이 있습니다.

문제는 `clay_ai.models.tasks`를 import 하는 부분인데, 현재 프로젝트가 clay_ai 프로젝트라면 이 import 문은 수정이 필요합니다. 프로젝트 내부에서는 상대 경로로 import하는 것이 좋습니다.

다음과 같이 수정하는 것을 제안드립니다:

```python:api/linter.py
// ... existing code ...
from typing import List, Optional
from fastapi import APIRouter, HTTPException

from agents.linter import LinterAgent
from models.linter import BulkLinterResult
from models.agents import AgentConfig
from models.tasks import Task, TaskResult  # clay_ai.models.tasks에서 models.tasks로 변경
// ... existing code ...
```

이렇게 수정하면:
1. 프로젝트 내부의 models 패키지에서 직접 Task와 TaskResult를 import 하게 됩니다.
2. 프로젝트의 구조를 더 잘 반영하게 됩니다.
3. 절대 경로 대신 상대 경로를 사용하여 import 문제를 해결할 수 있습니다.

만약 정확한 import 경로를 확인하고 싶다면, folder_structure.yaml 파일을 공유해 주시면 더 정확한 경로를 제안드릴 수 있습니다.

---



---

