import os
import time
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------

# Базовый URL OpenAI-совместимого сервера SGR
SGR_BASE_URL = os.getenv("SGR_BASE_URL", "http://localhost:8010")

# Полный путь до /v1/chat/completions в SGR
SGR_CHAT_URL = os.getenv(
    "SGR_CHAT_URL",
    f"{SGR_BASE_URL}/v1/chat/completions",
)

# Полный путь до /v1/models в SGR
SGR_MODELS_URL = os.getenv(
    "SGR_MODELS_URL",
    f"{SGR_BASE_URL}/v1/models",
)

# Имя модели по умолчанию (должно совпадать с агентом в /v1/models SGR)
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "sgr_agent")

# Параметры самого прокси
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))

app = FastAPI(title="SGR → OpenAI proxy")


# ---------------------------------------------------------
# /v1/models — проксирование списка моделей SGR
# ---------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.get(SGR_MODELS_URL)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return JSONResponse(
                {
                    "error": {
                        "message": f"SGR /v1/models error {exc.response.status_code}: {exc.response.text}",
                        "type": "sgr_models_error",
                    }
                },
                status_code=exc.response.status_code,
            )
        return JSONResponse(resp.json())


# ---------------------------------------------------------
# Вспомогательные функции OpenAI-формата
# ---------------------------------------------------------

def make_chunk(content: str, *, first: bool = False, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Собираем OpenAI-совместимый chat.completion.chunk.
    """
    delta: Dict[str, Any] = {"content": content}
    if first:
        delta["role"] = "assistant"

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or OPENAI_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": None,
            }
        ],
    }


def make_final_response(full_content: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Нестриминговый ответ в формате chat.completion.
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or OPENAI_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ---------------------------------------------------------
# Разбор JSON-объектов из reasoning
# ---------------------------------------------------------

def extract_answer_from_obj(obj: Dict[str, Any]) -> Any:
    """
    Ищем answer / final_answer в объекте SGR.
    """
    if "answer" in obj:
        return obj["answer"]
    if "final_answer" in obj:
        return obj["final_answer"]

    func = obj.get("function")
    if isinstance(func, dict):
        if "answer" in func:
            return func["answer"]
        if "final_answer" in func:
            return func["final_answer"]

    return None


def normalize_answer_text(answer: Any) -> str:
    """
    Превращаем answer (строка или dict) в человекочитаемый текст.
    """
    if answer is None:
        return ""

    if isinstance(answer, str):
        s = answer.strip()
        if s.startswith("{"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    t = obj.get("title")
                    if isinstance(t, str):
                        return t
                    return json.dumps(obj, ensure_ascii=False)
                return s
            except json.JSONDecodeError:
                return s
        return s

    if isinstance(answer, dict):
        t = answer.get("title")
        if isinstance(t, str):
            return t
        return json.dumps(answer, ensure_ascii=False)

    return str(answer)


def split_json_objects(s: str) -> List[Dict[str, Any]]:
    """
    Из строки с несколькими JSON-объектами подряд достаём список dict.
    """
    objs: List[Dict[str, Any]] = []
    depth = 0
    in_string = False
    escape = False
    start = None

    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = s[start : i + 1]
                try:
                    obj = json.loads(fragment)
                    objs.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    return objs


def extract_final_message_from_full(full_content: str) -> Tuple[str, bool]:
    """
    Из полного JSON-лога SGR достаём:
      - текст для пользователя (ответ или вопрос),
      - флаг expects_clarification:
          True  → агент ждёт уточнения (clarificationtool, без финального ответа);
          False → диалог завершён финальным ответом.

    Если нет ни того, ни другого — возвращаем ("", False).
    """
    text = full_content.strip()
    if not text.startswith("{"):
        return "", False

    objs = split_json_objects(text)

    final_answer_text = ""
    clarification_text = ""
    has_final = False
    has_clar = False

    for obj in objs:
        func = obj.get("function") or {}
        discr = func.get("tool_name_discriminator") or obj.get("tool_name_discriminator")
        status = func.get("status") or obj.get("status")

        # Финальный ответ
        if discr == "finalanswertool" or status == "completed":
            ans = extract_answer_from_obj(obj)
            if ans is not None:
                final_answer_text = normalize_answer_text(ans)
                has_final = True

        # Уточнение
        if discr == "clarificationtool":
            questions = obj.get("questions") or func.get("questions")
            if isinstance(questions, list) and questions:
                clarification_text = "\n\n".join(str(q) for q in questions)
                has_clar = True

    if has_final and final_answer_text:
        # Есть финал — это обычный завершённый ответ, никакого agent_id потом не надо
        return final_answer_text, False

    if has_clar and clarification_text:
        # Есть только уточнение — ждём ответ пользователя
        return clarification_text, True

    return "", False


# ---------------------------------------------------------
# Вытаскиваем agent_id из последнего assistant-сообщения
# ---------------------------------------------------------

def get_agent_id_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Ищем маркер вида <!--sgr_agent_id:...--> только в ПОСЛЕДНЕМ
    assistant-сообщении. Если там маркера нет — считаем, что это новый запрос.
    """
    marker = "<!--sgr_agent_id:"
    # идём с конца, берём первый assistant и только его
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            return None
        pos = content.rfind(marker)
        if pos == -1:
            return None
        start = pos + len(marker)
        end = content.find("-->", start)
        if end == -1:
            return None
        return content[start:end].strip()
    return None


# ---------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    messages: List[Dict[str, Any]] = body.get("messages", []) or []
    requested_model = body.get("model", OPENAI_MODEL_NAME)
    stream = body.get("stream", False)

    # если в последнем ответе ассистента есть маркер — продолжаем этого агента
    agent_id = get_agent_id_from_messages(messages)
    true_model = agent_id or requested_model

    if stream:
        # ---- STREAM MODE ----
        async def event_generator():
            async with httpx.AsyncClient(timeout=None) as client:
                payload = dict(body)
                payload["model"] = true_model
                payload["stream"] = True

                buffer_parts: List[str] = []
                thinking_opened = False
                first_chunk = True
                current_agent_model_id: Optional[str] = None

                async with client.stream("POST", SGR_CHAT_URL, json=payload) as resp:
                    if resp.status_code != 200:
                        error_bytes = await resp.aread()
                        try:
                            error_text = error_bytes.decode("utf-8")
                        except Exception:
                            error_text = str(error_bytes)
                        print("SGR stream error:", resp.status_code, error_text)
                        chunk = make_chunk(
                            f"[SGR error {resp.status_code}] {error_text}",
                            first=True,
                            model=requested_model,
                        )
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: ") :]
                        if line.strip() == "[DONE]":
                            break

                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        current_agent_model_id = obj.get("model", current_agent_model_id)

                        choices = obj.get("choices")
                        if not isinstance(choices, list) or not choices:
                            continue
                        c0 = choices[0] or {}
                        delta = c0.get("delta") or {}
                        part = delta.get("content")
                        if not isinstance(part, str) or not part:
                            continue

                        buffer_parts.append(part)

                        if not thinking_opened:
                            thinking_opened = True
                            open_tag = "<thinking>\n"
                            open_chunk = make_chunk(open_tag, first=True, model=requested_model)
                            yield f"data: {json.dumps(open_chunk, ensure_ascii=False)}\n\n"
                            first_chunk = False

                        reasoning_chunk = make_chunk(part, first=False, model=requested_model)
                        yield f"data: {json.dumps(reasoning_chunk, ensure_ascii=False)}\n\n"

                full_content = "".join(buffer_parts)

                if thinking_opened:
                    close_tag = "\n</thinking>\n\n"
                    close_chunk = make_chunk(close_tag, first=False, model=requested_model)
                    yield f"data: {json.dumps(close_chunk, ensure_ascii=False)}\n\n"

                final_message, expects_clarification = extract_final_message_from_full(full_content)

                # Маркер agent_id добавляем ТОЛЬКО если агент ждёт уточнения
                if expects_clarification and current_agent_model_id and final_message:
                    final_message = (
                        f"{final_message}\n\n<!--sgr_agent_id:{current_agent_model_id}-->"
                    )

                if final_message:
                    answer_chunk = make_chunk(
                        final_message,
                        first=not thinking_opened,
                        model=requested_model,
                    )
                    yield f"data: {json.dumps(answer_chunk, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---- NON-STREAM MODE ----
    async with httpx.AsyncClient(timeout=None) as client:
        payload = dict(body)
        payload["model"] = true_model
        payload["stream"] = True
        parts: List[str] = []
        current_agent_model_id: Optional[str] = None

        async with client.stream("POST", SGR_CHAT_URL, json=payload) as resp:
            if resp.status_code != 200:
                error_bytes = await resp.aread()
                try:
                    error_text = error_bytes.decode("utf-8")
                except Exception:
                    error_text = str(error_bytes)
                print("SGR non-stream error:", resp.status_code, error_text)
                return JSONResponse(
                    {
                        "error": {
                            "message": f"SGR error {resp.status_code}: {error_text}",
                            "type": "sgr_chat_error",
                        }
                    },
                    status_code=resp.status_code,
                )

            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[len("data: ") :]
                if line.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                current_agent_model_id = obj.get("model", current_agent_model_id)

                choices = obj.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                c0 = choices[0] or {}
                delta = c0.get("delta") or {}
                part = delta.get("content")
                if isinstance(part, str):
                    parts.append(part)

    full_text = "".join(parts)
    final_message, expects_clarification = extract_final_message_from_full(full_text)

    if not final_message:
        final_message = full_text

    # В non-stream тоже добавляем маркер только если ожидается уточнение
    if expects_clarification and current_agent_model_id and final_message:
        final_message = f"{final_message}\n\n<!--sgr_agent_id:{current_agent_model_id}-->"

    response = make_final_response(final_message, model=requested_model)
    return JSONResponse(response)


# ---------------------------------------------------------
# Точка входа
# ---------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=PROXY_HOST,
        port=PROXY_PORT,
    )
