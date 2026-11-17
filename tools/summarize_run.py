"""Summarize one OmniGamer telemetry run using only the standard library."""
import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


STAGES = (
    ("capture", "Захват кадра"),
    ("memory_lookup", "Поиск памяти"),
    ("vlm_decide", "Решение VLM"),
    ("grounding", "Локализация цели"),
    ("input_send", "Выполнение действия"),
    ("post_wait", "Пауза после действия"),
    ("preflight_eval", "Предварительная проверка"),
    ("result_eval", "Оценка результата"),
    ("memory_write", "Запись памяти"),
    ("total", "Весь шаг"),
)
STATUSES = ("success", "no_effect", "uncertain", "refusal", "camera_motion")


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Некорректный JSON в {path.name}:{line_number}: {exc}") from exc
    return rows


def _parse_time(value):
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _p95(values):
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def _ms(value):
    return "—" if value is None else f"{value:.2f} мс"


def _mb(value):
    return "—" if value is None else f"{value:.2f} МБ"


def _format_stats(values):
    if not values:
        return "нет данных"
    return (f"N={len(values)}; среднее {_ms(statistics.mean(values))}; "
            f"медиана {_ms(statistics.median(values))}; p95 {_ms(_p95(values))}")


def summarize(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Каталог прогона не найден: {run_dir}")
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Нет run_meta.json: {run_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    steps = _read_jsonl(run_dir / "steps.jsonl")
    resources = _read_jsonl(run_dir / "resources.jsonl")
    started = _parse_time(meta.get("started_at"))
    ended = _parse_time(meta.get("ended_at"))
    if started and ended:
        duration_s = max(0.0, (ended - started).total_seconds())
    else:
        duration_s = 0.0
    rate = len(steps) * 60 / duration_s if duration_s > 0 else None

    stage_values = {}
    total_values = [step.get("timings_ms", {}).get("total") for step in steps]
    total_sum = sum(value for value in total_values if _number(value))
    for key, _ in STAGES:
        values = [step.get("timings_ms", {}).get(key) for step in steps]
        values = [value for value in values if _number(value)]
        stage_values[key] = values

    by_action = defaultdict(Counter)
    failure_reasons = Counter()
    errors = Counter()
    for step in steps:
        decision = step.get("decision") or {}
        outcome = step.get("outcome") or {}
        action = decision.get("action_type") or "unknown"
        by_action[action][outcome.get("status", "uncertain")] += 1
        if decision.get("refusal_reason"):
            failure_reasons[decision["refusal_reason"]] += 1
        error = step.get("error") or {}
        if error:
            errors[f"{error.get('type', 'Error')}: {error.get('message', '')}"] += 1
    streaks = [(step.get("loop") or {}).get("no_effect_streak", 0) for step in steps]
    camera_streaks = [(step.get("loop") or {}).get("camera_only_streak", 0) for step in steps]
    ram = [row.get("process_ram_mb") for row in resources if _number(row.get("process_ram_mb"))]
    vram = [row.get("gpu_vram_used_mb") for row in resources if _number(row.get("gpu_vram_used_mb"))]
    fallback_steps = sum(bool((step.get("decision") or {}).get("fallback_reason")) for step in steps)
    skill_steps = _read_jsonl(run_dir / "skill_steps.jsonl")
    goal_events = _read_jsonl(run_dir / "goal_steps.jsonl")
    goal_result = next((row for row in reversed(goal_events) if row.get("event") == "run_finish"), {})
    skill_outcomes = defaultdict(Counter)
    for row in skill_steps:
        if row.get("event") == "finish":
            result = row.get("result") or {}
            if result.get("status"):
                skill_outcomes[row.get("skill", "unknown")][result["status"]] += 1

    lines = [
        f"Отчёт о прогоне: {meta.get('run_id', run_dir.name)}",
        f"Длительность: {duration_s:.1f} с; шагов: {len(steps)}; шагов/мин: {'—' if rate is None else f'{rate:.2f}'}",
        f"Причина завершения: {meta.get('termination_reason') or 'ещё выполняется'}",
        "",
        "Тайминги этапов (среднее / медиана / p95 / доля общего времени):",
    ]
    for key, label in STAGES:
        values = stage_values[key]
        avg = statistics.mean(values) if values else None
        med = statistics.median(values) if values else None
        p95 = _p95(values)
        share = (sum(values) / total_sum * 100) if values and total_sum > 0 else None
        lines.append(f"  {label}: {_ms(avg)} / {_ms(med)} / {_ms(p95)} / {'—' if share is None else f'{share:.1f}%'}")
    lines.extend(["", "Действия и результаты:"])
    if by_action:
        for action in sorted(by_action):
            counts = by_action[action]
            total = sum(counts.values())
            percentages = ", ".join(f"{status} {counts[status] * 100 / total:.1f}%" for status in STATUSES)
            lines.append(f"  {action}: {total}; {percentages}")
    else:
        lines.append("  Нет записанных шагов")
    lines.extend(["", "Частые причины отказов:"])
    lines.extend(f"  {reason}: {count}" for reason, count in failure_reasons.most_common(5))
    if not failure_reasons:
        lines.append("  Нет")
    lines.append("Частые ошибки:")
    lines.extend(f"  {reason}: {count}" for reason, count in errors.most_common(5))
    if not errors:
        lines.append("  Нет")
    lines.append(f"Максимальная серия шагов без видимого эффекта: {max(streaks, default=0)}")
    lines.append(f"Максимальная серия look/wait: {max(camera_streaks, default=0)}")
    lines.append(f"Шагов с fallback: {fallback_steps} ({fallback_steps * 100 / len(steps):.1f}%)" if steps
                 else "Шагов с fallback: 0 (—)")
    lines.append(f"RAM процесса, пик/среднее: {_mb(max(ram) if ram else None)} / {_mb(statistics.mean(ram) if ram else None)}")
    lines.append(f"VRAM, пик/среднее: {_mb(max(vram) if vram else None)} / {_mb(statistics.mean(vram) if vram else None)}")
    finished = bool(goal_result)
    succeeded = finished and goal_result.get("status") == "SUCCESS"
    verified_skill_finishes = [
        row for row in goal_events if row.get("event") == "skill_result"
        and (row.get("result") or {}).get("result_code") == "SUCCESS_VERIFIED"
    ]
    skill_finishes = [row for row in goal_events if row.get("event") == "skill_result"
                      and (row.get("result") or {}).get("status")]
    vlm_calls = sum(value for step in steps
                    for value in [(step.get("decision") or {}).get("vlm_http_requests")]
                    if _number(value))
    planner_calls = sum(row.get("request_count", 0) for row in goal_events
                        if row.get("event") == "plan" and _number(row.get("request_count")))
    uncertain_count = sum((step.get("outcome") or {}).get("status") == "uncertain" for step in steps)
    uncertain_count += sum((row.get("result") or {}).get("result_code") == "UNCONFIRMED"
                           for row in goal_events if row.get("event") == "skill_result")
    lost_count = sum(row.get("state") == "LOST" for row in skill_steps)
    look_corrections = sum((row.get("action") or {}).get("type") == "look" for row in skill_steps)
    recoveries = sum((row.get("action") or {}).get("type") == "jump_recovery" for row in skill_steps)
    repeated_failures = sum(bool((step.get("loop") or {}).get("state_repeat")
                                 or (step.get("loop") or {}).get("repeat_warning")) for step in steps)
    if goal_result.get("status") == "DRY_RUN":
        task_success = "не применяется (dry-run)"
    elif finished:
        task_success = "1/1 (100.0%)" if succeeded else "0/1 (0.0%)"
    else:
        task_success = "нет завершённого goal-прогона"
    lines.extend(["", "Метрики качества (только наблюдаемая телеметрия):",
                  f"  task_success_rate: {task_success}",
                  f"  skill_success_rate: {len(verified_skill_finishes)}/{len(skill_finishes)} "
                  f"({100 * len(verified_skill_finishes) / len(skill_finishes):.1f}%)" if skill_finishes
                  else "  skill_success_rate: нет завершённых проверяемых навыков",
                  f"  time_to_success: {duration_s:.1f} с (длительность всего прогона; прокси)"
                  if succeeded else "  time_to_success: нет подтверждённого успеха",
                  f"  actions_per_success: {goal_result.get('input_actions_used')}"
                  if succeeded and _number(goal_result.get("input_actions_used"))
                  else "  actions_per_success: нет данных для подтверждённого успеха",
                  f"  vlm_calls_per_success: {vlm_calls + planner_calls} запросов "
                  f"(шаги: {vlm_calls}; планирование: {planner_calls}; проверки goal не включены)"
                  if succeeded else "  vlm_calls_per_success: нет подтверждённого успеха",
                  "  detector_calls_per_success: не измеряется как число вызовов; записаны только задержки отдельных наблюдений",
                  "  target_switch_count: нет данных (нет устойчивого ID цели)",
                  f"  target_lost_count: {lost_count} записей состояния LOST",
                  f"  camera_correction_count: {look_corrections} действий look в журналах навыков",
                  f"  recovery_attempt_count: {recoveries} действий jump_recovery",
                  f"  repeated_failure_count: {repeated_failures} шагов с state_repeat/repeat_warning",
                  f"  memory_reuse_count: {goal_result.get('memory_reuse_count', 'нет данных')}",
                  f"  uncertain_outcome_count: {uncertain_count}"])
    lines.extend(["", "Пять самых долгих шагов:"])
    ranked = sorted(steps, key=lambda step: step.get("timings_ms", {}).get("total") or -1, reverse=True)[:5]
    for step in ranked:
        timings = step.get("timings_ms") or {}
        detail = ", ".join(f"{key}={timings.get(key):.1f}" for key, _ in STAGES[:-1]
                            if _number(timings.get(key)))
        lines.append(f"  шаг {step.get('step')}: total={_ms(timings.get('total'))}; {detail}")
    if (run_dir / "skill_steps.jsonl").exists():
        result = meta.get("skill_result") or {}
        final_result = next((row.get("result") for row in reversed(skill_steps)
                             if row.get("event") == "finish" and row.get("result")), {})
        state = result.get("state") or final_result.get("status", "не записано")
        reason = result.get("reason") or final_result.get("reason", "не записана")
        latencies = []
        heights = []
        for row in skill_steps:
            latency = row.get("detector_latency_ms") or {}
            for value in latency.values():
                if _number(value):
                    latencies.append(value)
            height = (row.get("h") or {}).get("after")
            if _number(height):
                heights.append((row.get("iteration"), height))
        lines.extend(["", "Сводка навыков:"])
        if skill_outcomes:
            for skill_name, outcomes in sorted(skill_outcomes.items()):
                counts = ", ".join(f"{status}: {count}" for status, count in sorted(outcomes.items()))
                lines.append(f"  {skill_name}: {counts}")
        else:
            lines.append("  Нет финальных результатов в общем формате")
        lines.extend([
            "", "Подробности последнего навыка:",
            f"  Состояние: {state}; причина: {reason}",
            f"  Итераций: {result.get('iterations', len(skill_steps))}; время: "
            f"{result.get('elapsed_seconds', 'не записано')} с",
            f"  Средняя задержка детектора: {_ms(statistics.mean(latencies) if latencies else None)}",
            "  Кривая h по итерациям: " + (", ".join(f"{step}={height:.3f}" for step, height in heights)
                                             if heights else "нет измерений"),
        ])
    if goal_events:
        plans = [row for row in goal_events if row.get("event") == "plan"]
        would_send = [row for row in goal_events if row.get("event") == "would_send"]
        verification_checks = {
            "goal_postcondition": [],
            "plan_transition": [],
            "interaction_outcome": [],
            "region_change": [],
        }
        other_checks = []
        for row in goal_events:
            verification = ((row.get("result") or {}).get("evidence") or {}).get("verification") or {}
            duration = verification.get("duration_seconds")
            if _number(duration):
                scope = verification.get("scope") or "unknown"
                destination = verification_checks.get(scope, other_checks)
                destination.append(duration * 1000)
        verified_examples = sum(row.get("verified_example_count", 0) for row in plans
                                if _number(row.get("verified_example_count")))
        memory_reuse = goal_result.get("memory_reuse_count")
        if not _number(memory_reuse):
            memory_reuse = sum(row.get("memory_reuse_count", 0) for row in plans
                               if _number(row.get("memory_reuse_count")))
        lines.extend(["", "Сводка goal-режима:",
                      f"  Проверенных планов: {len(plans)}; перепланирований: {max(0, len(plans) - 1)}",
                      f"  would_send: {len(would_send)}",
                      f"  Доступно verified-примеров по планам: {verified_examples}; точных повторов args: {memory_reuse}",
                      f"  Проверки постусловия цели: {_format_stats(verification_checks['goal_postcondition'])}",
                      f"  Проверки перехода плана: {_format_stats(verification_checks['plan_transition'])}",
                      f"  Проверки исхода взаимодействия: {_format_stats(verification_checks['interaction_outcome'])}",
                      f"  Проверки дельты региона: {_format_stats(verification_checks['region_change'])}",
                      f"  Прочие проверки: {_format_stats(other_checks)}",
                      f"  Завершение: {goal_result.get('status', 'не записано')} — "
                      f"{goal_result.get('reason', 'не записана')}"])
    return "\n".join(lines)


def latest_run(runs_dir):
    candidates = [path for path in Path(runs_dir).iterdir()
                  if path.is_dir() and (path / "run_meta.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"Нет прогонов в {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    default_runs = Path(__file__).resolve().parents[1] / "data" / "runs"
    parser = argparse.ArgumentParser(description="Сводка телеметрии OmniGamer")
    parser.add_argument("run_dir", nargs="?", help="Путь к каталогу прогона; по умолчанию последний прогон")
    args = parser.parse_args(argv)
    try:
        run_dir = Path(args.run_dir) if args.run_dir else latest_run(default_runs)
        print(summarize(run_dir))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
