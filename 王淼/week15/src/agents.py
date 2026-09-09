"""
主 Agent + 并行 Subagent 编排（旅行规划版）

教学重点：
  1. 主 agent 自己是 ReAct 循环，有 2 个工具：
     - web_search：单次联网搜索（简单问题直接用）
     - dispatch_subagents：派发多个 subagent 并行调研（多侧面旅行规划用）
     主 agent 根据 query 自行决定用哪个——不是固定拓扑，是 LLM 自主路由
  2. 并行优势凸显：dispatch_subagents 一次派发 N 个 subagent，
     ThreadPoolExecutor 并行跑，wall-clock ≈ max(单agent时长)，
     而非 sum——这就是 subagent 并行的核心价值
  3. 每个 subagent 也是 ReAct 循环（只 web_search 工具），
     trace 全程捕获存入 shared_state，供可视化「点节点看 ReAct 过程」

架构对应 PPT Part 6.3 的 Orchestrator-Workers 拓扑（动态：主 agent 决定派几个）。
本项目由 market_research_subagents 改造：引擎(react_loop)/搜索(tavily)/服务(serve)
全部复用，只改 agents.py 的 MAIN_SYSTEM prompt 与测试问题。
"""

import os, time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是旅行规划主规划师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子规划师并行调研（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要旅行需求涉及 2 个及以上方面（如「旅行规划」「行程安排」「去XX玩几天」「旅游攻略」等），
  必须用 dispatch_subagents 把各方面拆给子规划师并行处理，不要自己串行 web_search 多次。
  示例："成都3天旅行规划：景点、美食、交通住宿" → Action: dispatch_subagents
        Action Input: 成都必去景点与3天行程路线安排 | 成都特色美食与餐厅推荐 | 成都交通方式与住宿区域建议
- 只有单一事实问题（如"成都双流机场到市区多远"）才直接 web_search
- 拿到子规划结果后，综合成结构化行程方案

报告要求：按天或按维度组织行程，每个推荐带来源，末尾给预算估算与出行注意事项。

【示例】
Question: 云南5天旅行规划：昆明大理丽江，景点路线、特色美食、交通住宿
Thought: 这是多方面旅行规划（3个侧面），必须派发子规划师并行收集，不能自己串行搜索
Action: dispatch_subagents
Action Input: 云南昆明大理丽江5天景点路线与行程安排 | 云南特色美食与餐厅推荐 | 云南城际交通方式与住宿区域建议
Observation: 并行调研完成：3 个子规划师...（各子课题结果）
Thought: 已收齐三个维度的并行调研结果，综合成行程方案
Final Answer: （按天行程方案）"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子课题1 | 子课题2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs sum_durations。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]  #先从action_input解析出子课题，最多6个子课题
    if not subtopics:
        return "未解析出子课题"
    shared_state = shared_state if shared_state is not None else {} 
    shared_state.setdefault("subagents", {})

    # 构造 (sid, subagent, subtopic) 三元组
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={"web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                                  "联网搜索，参数是查询词")},
            max_steps=4, model_tag="deepseek-chat(子)")
        defs.append((sid, sub, topic))

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 subagent id
    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)   # 真实 id，前端加的节点和后续 subagent_step 对得上

    t0 = time.time()
    results = {}
    # ── 执行：serial=False 并行(ThreadPool) / serial=True 串行(for 循环) ──
    def _run_one(sid=sid, sub=sub, topic=topic):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        # 串行：一个接一个，凸显并行的意义（eval A/B 对比基线）
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            topic = next(t for s, _, t in defs if s == sid)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免主 agent context 过长）
    parts = [f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行调研完成：{len(defs)} 个子规划师，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    """执行一次旅行规划。返回 {final_answer, main_trace, subagents, parallel_stats}。
    serial=True 时 subagent 串行执行（eval A/B 对比基线）。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        # dispatch 事件由 _dispatch_subagents 用真实 subagent id 发出
        # （不能在这里预生成 id，否则和 subagent_step 的 id 对不上）
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子规划师并行调研，参数=用 | 分隔的多个子课题"),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,   # ← 传主 agent 的派发引导 prompt
    )
    # 把 shared_state 注入主 agent run
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = "成都3天旅行规划：必去景点与行程路线、特色美食推荐、交通与住宿建议"
    r = run_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n报告头:\n{r['final_answer'][:200]}")
