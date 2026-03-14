import { useRef, useState } from "react";
import type { PlanStep } from "../api";

interface Props {
  steps: PlanStep[];
  onConfirm: (steps: PlanStep[]) => void;
  onRevise: (prompt: string) => void;
  disabled?: boolean;
}

export default function TaskPlanCard({ steps: initialSteps, onConfirm, onRevise, disabled }: Props) {
  const [steps, setSteps] = useState<PlanStep[]>(() =>
    initialSteps.map((s) => ({ ...s, status: s.status || "pending" }))
  );
  const [confirmed, setConfirmed] = useState(false);
  const [revising, setRevising] = useState(false);
  const [reviseText, setReviseText] = useState("");
  const composingRef = useRef(false);

  // Track which steps are enabled (not skipped)
  const [enabled, setEnabled] = useState<Set<number>>(() => new Set(steps.map((s) => s.id)));

  const toggleStep = (id: number) => {
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleConfirm = () => {
    setConfirmed(true);
    const finalSteps = steps.map((s) => ({
      ...s,
      status: enabled.has(s.id) ? "pending" as const : "skipped" as const,
    }));
    setSteps(finalSteps);
    onConfirm(finalSteps);
  };

  const handleReviseSubmit = () => {
    const text = reviseText.trim();
    if (!text) return;
    setConfirmed(true);
    onRevise(text);
  };

  // Confirmed view: show progress
  if (confirmed) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-xl p-3 my-2">
        <div className="text-xs font-medium text-gray-600 mb-2">任务计划</div>
        <div className="space-y-1">
          {steps.map((step) => {
            const isSkipped = step.status === "skipped" || !enabled.has(step.id);
            const isDone = step.status === "done";
            const isActive = step.status === "active";
            return (
              <div key={step.id} className={`flex items-center gap-2 text-xs ${isSkipped ? "opacity-40" : ""}`}>
                <span className="w-4 text-center flex-shrink-0">
                  {isDone ? (
                    <span className="text-green-500">✓</span>
                  ) : isActive ? (
                    <span className="inline-block w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                  ) : isSkipped ? (
                    <span className="text-gray-300">—</span>
                  ) : (
                    <span className="text-gray-300">○</span>
                  )}
                </span>
                <span className={`${isDone ? "text-gray-500" : isActive ? "text-blue-600 font-medium" : "text-gray-500"}`}>
                  {step.label}
                </span>
                {step.needs_confirm && !isSkipped && !isDone && (
                  <span className="text-[10px] text-blue-400 bg-blue-50 px-1 rounded">需确认</span>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // Proposal view: let user toggle steps and confirm
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-3 my-2">
      <div className="text-xs font-medium text-blue-600 mb-2">任务计划 — 请确认或调整：</div>
      <div className="space-y-1 mb-3">
        {steps.map((step) => {
          const isEnabled = enabled.has(step.id);
          return (
            <button
              key={step.id}
              onClick={() => toggleStep(step.id)}
              className={`w-full flex items-center gap-2 text-xs text-left px-2 py-1.5 rounded-lg transition-colors ${
                isEnabled ? "bg-white border border-blue-200" : "bg-gray-100 border border-gray-200 opacity-50"
              }`}
            >
              <div className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 ${
                isEnabled ? "bg-blue-500 border-blue-500 text-white" : "border-gray-300"
              }`}>
                {isEnabled && <span className="text-[10px]">✓</span>}
              </div>
              <span className="text-gray-700">{step.id}. {step.label}</span>
              {step.needs_confirm && (
                <span className="text-[10px] text-blue-400 bg-blue-50 px-1 rounded ml-auto">需确认</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Revise input */}
      {revising && (
        <div className="mb-3 flex gap-2">
          <input
            type="text"
            value={reviseText}
            onChange={(e) => setReviseText(e.target.value)}
            onCompositionStart={() => { composingRef.current = true; }}
            onCompositionEnd={() => { composingRef.current = false; }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !composingRef.current) {
                e.preventDefault();
                handleReviseSubmit();
              }
            }}
            placeholder="输入修改指令，例如：不需要写剧本，直接生成..."
            className="flex-1 px-2.5 py-1.5 text-xs border border-gray-300 rounded-lg focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400"
            autoFocus
          />
          <button
            onClick={handleReviseSubmit}
            disabled={!reviseText.trim()}
            className="px-3 py-1.5 bg-blue-500 text-white text-xs font-medium rounded-lg hover:bg-blue-600 disabled:opacity-30 transition-colors"
          >
            发送
          </button>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={handleConfirm}
          disabled={enabled.size === 0 || disabled}
          className="flex-1 py-1.5 bg-blue-500 text-white text-xs font-medium rounded-lg hover:bg-blue-600 disabled:opacity-30 transition-colors"
        >
          开始执行 ({enabled.size} 步)
        </button>
        <button
          onClick={() => setRevising(!revising)}
          className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
            revising ? "bg-blue-100 text-blue-600" : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          修改计划
        </button>
      </div>
    </div>
  );
}
