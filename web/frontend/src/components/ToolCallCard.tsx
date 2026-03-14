import { useState } from "react";
import type { ToolCallInfo } from "../api";

interface Props {
  toolCall: ToolCallInfo;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export default function ToolCallCard({ toolCall }: Props) {
  const [expanded, setExpanded] = useState(false);
  const isPending = toolCall.pending;
  const hasError = toolCall.result && "error" in toolCall.result;

  return (
    <div
      className={`rounded-lg border transition-colors cursor-pointer ${
        isPending
          ? "border-blue-200 bg-blue-50"
          : hasError
            ? "border-red-200 bg-red-50"
            : "border-green-200 bg-green-50"
      }`}
      onClick={() => !isPending && setExpanded(!expanded)}
    >
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex items-center gap-2">
          {isPending ? (
            <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          ) : (
            <span className={`text-sm ${hasError ? "text-red-500" : "text-green-500"}`}>
              {hasError ? "✗" : "✓"}
            </span>
          )}
          <span className="text-sm font-medium text-gray-700">
            {toolCall.tool}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {toolCall.duration_ms != null && (
            <span className="text-xs text-gray-400">
              {formatDuration(toolCall.duration_ms)}
            </span>
          )}
          {!isPending && (
            <span className="text-xs text-gray-400">
              {expanded ? "▲" : "▶"}
            </span>
          )}
        </div>
      </div>
      {expanded && !isPending && (
        <div className="px-3 pb-2 space-y-1 border-t border-green-100">
          <div className="text-xs text-gray-500 mt-1.5">
            <div className="font-medium mb-0.5">Args:</div>
            <pre className="whitespace-pre-wrap bg-white/60 rounded p-1.5 overflow-x-auto">
              {JSON.stringify(toolCall.args, null, 2)}
            </pre>
          </div>
          {toolCall.result && (
            <div className="text-xs text-gray-500">
              <div className="font-medium mb-0.5">Result:</div>
              <pre className="whitespace-pre-wrap bg-white/60 rounded p-1.5 overflow-x-auto max-h-40 overflow-y-auto">
                {JSON.stringify(toolCall.result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
