import { useEffect, useState } from "react";
import type { Asset } from "../api";
import { getFileUrl, updateAsset } from "../api";

interface Props {
  asset: Asset;
  category?: string;
  onClose: () => void;
  onSendToChat: () => void;
  onAssetUpdated?: () => void;
}

const CATEGORY_LABELS: Record<string, string> = {
  originals: "原始图片",
  characters: "角色",
  faces: "人脸",
  scenes: "场景",
  scenes_raw: "场景(原图)",
  panels: "条漫",
  scripts: "剧本",
};

export default function PreviewPanel({ asset, category, onClose, onSendToChat, onAssetUpdated }: Props) {
  const isImage = asset.type === "image";
  const isMarkdown = asset.type === "markdown";

  const [editContent, setEditContent] = useState(asset.content || "");
  const [saving, setSaving] = useState(false);

  // Reset content when asset changes
  useEffect(() => {
    setEditContent(asset.content || "");
  }, [asset.path, asset.content]);

  const dirty = isMarkdown && editContent !== (asset.content || "");

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await updateAsset(asset.path, editContent);
      if (res.updated) {
        onAssetUpdated?.();
      }
    } catch (err) {
      console.error("Save failed:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleRevert = () => {
    setEditContent(asset.content || "");
  };

  return (
    <div className="w-80 bg-white border-l border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="p-3 border-b border-gray-200 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700 truncate">预览</h3>
        <button
          onClick={onClose}
          className="w-6 h-6 flex items-center justify-center text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {isImage && (
          <div className="flex justify-center bg-gray-50 rounded-lg border border-gray-100 p-2 mb-3">
            <img
              src={getFileUrl(asset.url)}
              alt={asset.name}
              className="max-w-full max-h-[50vh] object-contain rounded"
            />
          </div>
        )}

        {isMarkdown && (
          <div className="mb-3">
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              className={`w-full h-[40vh] bg-gray-50 rounded-lg border p-3 text-xs text-gray-700 font-mono resize-none focus:outline-none focus:ring-1 focus:ring-blue-400 ${
                dirty ? "border-blue-300" : "border-gray-100"
              }`}
            />
          </div>
        )}

        {/* Metadata */}
        <div className="space-y-2">
          <div>
            <div className="text-[10px] text-gray-400 mb-0.5">文件名</div>
            <div className="text-xs text-gray-700 break-all">{asset.name}</div>
          </div>

          {category && (
            <div>
              <div className="text-[10px] text-gray-400 mb-0.5">分类</div>
              <div className="text-xs text-gray-700">{CATEGORY_LABELS[category] || category}</div>
            </div>
          )}

          {asset.description && (
            <div>
              <div className="text-[10px] text-gray-400 mb-0.5">描述</div>
              <div className="text-xs text-gray-700">{asset.description}</div>
            </div>
          )}

          {asset.source_face && (
            <div>
              <div className="text-[10px] text-gray-400 mb-0.5">来源人脸</div>
              <div className="text-xs text-gray-700">{asset.source_face}</div>
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="p-3 border-t border-gray-200 space-y-2">
        {dirty && (
          <div className="flex gap-2">
            <button
              onClick={handleRevert}
              className="flex-1 py-2 bg-gray-100 text-gray-700 text-xs font-medium rounded-lg hover:bg-gray-200 transition-colors"
            >
              撤销
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex-1 py-2 bg-blue-500 text-white text-xs font-medium rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors"
            >
              {saving ? "保存中..." : "保存"}
            </button>
          </div>
        )}
        <button
          onClick={onSendToChat}
          className="w-full py-2 bg-blue-500 text-white text-xs font-medium rounded-lg hover:bg-blue-600 transition-colors"
        >
          发送到聊天
        </button>
      </div>
    </div>
  );
}
