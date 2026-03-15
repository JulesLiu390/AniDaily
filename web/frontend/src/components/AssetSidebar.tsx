import { useState } from "react";
import type { Assets } from "../api";
import { getFileUrl, deleteAsset } from "../api";
import { useLang } from "../LanguageContext";
import ConfirmDialog from "./ConfirmDialog";

const CAT_KEYS: Record<string, string> = {
  style: "cat.style",
  originals: "cat.input",
  characters: "cat.stylized",
  faces: "cat.faces",
  scenes: "cat.scenes_stylized",
  scenes_raw: "cat.scenes_no_people",
  panels: "cat.panels",
  videos: "cat.videos",
  storyboard_strips: "cat.storyboard_strips",
  storyboard_frames: "cat.storyboard_frames",
  storyboards: "cat.storyboards",
  clip_scripts: "cat.clip_scripts",
  scripts: "cat.scripts",
};

interface Props {
  assets: Assets;
  loading?: boolean;
  onAssetClick?: (path: string) => void;
  onRefresh?: () => void;
}

export default function AssetSidebar({ assets, loading, onAssetClick, onRefresh }: Props) {
  const { t } = useLang();
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const toggleCategory = (category: string) => {
    setCollapsed((prev) => ({ ...prev, [category]: !prev[category] }));
  };

  // Categories that show one item per row at original aspect ratio, sorted numerically
  const SINGLE_COL_CATS = new Set(["storyboard_strips", "storyboard_frames"]);
  const NUM_SORT_CATS = new Set(["storyboard_strips", "storyboard_frames", "clip_scripts", "videos"]);

  const numSort = (a: { name: string }, b: { name: string }) => {
    const na = parseInt(a.name.match(/\d+/)?.[0] || "0", 10);
    const nb = parseInt(b.name.match(/\d+/)?.[0] || "0", 10);
    return na - nb;
  };

  const [deleteTarget, setDeleteTarget] = useState<{ name: string; path: string } | null>(null);

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;
    try {
      await deleteAsset(deleteTarget.path);
      onRefresh?.();
    } catch (err) {
      console.error("Delete failed:", err);
    }
    setDeleteTarget(null);
  };

  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col h-full">
      <div className="p-3 border-b border-gray-200 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-700">{t("sidebar.title")}</h2>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="text-xs text-blue-500 hover:text-blue-700"
          >
            {t("common.refresh")}
          </button>
        )}
      </div>
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center py-8">
            <div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
          </div>
        )}
        {!loading && Object.entries(assets).map(([category, items]) => (
          <div key={category} className="border-b border-gray-100">
            <button
              onClick={() => toggleCategory(category)}
              className="w-full px-3 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide bg-gray-50 flex items-center justify-between hover:bg-gray-100 transition-colors"
            >
              <span>{t(CAT_KEYS[category] || category)} ({items.length})</span>
              <span className={`text-[10px] transition-transform ${collapsed[category] ? "" : "rotate-90"}`}>▶</span>
            </button>
            {!collapsed[category] && (items.length === 0 ? (
              <div className="px-3 py-2 text-xs text-gray-400">{t("sidebar.empty")}</div>
            ) : SINGLE_COL_CATS.has(category) ? (
              <div className="p-2 flex flex-col gap-1.5">
                {[...items].sort(numSort).map((item) => (
                  <div
                    key={item.name}
                    className="cursor-pointer group relative"
                    onClick={() => onAssetClick?.(item.path)}
                  >
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteTarget({ name: item.name, path: item.path });
                      }}
                      className="absolute -top-1 -right-1 z-10 w-4 h-4 bg-gray-500 text-white rounded-full text-[10px] leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-sm hover:bg-red-500"
                    >
                      ×
                    </button>
                    <img
                      src={getFileUrl(item.url)}
                      alt={item.name}
                      className="w-full rounded border border-gray-200 group-hover:border-blue-400 transition-colors"
                    />
                    <div className="text-[10px] text-gray-500 truncate mt-0.5 text-center">
                      {item.name}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-2 grid grid-cols-2 gap-1.5">
                {(NUM_SORT_CATS.has(category) ? [...items].sort(numSort) : items).map((item) =>
                  item.type === "image" ? (
                    <div
                      key={item.name}
                      className="cursor-pointer group relative"
                      onClick={() => onAssetClick?.(item.path)}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteTarget({ name: item.name, path: item.path });
                        }}
                        className="absolute -top-1 -right-1 z-10 w-4 h-4 bg-gray-500 text-white rounded-full text-[10px] leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-sm hover:bg-red-500"
                      >
                        ×
                      </button>
                      <img
                        src={getFileUrl(item.url)}
                        alt={item.name}
                        className="w-full aspect-square object-cover rounded border border-gray-200 group-hover:border-blue-400 transition-colors"
                      />
                      <div className="text-[10px] text-gray-500 truncate mt-0.5 text-center">
                        {item.name}
                      </div>
                    </div>
                  ) : item.type === "video" ? (
                    <div
                      key={item.name}
                      className="cursor-pointer group relative"
                      onClick={() => onAssetClick?.(item.path)}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteTarget({ name: item.name, path: item.path });
                        }}
                        className="absolute -top-1 -right-1 z-10 w-4 h-4 bg-gray-500 text-white rounded-full text-[10px] leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-sm hover:bg-red-500"
                      >
                        ×
                      </button>
                      <div className="w-full aspect-video bg-gradient-to-br from-gray-800 to-gray-900 rounded border border-gray-200 group-hover:border-blue-400 transition-colors flex flex-col items-center justify-center gap-1">
                        <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347a1.125 1.125 0 0 1-1.667-.986V5.653Z" />
                        </svg>
                        <span className="text-[9px] text-gray-400 truncate max-w-full px-1">{item.name}</span>
                      </div>
                    </div>
                  ) : (
                    <div
                      key={item.name}
                      className="col-span-2 px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-100 rounded cursor-pointer truncate group relative"
                      onClick={() => onAssetClick?.(item.path)}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteTarget({ name: item.name, path: item.path });
                        }}
                        className="absolute top-1 right-1 w-4 h-4 bg-gray-500 text-white rounded-full text-[10px] leading-none flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-sm hover:bg-red-500"
                      >
                        ×
                      </button>
                      {item.name}
                    </div>
                  )
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
      {deleteTarget && (
        <ConfirmDialog
          message={t("sidebar.deleteConfirm", { name: deleteTarget.name })}
          onConfirm={handleDeleteConfirm}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  );
}
