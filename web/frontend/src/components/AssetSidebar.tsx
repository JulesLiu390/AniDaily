import type { Assets } from "../api";
import { getFileUrl } from "../api";

const CATEGORY_LABELS: Record<string, string> = {
  characters: "角色",
  faces: "人脸",
  scenes: "场景",
  scenes_raw: "场景(原图)",
  panels: "条漫",
  scripts: "剧本",
};

interface Props {
  assets: Assets;
  onAssetClick?: (path: string) => void;
  onRefresh?: () => void;
}

export default function AssetSidebar({ assets, onAssetClick, onRefresh }: Props) {
  return (
    <div className="w-64 bg-white border-r border-gray-200 flex flex-col h-full">
      <div className="p-3 border-b border-gray-200 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-700">素材库</h2>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="text-xs text-blue-500 hover:text-blue-700"
          >
            刷新
          </button>
        )}
      </div>
      <div className="flex-1 overflow-y-auto">
        {Object.entries(assets).map(([category, items]) => (
          <div key={category} className="border-b border-gray-100">
            <div className="px-3 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide bg-gray-50">
              {CATEGORY_LABELS[category] || category} ({items.length})
            </div>
            {items.length === 0 ? (
              <div className="px-3 py-2 text-xs text-gray-400">暂无</div>
            ) : (
              <div className="p-2 grid grid-cols-2 gap-1.5">
                {items.map((item) =>
                  item.type === "image" ? (
                    <div
                      key={item.name}
                      className="cursor-pointer group"
                      onClick={() => onAssetClick?.(item.path)}
                    >
                      <img
                        src={getFileUrl(item.url)}
                        alt={item.name}
                        className="w-full aspect-square object-cover rounded border border-gray-200 group-hover:border-blue-400 transition-colors"
                      />
                      <div className="text-[10px] text-gray-500 truncate mt-0.5 text-center">
                        {item.name}
                      </div>
                    </div>
                  ) : (
                    <div
                      key={item.name}
                      className="col-span-2 px-2 py-1.5 text-xs text-gray-600 hover:bg-gray-100 rounded cursor-pointer truncate"
                      onClick={() => onAssetClick?.(item.path)}
                    >
                      📄 {item.name}
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
