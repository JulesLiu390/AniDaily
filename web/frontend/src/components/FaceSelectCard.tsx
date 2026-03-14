import { useState } from "react";
import { getFileUrl } from "../api";

export interface FaceInfo {
  index: number;
  name: string;
  description: string;
  age: number | null;
  gender: string | null;
  crop_path: string;
  crop_url: string;
}

interface Props {
  faces: FaceInfo[];
  skippedSmall: number;
  skippedBlurry: number;
  onConfirm: (selected: FaceInfo[]) => void;
  disabled?: boolean;
}

export default function FaceSelectCard({ faces, skippedSmall, skippedBlurry, onConfirm, disabled }: Props) {
  const [selected, setSelected] = useState<Set<number>>(() => new Set(faces.map((f) => f.index)));
  const [confirmed, setConfirmed] = useState(false);

  const toggle = (index: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const selectAll = () => setSelected(new Set(faces.map((f) => f.index)));
  const selectNone = () => setSelected(new Set());

  const handleConfirm = () => {
    setConfirmed(true);
    onConfirm(faces.filter((f) => selected.has(f.index)));
  };

  const handleSkip = () => {
    setConfirmed(true);
    onConfirm([]);
  };

  if (confirmed) {
    const count = faces.filter((f) => selected.has(f.index)).length;
    return (
      <div className="bg-green-50 border border-green-200 rounded-xl p-3 my-2">
        <div className="text-xs text-green-600 font-medium">
          {count > 0 ? `已确认风格化 ${count} 张人脸` : "已跳过风格化"}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-3 my-2">
      <div className="text-xs text-blue-600 font-medium mb-2">
        检测到 {faces.length} 张人脸，请选择要风格化的人脸：
      </div>

      {(skippedSmall > 0 || skippedBlurry > 0) && (
        <div className="text-[10px] text-gray-400 mb-2">
          已过滤：{skippedSmall > 0 && `${skippedSmall} 张太小`}{skippedSmall > 0 && skippedBlurry > 0 && "、"}{skippedBlurry > 0 && `${skippedBlurry} 张模糊`}
        </div>
      )}

      {/* Face grid */}
      <div className="flex flex-wrap gap-2 mb-2">
        {faces.map((face) => {
          const isSelected = selected.has(face.index);
          return (
            <button
              key={face.index}
              onClick={() => toggle(face.index)}
              className={`flex items-center gap-2 rounded-lg px-2 py-1.5 border shadow-sm transition-colors ${
                isSelected
                  ? "bg-blue-100 border-blue-400 ring-1 ring-blue-400"
                  : "bg-white border-gray-200 hover:border-blue-300"
              }`}
            >
              <img
                src={getFileUrl(face.crop_url)}
                alt={face.name}
                className="w-12 h-12 rounded object-cover"
              />
              <div className="text-xs text-left min-w-0">
                <div className="font-medium text-gray-700 truncate max-w-[100px]">
                  {face.name}
                </div>
                <div className="text-gray-400 text-[10px]">
                  {face.gender === "M" ? "男" : face.gender === "F" ? "女" : ""}{face.age ? ` ${face.age}岁` : ""}
                </div>
                {face.description && (
                  <div className="text-gray-400 truncate max-w-[100px] text-[10px]">{face.description}</div>
                )}
              </div>
              <div className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 ${
                isSelected ? "bg-blue-500 border-blue-500 text-white" : "border-gray-300"
              }`}>
                {isSelected && <span className="text-[10px]">✓</span>}
              </div>
            </button>
          );
        })}
      </div>

      {/* Quick actions */}
      {faces.length > 1 && (
        <div className="flex gap-2 mb-2">
          <button onClick={selectAll} className="text-[10px] text-blue-500 hover:underline">全选</button>
          <button onClick={selectNone} className="text-[10px] text-gray-400 hover:underline">取消全选</button>
        </div>
      )}

      {/* Confirm / Skip */}
      <div className="flex gap-2">
        <button
          onClick={handleConfirm}
          disabled={selected.size === 0 || disabled}
          className="flex-1 py-1.5 bg-blue-500 text-white text-xs font-medium rounded-lg hover:bg-blue-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          风格化选中 ({selected.size})
        </button>
        <button
          onClick={handleSkip}
          disabled={disabled}
          className="px-3 py-1.5 bg-gray-100 text-gray-600 text-xs font-medium rounded-lg hover:bg-gray-200 disabled:opacity-30 transition-colors"
        >
          跳过
        </button>
      </div>
    </div>
  );
}
