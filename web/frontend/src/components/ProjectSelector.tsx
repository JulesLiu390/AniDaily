import { useState } from "react";
import type { Project } from "../api";
import { useLang } from "../LanguageContext";

interface Props {
  projects: Project[];
  current: string | null;
  onSelect: (name: string) => void;
  onCreate: (name: string) => void;
  onDelete: (name: string) => void;
}

export default function ProjectSelector({ projects, current, onSelect, onCreate, onDelete }: Props) {
  const { t } = useLang();
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState("");

  const handleCreate = () => {
    const name = newName.trim();
    if (!name) return;
    onCreate(name);
    setNewName("");
    setShowNew(false);
  };

  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-white border-b border-gray-200">
      <span className="text-sm text-gray-500 shrink-0">{t("project.label")}</span>

      <select
        value={current || ""}
        onChange={(e) => onSelect(e.target.value)}
        className="text-sm border border-gray-300 rounded-lg px-2 py-1 bg-white focus:outline-none focus:border-blue-400 min-w-[120px]"
      >
        <option value="" disabled>{t("project.select")}</option>
        {projects.map((p) => (
          <option key={p.name} value={p.name}>{p.name}</option>
        ))}
      </select>

      {current && (
        <button
          onClick={() => {
            if (confirm(t("project.deleteConfirm", { name: current }))) {
              onDelete(current);
            }
          }}
          className="text-xs text-red-400 hover:text-red-600 transition-colors"
          title={t("project.deleteTitle")}
        >
          {t("common.delete")}
        </button>
      )}

      {showNew ? (
        <div className="flex items-center gap-1">
          <input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            placeholder={t("project.namePlaceholder")}
            className="text-sm border border-gray-300 rounded-lg px-2 py-1 w-32 focus:outline-none focus:border-blue-400"
          />
          <button
            onClick={handleCreate}
            className="text-xs text-blue-500 hover:text-blue-700"
          >
            {t("common.confirm")}
          </button>
          <button
            onClick={() => { setShowNew(false); setNewName(""); }}
            className="text-xs text-gray-400 hover:text-gray-600"
          >
            {t("common.cancel")}
          </button>
        </div>
      ) : (
        <button
          onClick={() => setShowNew(true)}
          className="text-sm text-blue-500 hover:text-blue-700 transition-colors"
        >
          {t("project.createNew")}
        </button>
      )}
    </div>
  );
}
