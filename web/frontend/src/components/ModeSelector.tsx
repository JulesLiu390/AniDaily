import { useLang } from "../LanguageContext";

interface Props {
  mode: "comic" | "storyboard";
  onChange: (mode: "comic" | "storyboard") => void;
  disabled?: boolean;
}

export default function ModeSelector({ mode, onChange, disabled }: Props) {
  const { t } = useLang();
  return (
    <div className="flex items-center bg-gray-100 rounded-lg p-0.5 text-xs">
      <button
        onClick={() => onChange("comic")}
        disabled={disabled}
        className={`px-2.5 py-1 rounded-md transition-colors ${
          mode === "comic"
            ? "bg-white text-gray-800 shadow-sm font-medium"
            : "text-gray-500 hover:text-gray-700"
        } disabled:opacity-50`}
      >
        {t("mode.comic")}
      </button>
      <button
        onClick={() => onChange("storyboard")}
        disabled={disabled}
        className={`px-2.5 py-1 rounded-md transition-colors ${
          mode === "storyboard"
            ? "bg-white text-gray-800 shadow-sm font-medium"
            : "text-gray-500 hover:text-gray-700"
        } disabled:opacity-50`}
      >
        {t("mode.storyboard")}
      </button>
    </div>
  );
}
