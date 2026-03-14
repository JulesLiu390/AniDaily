import { useState, useRef, useEffect, useCallback } from "react";
import type { AttachedImage, ChatMessage, ToolCallInfo } from "../api";
import { getFileUrl, streamMessage, uploadImage } from "../api";
import ToolCallCard from "./ToolCallCard";

interface Props {
  project: string;
  onNewImages?: () => void;
  pendingAssets?: AttachedImage[];
  onClearPendingAssets?: () => void;
}

export default function ChatPanel({ project, onNewImages, pendingAssets, onClearPendingAssets }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [attachedImages, setAttachedImages] = useState<AttachedImage[]>([]);
  const [uploading, setUploading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Track whether new images were generated during the current stream
  const hasNewImagesRef = useRef(false);

  // Reset chat when project changes
  useEffect(() => {
    setMessages([]);
    setConversationId(null);
    setInput("");
    setAttachedImages([]);
  }, [project]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Receive assets from sidebar
  useEffect(() => {
    if (pendingAssets && pendingAssets.length > 0) {
      setAttachedImages((prev) => {
        const existing = new Set(prev.map((i) => i.path));
        const newOnes = pendingAssets.filter((a) => !existing.has(a.path));
        return [...prev, ...newOnes];
      });
      onClearPendingAssets?.();
    }
  }, [pendingAssets, onClearPendingAssets]);

  const addImageFiles = async (files: File[]) => {
    if (files.length === 0) return;
    setUploading(true);
    try {
      for (const file of files) {
        const previewUrl = URL.createObjectURL(file);
        const result = await uploadImage(file, project);
        setAttachedImages((prev) => [
          ...prev,
          {
            path: result.path,
            url: result.url,
            name: result.name,
            previewUrl,
          },
        ]);
      }
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    await addImageFiles(Array.from(files));
  };

  const handlePaste = async (e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    const imageFiles: File[] = [];
    for (const item of Array.from(items)) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) imageFiles.push(file);
      }
    }
    if (imageFiles.length > 0) {
      e.preventDefault();
      await addImageFiles(imageFiles);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files).filter((f) =>
      f.type.startsWith("image/")
    );
    if (files.length > 0) {
      await addImageFiles(files);
    }
  };

  const [dragOver, setDragOver] = useState(false);

  const removeAttached = (index: number) => {
    setAttachedImages((prev) => {
      const img = prev[index];
      if (img.previewUrl) URL.revokeObjectURL(img.previewUrl);
      return prev.filter((_, i) => i !== index);
    });
  };

  // Helper to update the last assistant message
  const updateLastAssistant = useCallback((updater: (msg: ChatMessage) => ChatMessage) => {
    setMessages((prev) => {
      const copy = [...prev];
      for (let i = copy.length - 1; i >= 0; i--) {
        if (copy[i].role === "assistant") {
          copy[i] = updater(copy[i]);
          return copy;
        }
      }
      return prev;
    });
  }, []);

  const handleSend = async () => {
    const text = input.trim();
    if ((!text && attachedImages.length === 0) || loading) return;

    const currentAttached = [...attachedImages];
    const imagePaths = currentAttached.map((i) => i.path);

    setInput("");
    setAttachedImages([]);

    const userMsg: ChatMessage = {
      role: "user",
      content: text || "(图片)",
      attachedImages: currentAttached.length > 0 ? currentAttached : undefined,
    };

    // Add user message and empty assistant message for streaming
    setMessages((prev) => [
      ...prev,
      userMsg,
      { role: "assistant", content: "", toolCalls: [], images: [] },
    ]);
    setLoading(true);
    hasNewImagesRef.current = false;

    try {
      await streamMessage(
        text || "请分析这些图片",
        conversationId,
        {
          onConversationId: (cid) => {
            setConversationId(cid);
          },
          onTextDelta: (delta) => {
            updateLastAssistant((msg) => ({
              ...msg,
              content: msg.content + delta,
            }));
          },
          onToolStart: (_index, tool, args) => {
            const tc: ToolCallInfo = { tool, args, pending: true };
            updateLastAssistant((msg) => ({
              ...msg,
              toolCalls: [...(msg.toolCalls || []), tc],
            }));
          },
          onToolEnd: (index, tool, result, durationMs, images) => {
            if (images && images.length > 0) {
              hasNewImagesRef.current = true;
            }
            updateLastAssistant((msg) => {
              const toolCalls = [...(msg.toolCalls || [])];
              // Find the pending tool call by index
              const tcIdx = toolCalls.findIndex(
                (tc) => tc.tool === tool && tc.pending
              );
              if (tcIdx >= 0) {
                toolCalls[tcIdx] = {
                  ...toolCalls[tcIdx],
                  result,
                  duration_ms: durationMs,
                  pending: false,
                  images,
                };
              }
              const newImages = images ? [...(msg.images || []), ...images] : msg.images;
              return { ...msg, toolCalls, images: newImages };
            });
          },
          onDone: () => {
            setLoading(false);
            if (hasNewImagesRef.current) {
              onNewImages?.();
            }
          },
          onError: (error) => {
            updateLastAssistant((msg) => ({
              ...msg,
              content: msg.content + `\n错误: ${error}`,
            }));
            setLoading(false);
          },
        },
        imagePaths.length > 0 ? imagePaths : undefined,
        project,
      );
    } catch (err) {
      updateLastAssistant((msg) => ({
        ...msg,
        content: msg.content + `\n错误: ${err}`,
      }));
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <div className="text-4xl mb-4">🎬</div>
            <div className="text-lg font-medium">AniDaily</div>
            <div className="text-sm mt-1">动画条漫生成助手</div>
            <div className="text-xs mt-4 text-gray-300">
              试试: &quot;帮我检测这张图中的人脸&quot; 或 &quot;生成一个穿红裙子的女孩角色&quot;
            </div>
            <div className="text-xs mt-1 text-gray-300">
              可以上传图片或点击侧边栏素材引用到对话中
            </div>
          </div>
        )}

        {messages.map((msg, i) => {
          // Skip empty assistant messages that haven't started streaming
          if (msg.role === "assistant" && !msg.content && (!msg.toolCalls || msg.toolCalls.length === 0) && !loading) {
            return null;
          }

          return (
            <div
              key={i}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-2.5 ${
                  msg.role === "user"
                    ? "bg-blue-500 text-white"
                    : "bg-white border border-gray-200 text-gray-800"
                }`}
              >
                {/* Attached images (user message) */}
                {msg.attachedImages && msg.attachedImages.length > 0 && (
                  <div className="mb-2 flex flex-wrap gap-1.5">
                    {msg.attachedImages.map((img, j) => (
                      <img
                        key={j}
                        src={img.previewUrl || getFileUrl(img.url)}
                        alt={img.name}
                        className="w-16 h-16 object-cover rounded-lg border border-blue-300"
                      />
                    ))}
                  </div>
                )}

                {/* Tool calls (shown before text for assistant) */}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="mb-2 space-y-1.5">
                    {msg.toolCalls.map((tc, j) => (
                      <ToolCallCard key={j} toolCall={tc} />
                    ))}
                  </div>
                )}

                {/* Text */}
                {msg.content && (
                  <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                )}

                {/* Loading indicator when streaming but no content yet */}
                {msg.role === "assistant" && loading && !msg.content && (!msg.toolCalls || msg.toolCalls.length === 0) && i === messages.length - 1 && (
                  <div className="flex space-x-1.5">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.1s]" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                  </div>
                )}

                {/* Generated images (assistant message) */}
                {msg.images && msg.images.length > 0 && (
                  <div className="mt-2 grid grid-cols-2 gap-2">
                    {msg.images.map((img, j) => (
                      <a
                        key={j}
                        href={getFileUrl(img.url)}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <img
                          src={getFileUrl(img.url)}
                          alt={img.tool}
                          className="rounded-lg border border-gray-200 max-h-64 object-contain"
                        />
                      </a>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area - Gemini style */}
      <div className="border-t border-gray-200 bg-white p-3">
        <div
          className={`border rounded-2xl transition-colors ${
            dragOver
              ? "border-blue-400 bg-blue-50"
              : "border-gray-300 focus-within:border-blue-400 focus-within:ring-1 focus-within:ring-blue-400"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          {/* Attached images inside input box */}
          {(attachedImages.length > 0 || uploading) && (
            <div className="px-3 pt-3 flex flex-wrap gap-2">
              {attachedImages.map((img, i) => (
                <div key={i} className="relative group">
                  <img
                    src={img.previewUrl || getFileUrl(img.url)}
                    alt={img.name}
                    className="w-16 h-16 object-cover rounded-xl border border-gray-200"
                  />
                  <button
                    onClick={() => removeAttached(i)}
                    className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-gray-600 text-white rounded-full text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity shadow-sm"
                  >
                    ×
                  </button>
                </div>
              ))}
              {uploading && (
                <div className="w-16 h-16 rounded-xl border border-dashed border-gray-300 flex items-center justify-center bg-gray-50">
                  <div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                </div>
              )}
            </div>
          )}

          {/* Drag overlay hint */}
          {dragOver && (
            <div className="px-3 pt-2 text-xs text-blue-500 text-center">
              松开以添加图片
            </div>
          )}

          {/* Text input */}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder="输入消息，粘贴或拖拽图片... (Enter 发送)"
            className="w-full resize-none border-0 bg-transparent px-4 py-2.5 text-sm focus:outline-none max-h-32"
            rows={1}
          />

          {/* Bottom toolbar */}
          <div className="flex items-center justify-between px-2 pb-2">
            <div className="flex items-center space-x-1">
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
                title="上传图片"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
            <button
              onClick={handleSend}
              disabled={loading || (!input.trim() && attachedImages.length === 0)}
              className="p-1.5 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              title="发送"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="19" x2="12" y2="5"/>
                <polyline points="5 12 12 5 19 12"/>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
