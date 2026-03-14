import { useState, useEffect, useCallback } from "react";
import AssetSidebar from "./components/AssetSidebar";
import ChatPanel from "./components/ChatPanel";
import PreviewPanel from "./components/PreviewPanel";
import ProjectSelector from "./components/ProjectSelector";
import type { Asset, Assets, AttachedImage, Project } from "./api";
import { fetchAssets, fetchProjects, createProject, deleteProject } from "./api";

interface PreviewInfo {
  asset: Asset;
  category: string;
}

function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentProject, setCurrentProject] = useState<string | null>(null);
  const [assets, setAssets] = useState<Assets>({});
  const [pendingAssets, setPendingAssets] = useState<AttachedImage[]>([]);
  const [preview, setPreview] = useState<PreviewInfo | null>(null);

  const loadProjects = useCallback(async () => {
    try {
      const data = await fetchProjects();
      setProjects(data);
    } catch (err) {
      console.error("Failed to load projects:", err);
    }
  }, []);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  const loadAssets = useCallback(async () => {
    if (!currentProject) {
      setAssets({});
      return;
    }
    try {
      const data = await fetchAssets(currentProject);
      setAssets(data);
    } catch (err) {
      console.error("Failed to load assets:", err);
    }
  }, [currentProject]);

  useEffect(() => {
    loadAssets();
  }, [loadAssets]);

  const handleCreateProject = async (name: string) => {
    await createProject(name);
    await loadProjects();
    setCurrentProject(name);
  };

  const handleDeleteProject = async (name: string) => {
    await deleteProject(name);
    setCurrentProject(null);
    setAssets({});
    await loadProjects();
  };

  const handleAssetClick = (path: string) => {
    // Toggle preview: click same asset again to close
    if (preview?.asset.path === path) {
      setPreview(null);
      return;
    }
    for (const [category, items] of Object.entries(assets)) {
      const found = items.find((a) => a.path === path);
      if (found) {
        setPreview({ asset: found, category });
        return;
      }
    }
  };

  const handleSendToChat = () => {
    if (!preview) return;
    const { asset } = preview;
    setPendingAssets((prev) => {
      if (prev.some((a) => a.path === asset.path)) return prev;
      return [
        ...prev,
        {
          path: asset.path,
          url: asset.url,
          name: asset.name,
          fileType: asset.type || "image",
          content: asset.content,
          description: asset.description,
        },
      ];
    });
    setPreview(null);
  };

  const handlePendingAssetClick = (path: string) => {
    for (const [category, items] of Object.entries(assets)) {
      const found = items.find((a) => a.path === path);
      if (found) {
        setPreview({ asset: found, category });
        return;
      }
    }
  };

  return (
    <div className="flex flex-col h-screen">
      <ProjectSelector
        projects={projects}
        current={currentProject}
        onSelect={setCurrentProject}
        onCreate={handleCreateProject}
        onDelete={handleDeleteProject}
      />
      <div className="flex flex-1 min-h-0">
        <AssetSidebar
          assets={assets}
          onRefresh={loadAssets}
          onAssetClick={handleAssetClick}
        />
        <div className="flex-1 bg-gray-50">
          {currentProject ? (
            <ChatPanel
              project={currentProject}
              onNewImages={loadAssets}
              pendingAssets={pendingAssets}
              onClearPendingAssets={() => setPendingAssets([])}
              onPendingAssetClick={handlePendingAssetClick}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              <div className="text-center">
                <div className="text-4xl mb-4">📁</div>
                <div className="text-lg">请选择或新建一个项目</div>
              </div>
            </div>
          )}
        </div>
        {preview && (
          <PreviewPanel
            asset={preview.asset}
            category={preview.category}
            onClose={() => setPreview(null)}
            onSendToChat={handleSendToChat}
            onAssetUpdated={loadAssets}
          />
        )}
      </div>
    </div>
  );
}

export default App;
