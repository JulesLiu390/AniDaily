import { useState, useEffect, useCallback } from "react";
import AssetSidebar from "./components/AssetSidebar";
import ChatPanel from "./components/ChatPanel";
import ProjectSelector from "./components/ProjectSelector";
import type { Assets, AttachedImage, Project } from "./api";
import { fetchAssets, fetchProjects, createProject, deleteProject } from "./api";

function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentProject, setCurrentProject] = useState<string | null>(null);
  const [assets, setAssets] = useState<Assets>({});
  const [pendingAssets, setPendingAssets] = useState<AttachedImage[]>([]);

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
    for (const items of Object.values(assets)) {
      const found = items.find((a) => a.path === path && a.type === "image");
      if (found) {
        setPendingAssets((prev) => [
          ...prev,
          { path: found.path, url: found.url, name: found.name },
        ]);
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
      </div>
    </div>
  );
}

export default App;
