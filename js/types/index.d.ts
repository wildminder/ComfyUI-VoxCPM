/**
 * ComfyUI API Type Definitions
 * 
 * These types provide TypeScript support for ComfyUI's frontend API.
 * Based on ComfyUI V3 extension system.
 */

// ============================================================================
// ComfyApp Types
// ============================================================================

declare module "../../scripts/app.js" {
  import type { ComfyApi, ExtensionManager, LGraph, LGraphCanvas } from "./index";

  export interface ComfyApp {
    api: ComfyApi;
    extensionManager: ExtensionManager;
    graph: LGraph;
    rootGraph: LGraph;
    canvas: LGraphCanvas;
    canvasEl: HTMLCanvasElement;
    nodeOutputs: Record<string, unknown>;
    nodePreviewImages: Record<string, unknown[]>;
    configuringGraph: boolean;
    vueAppReady: boolean;
    bodyTop: HTMLElement;
    bodyLeft: HTMLElement;
    bodyRight: HTMLElement;
    bodyBottom: HTMLElement;
    canvasContainer: HTMLElement;

    registerExtension(extension: ComfyExtension): void;
    loadGraphData(graphData?: unknown, clean?: boolean, restore_view?: boolean, workflow?: unknown, options?: unknown): Promise<void>;
    graphToPrompt(graph?: LGraph): Promise<Record<string, unknown>>;
    queuePrompt(number: number, batchCount?: number, queueNodeIds?: number[]): Promise<void>;
    refreshComboInNodes(): void;
    clean(): void;
    getNodeDefs(): Promise<Record<string, unknown>>;
    handleFile(file: File, source?: unknown): Promise<void>;
    handleFileList(fileList: FileList): Promise<void>;
    clientPosToCanvasPos(pos: { x: number; y: number }): { x: number; y: number };
    canvasPosToClientPos(pos: { x: number; y: number }): { x: number; y: number };
  }

  export const app: ComfyApp;
  export { ComfyApp };
}

// ============================================================================
// ComfyApi Types
// ============================================================================

declare module "../../scripts/api.js" {
  export interface ComfyApi extends EventTarget {
    api_host: string;
    api_base: string;
    initialClientId: string;
    clientId: string;
    user: string;
    socket: WebSocket | null;
    authToken: string | null;
    apiKey: string | null;

    addEventListener(type: string, callback: (event: CustomEvent) => void): void;
    removeEventListener(type: string, callback: (event: CustomEvent) => void): void;
    fetchApi(route: string, options?: RequestInit): Promise<Response>;
    getNodeDefs(): Promise<Record<string, unknown>>;
    getExtensions(): Promise<string[]>;
    queuePrompt(number: number, data: unknown, options?: unknown): Promise<unknown>;
    interrupt(runningJobId?: string): Promise<void>;
    getQueue(): Promise<unknown>;
    getHistory(): Promise<unknown>;
    getSettings(): Promise<Record<string, unknown>>;
    storeSettings(settings: Record<string, unknown>): Promise<void>;
    getSystemStats(): Promise<unknown>;
    getFolderPaths(): Promise<Record<string, string[]>>;
    serverSupportsFeature(name: string): boolean;
    getServerFeature(name: string, defaultValue?: unknown): unknown;
  }

  export const api: ComfyApi;
  export { ComfyApi };
}

// ============================================================================
// Extension Manager Types
// ============================================================================

export interface ExtensionManager {
  toast: ToastManager;
  dialog: DialogManager;
  setting: SettingManager;
  registerSidebarTab(tab: SidebarTab): void;
  renderMarkdownToHtml(markdown: string, baseUrl?: string): string;
  lastNodeErrors: Record<string, unknown> | null;
  lastExecutionError: ExecutionError | null;
}

export interface ToastManager {
  add(options: ToastOptions): void;
}

export interface ToastOptions {
  severity: "success" | "info" | "warn" | "error";
  summary: string;
  detail?: string;
  life?: number;
  closable?: boolean;
}

export interface DialogManager {
  confirm(options: DialogOptions): Promise<boolean>;
  prompt(options: PromptDialogOptions): Promise<string | null>;
}

export interface DialogOptions {
  title?: string;
  message: string;
}

export interface PromptDialogOptions extends DialogOptions {
  defaultValue?: string;
}

export interface SettingManager {
  get(id: string): unknown;
  set(id: string, value: unknown): void;
}

// ============================================================================
// Extension Interface
// ============================================================================

export interface ComfyExtension {
  name: string;
  init?(app: ComfyApp): Promise<void> | void;
  setup?(app: ComfyApp): Promise<void> | void;
  beforeRegisterNodeDef?(nodeType: unknown, nodeData: unknown, app: ComfyApp): Promise<void> | void;
  nodeCreated?(node: unknown, app: ComfyApp): void;
  loadedGraphNode?(node: unknown, app: ComfyApp): void;
  beforeConfigureGraph?(graphData: unknown, missingNodeTypes: string[], app: ComfyApp): Promise<void> | void;
  afterConfigureGraph?(missingNodeTypes: string[], app: ComfyApp): Promise<void> | void;
  getCustomWidgets?(app: ComfyApp): Record<string, CustomWidgetConstructor>;
  getCanvasMenuItems?(canvas: LGraphCanvas): MenuItem[];
  getNodeMenuItems?(node: unknown): MenuItem[];
  commands?: Command[];
  settings?: Setting[];
  keybindings?: Keybinding[];
  menuCommands?: MenuCommand[];
  sidebarTabs?: SidebarTab[];
  bottomPanelTabs?: BottomPanelTab[];
  aboutPageBadges?: AboutPageBadge[];
  topbarBadges?: TopbarBadge[];
  actionBarButtons?: ActionBarButton[];
}

export type CustomWidgetConstructor = (
  node: unknown,
  inputName: string,
  inputData: unknown,
  app: ComfyApp
) => { widget: unknown; minWidth?: number; minHeight?: number };

export interface MenuItem {
  content?: string;
  disabled?: boolean;
  has_submenu?: boolean;
  separator?: boolean;
  callback?: () => void;
}

export interface Command {
  id: string;
  label: string;
  icon?: string;
  function?: () => void;
}

export interface Setting {
  id: string;
  name: string;
  type: string;
  defaultValue?: unknown;
  onChange?: (value: unknown) => void;
  options?: unknown;
}

export interface Keybinding {
  commandId: string;
  combo: { key: string; ctrl?: boolean; shift?: boolean; alt?: boolean };
}

export interface MenuCommand {
  path: string[];
  commands: string[];
}

export interface SidebarTab {
  id: string;
  title: string;
  icon: string;
  type: "custom" | "vue";
  render?: (container: HTMLElement) => void;
  destroy?: () => void;
}

export interface BottomPanelTab {
  id: string;
  title: string;
  type: "custom" | "vue";
  render?: (container: HTMLElement) => void;
}

export interface AboutPageBadge {
  label: string;
  url?: string;
  icon?: string;
  severity?: "danger" | "warn";
}

export interface TopbarBadge {
  text: string;
  label?: string;
  variant: "info" | "warning" | "error";
  icon?: string;
  tooltip?: string;
}

export interface ActionBarButton {
  icon: string;
  label?: string;
  tooltip?: string;
  onClick: () => void;
}

// ============================================================================
// Graph Types
// ============================================================================

export interface LGraph {
  _nodes: LGraphNode[];
  add(node: LGraphNode): void;
  remove(node: LGraphNode): void;
  getNodeById(id: number): LGraphNode | undefined;
}

export interface LGraphNode {
  id: number;
  type: string;
  title: string;
  pos: [number, number];
  size: [number, number];
  flags: Record<string, unknown>;
  mode: number;
  inputs: unknown[];
  outputs: unknown[];
  widgets: unknown[];
  comfyClass?: string;
  isVirtualNode?: boolean;
  imgs?: unknown[];
  imageIndex?: number;
}

export interface LGraphCanvas {
  canvas: HTMLCanvasElement;
  graph: LGraph;
  selected_nodes: Record<number, LGraphNode>;
  selected_node: LGraphNode | null;
}

// ============================================================================
// Error Types
// ============================================================================

export interface ExecutionError {
  prompt_id: string;
  node_id: string;
  node_type?: string;
  exception_message: string;
  exception_type: string;
  traceback: string[];
}

// ============================================================================
// Export all types
// ============================================================================

export type {
  ComfyApp,
  ComfyApi,
  ExtensionManager,
  ToastManager,
  ToastOptions,
  DialogManager,
  DialogOptions,
  PromptDialogOptions,
  SettingManager,
  ComfyExtension,
  CustomWidgetConstructor,
  MenuItem,
  Command,
  Setting,
  Keybinding,
  MenuCommand,
  SidebarTab,
  BottomPanelTab,
  AboutPageBadge,
  TopbarBadge,
  ActionBarButton,
  LGraph,
  LGraphNode,
  LGraphCanvas,
  ExecutionError,
};
