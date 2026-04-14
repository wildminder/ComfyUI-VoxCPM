/**
 * Custom Widgets Module
 * 
 * This module provides custom widget implementations for VoxCPM nodes.
 * 
 * @module voxcpm-widgets
 * 
 * ## Available Widgets (Future Implementation)
 * 
 * - AudioPreviewWidget: Preview audio directly on the node
 * - VoicePresetWidget: Dropdown for voice presets
 * - ProgressWidget: Custom progress visualization
 * 
 * ## Usage
 * 
 * ```typescript
 * import { AudioPreviewWidget } from "./widgets";
 * 
 * // In beforeRegisterNodeDef:
 * AudioPreviewWidget.create(node, "preview");
 * ```
 * 
 * ## Development Guide
 * 
 * To create a new widget:
 * 
 * 1. Create a new file in this directory (e.g., `MyWidget.ts`)
 * 2. Implement the widget class with a static `create` method
 * 3. Export from this index.ts file
 * 4. Use in `extension.ts` via `getCustomWidgets` hook
 */

// Placeholder for future widget implementations
// export { AudioPreviewWidget } from "./AudioPreviewWidget";
// export { VoicePresetWidget } from "./VoicePresetWidget";

export {};
