/**
 * VoxCPM Frontend Extension Entry Point
 * 
 * This file is the main entry point for the VoxCPM frontend extension.
 * It registers the extension with ComfyUI's extension system.
 * 
 * @module voxcpm-frontend
 */

import { app } from "../../scripts/app.js";
import { VoxCPMExtension } from "./extension";

// Register extension with ComfyUI
app.registerExtension(VoxCPMExtension);

// Log registration (for debugging)
console.log("[VoxCPM] Extension registered:", VoxCPMExtension.name);
