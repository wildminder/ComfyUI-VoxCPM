/**
 * VoxCPM Configuration Types
 * 
 * Configuration constants and types for the VoxCPM frontend extension.
 */

// ============================================================================
// Configuration Constants
// ============================================================================

/**
 * VoxCPM frontend configuration
 */
export const VOXCPM_CONFIG = {
  /** Log prefix for console output */
  logPrefix: "[VoxCPM]",

  /** Session storage key for notification shown flag */
  notificationShownKey: "voxcpm.normalization_notification_shown",

  /** Default toast notification lifetime in milliseconds */
  defaultToastLife: 10000,

  /** Extension name for registration */
  extensionName: "voxcpm.frontend",
} as const;

// ============================================================================
// Types
// ============================================================================

/**
 * VoxCPM configuration interface
 */
export interface VoxCPMConfig {
  logPrefix: string;
  notificationShownKey: string;
  defaultToastLife: number;
  extensionName: string;
}

/**
 * Feature availability status (synced with backend)
 */
export interface VoxCPMFeatureStatus {
  /** Whether text normalization packages are available */
  textNormalizationAvailable: boolean;

  /** Whether VoxCPM2 model is loaded */
  voxcpm2ModelLoaded?: boolean;

  /** Whether LoRA is active */
  loraActive?: boolean;
}

/**
 * Voice preset for future voice library feature
 */
export interface VoicePreset {
  id: string;
  name: string;
  description: string;
  controlInstruction: string;
  language?: string;
  gender?: "male" | "female" | "neutral";
  tags?: string[];
}

// Types are already exported via interface declarations above
