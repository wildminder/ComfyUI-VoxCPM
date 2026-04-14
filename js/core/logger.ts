/**
 * VoxCPM Logger Utility
 * 
 * Provides consistent logging with prefix for all VoxCPM frontend modules.
 */

import { DEFAULTS } from "./constants";

// ============================================================================
// Logger Implementation
// ============================================================================

/**
 * Logger instance with prefixed console output
 */
export const logger = {
  /**
   * Log informational message
   */
  log: (...args: unknown[]): void => {
    console.log(DEFAULTS.LOG_PREFIX, ...args);
  },

  /**
   * Log informational message (alias for log)
   */
  info: (...args: unknown[]): void => {
    console.info(DEFAULTS.LOG_PREFIX, ...args);
  },

  /**
   * Log warning message
   */
  warn: (...args: unknown[]): void => {
    console.warn(DEFAULTS.LOG_PREFIX, ...args);
  },

  /**
   * Log error message
   */
  error: (...args: unknown[]): void => {
    console.error(DEFAULTS.LOG_PREFIX, ...args);
  },

  /**
   * Log debug message (only in development)
   */
  debug: (...args: unknown[]): void => {
    // Check for development mode (can be set via import.meta.env or window)
    const isDev = typeof window !== "undefined" && (window as unknown as Record<string, boolean>).__VOXCPM_DEBUG__;
    if (isDev) {
      console.debug(DEFAULTS.LOG_PREFIX, "[DEBUG]", ...args);
    }
  },

  /**
   * Log group start
   */
  group: (label: string): void => {
    console.group(`${DEFAULTS.LOG_PREFIX} ${label}`);
  },

  /**
   * Log group end
   */
  groupEnd: (): void => {
    console.groupEnd();
  },

  /**
   * Log table data
   */
  table: (data: unknown): void => {
    console.log(DEFAULTS.LOG_PREFIX);
    console.table(data);
  },
};

/**
 * Create a child logger with additional context prefix
 */
export function createChildLogger(context: string): typeof logger {
  const prefix = `${DEFAULTS.LOG_PREFIX}[${context}]`;
  return {
    log: (...args: unknown[]) => console.log(prefix, ...args),
    info: (...args: unknown[]) => console.info(prefix, ...args),
    warn: (...args: unknown[]) => console.warn(prefix, ...args),
    error: (...args: unknown[]) => console.error(prefix, ...args),
    debug: (...args: unknown[]) => {
      const isDev = typeof window !== "undefined" && (window as unknown as Record<string, boolean>).__VOXCPM_DEBUG__;
      if (isDev) {
        console.debug(prefix, "[DEBUG]", ...args);
      }
    },
    group: (label: string) => console.group(`${prefix} ${label}`),
    groupEnd: () => console.groupEnd(),
    table: (data: unknown) => {
      console.log(prefix);
      console.table(data);
    },
  };
}

export default logger;
