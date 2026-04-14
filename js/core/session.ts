/**
 * VoxCPM Session State Management
 * 
 * Provides utilities for managing session-scoped state using sessionStorage.
 * Session state persists until the browser tab/window is closed.
 */

// ============================================================================
// Session Storage Utilities
// ============================================================================

/**
 * Get a value from session storage
 * 
 * @param key - Storage key
 * @returns Stored value or null if not found
 */
export function getSessionItem(key: string): string | null {
  try {
    return sessionStorage.getItem(key);
  } catch {
    // sessionStorage not available (e.g., private browsing mode)
    return null;
  }
}

/**
 * Set a value in session storage
 * 
 * @param key - Storage key
 * @param value - Value to store
 * @returns True if successful, false if storage not available
 */
export function setSessionItem(key: string, value: string): boolean {
  try {
    sessionStorage.setItem(key, value);
    return true;
  } catch {
    // sessionStorage not available
    return false;
  }
}

/**
 * Remove a value from session storage
 * 
 * @param key - Storage key
 * @returns True if successful, false if storage not available
 */
export function removeSessionItem(key: string): boolean {
  try {
    sessionStorage.removeItem(key);
    return true;
  } catch {
    return false;
  }
}

/**
 * Clear all VoxCPM-related items from session storage
 * 
 * @param prefix - Key prefix to match (default: "voxcpm.")
 * @returns Number of items cleared
 */
export function clearSessionItems(prefix = "voxcpm."): number {
  try {
    const keysToRemove: string[] = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.startsWith(prefix)) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach((key) => sessionStorage.removeItem(key));
    return keysToRemove.length;
  } catch {
    return 0;
  }
}

// ============================================================================
// Flag Helpers
// ============================================================================

/**
 * Check if a session flag is set
 * 
 * @param key - Storage key for the flag
 * @returns True if flag is set
 */
export function isSessionFlagSet(key: string): boolean {
  return getSessionItem(key) !== null;
}

/**
 * Set a session flag
 * 
 * @param key - Storage key for the flag
 * @returns True if successful
 */
export function setSessionFlag(key: string): boolean {
  return setSessionItem(key, "true");
}

/**
 * Clear a session flag
 * 
 * @param key - Storage key for the flag
 * @returns True if successful
 */
export function clearSessionFlag(key: string): boolean {
  return removeSessionItem(key);
}

// ============================================================================
// JSON Helpers
// ============================================================================

/**
 * Get and parse a JSON value from session storage
 * 
 * @param key - Storage key
 * @returns Parsed value or null if not found/invalid
 */
export function getSessionJSON<T>(key: string): T | null {
  const value = getSessionItem(key);
  if (value === null) {
    return null;
  }
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

/**
 * Stringify and set a JSON value in session storage
 * 
 * @param key - Storage key
 * @param value - Value to store
 * @returns True if successful
 */
export function setSessionJSON<T>(key: string, value: T): boolean {
  try {
    return setSessionItem(key, JSON.stringify(value));
  } catch {
    return false;
  }
}

// ============================================================================
// Session State Class
// ============================================================================

/**
 * Session state manager for a specific feature
 */
export class SessionState<T> {
  private key: string;
  private defaultValue: T;

  constructor(key: string, defaultValue: T) {
    this.key = key;
    this.defaultValue = defaultValue;
  }

  /**
   * Get current value
   */
  get(): T {
    const stored = getSessionJSON<T>(this.key);
    return stored ?? this.defaultValue;
  }

  /**
   * Set value
   */
  set(value: T): boolean {
    return setSessionJSON(this.key, value);
  }

  /**
   * Clear value (reset to default)
   */
  clear(): boolean {
    return removeSessionItem(this.key);
  }

  /**
   * Check if value exists
   */
  exists(): boolean {
    return getSessionItem(this.key) !== null;
  }
}
