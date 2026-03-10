import { drizzle, type PostgresJsDatabase } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema";

let _db: PostgresJsDatabase<typeof schema> | null = null;

export function getDb() {
  if (!_db) {
    const databaseUrl = process.env.DATABASE_URL;
    if (!databaseUrl) {
      throw new Error(
        "FATAL: DATABASE_URL environment variable is not set. " +
          "Cannot initialize database connection."
      );
    }

    try {
      const client = postgres(databaseUrl, { prepare: false });
      _db = drizzle(client, { schema });
    } catch {
      throw new Error("Database connection failed. Check DATABASE_URL configuration.");
    }
  }
  return _db;
}

/** Lazy-initialized proxy — safe at build time when DATABASE_URL is absent */
export const db = new Proxy({} as PostgresJsDatabase<typeof schema>, {
  get(_, prop) {
    return (getDb() as any)[prop];
  },
});
