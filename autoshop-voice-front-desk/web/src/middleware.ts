import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { jwtVerify } from "jose";

const PUBLIC_PATHS = [
  "/",
  "/login",
  "/signup",
  "/api/auth/login",
  "/api/auth/signup",
  "/api/stripe/webhook",
  "/api/dms/webhook",
];

// Allowed static file extensions
const STATIC_EXT = /\.(js|css|png|jpg|jpeg|gif|svg|ico|webp|woff|woff2|ttf|eot|map)$/;

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Allow public paths
  if (PUBLIC_PATHS.some((p) => pathname === p || pathname.startsWith(p + "/"))) {
    return NextResponse.next();
  }

  // Allow static assets (strict extension check)
  if (pathname.startsWith("/_next") || pathname.startsWith("/favicon") || STATIC_EXT.test(pathname)) {
    return NextResponse.next();
  }

  // Validate session JWT (not just cookie existence)
  const sessionCookie = request.cookies.get("session");
  if (!sessionCookie?.value) {
    return redirectToLogin(request, pathname);
  }

  try {
    const secret = process.env.JWT_SECRET;
    if (!secret) {
      return redirectToLogin(request, pathname);
    }
    const jwtSecret = new TextEncoder().encode(secret);
    const { payload } = await jwtVerify(sessionCookie.value, jwtSecret);

    if (!payload.userId) {
      return redirectToLogin(request, pathname);
    }

    return NextResponse.next();
  } catch {
    // Token invalid or expired
    return redirectToLogin(request, pathname, true);
  }
}

function redirectToLogin(request: NextRequest, from: string, expired = false) {
  const loginUrl = new URL("/login", request.url);
  loginUrl.searchParams.set("from", from);
  if (expired) {
    loginUrl.searchParams.set("expired", "true");
  }
  return NextResponse.redirect(loginUrl);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
