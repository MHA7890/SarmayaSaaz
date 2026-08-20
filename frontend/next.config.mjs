/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Proxy API calls through Next in development so the browser never deals
  // with cross-origin requests and NEXT_PUBLIC_API_URL can stay relative.
  // /top-movers became the dashboard at / - keep old links working.
  async redirects() {
    return [{ source: "/top-movers", destination: "/", permanent: true }];
  },
  async rewrites() {
    const target = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${target}/api/:path*` }];
  },
};

export default nextConfig;
