/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['localhost'],
    unoptimized: true
  },
  async rewrites() {
    return [
      {
        source: '/api/python/:path*',
        destination: 'http://localhost:8501/:path*',
      },
    ]
  }
}

module.exports = nextConfig