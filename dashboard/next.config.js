/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/trading/:path*',
        destination: 'http://trading-system:5001/:path*',
      },
      {
        source: '/api/metrics/:path*',
        destination: 'http://trading-system:9999/:path*',
      },
    ]
  },
}

module.exports = nextConfig