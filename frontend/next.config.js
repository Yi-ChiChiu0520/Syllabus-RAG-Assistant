
/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // Use environment variable for the backend URL
    const backendUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
    
    console.log('Next.js proxy configured for:', backendUrl);
    
    return [
      { 
        source: "/api/:path*", 
        destination: `${backendUrl}/:path*` 
      },
    ];
  },
};

module.exports = nextConfig;