# Deployment Guide

This guide covers deploying the AI Calorie Analyzer frontend to various platforms.

## Build Process

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Type Check**
   ```bash
   npm run type-check
   ```

3. **Build for Production**
   ```bash
   npm run build:prod
   ```

4. **Preview Build Locally**
   ```bash
   npm run preview
   ```

## Deployment Options

### 1. Vercel (Recommended)

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy**
   ```bash
   vercel --prod
   ```

3. **Environment Variables**
   - `VITE_API_BASE_URL`: Backend API URL

### 2. Netlify

1. **Build Command**: `npm run build:prod`
2. **Publish Directory**: `dist`
3. **Environment Variables**:
   - `VITE_API_BASE_URL`: Backend API URL

### 3. GitHub Pages

1. **Install gh-pages**
   ```bash
   npm install --save-dev gh-pages
   ```

2. **Add to package.json**
   ```json
   {
     "homepage": "https://yourusername.github.io/food-analyzer-frontend",
     "scripts": {
       "deploy": "gh-pages -d dist"
     }
   }
   ```

3. **Deploy**
   ```bash
   npm run build:prod
   npm run deploy
   ```

### 4. Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM node:18-alpine as builder
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci
   COPY . .
   RUN npm run build:prod

   FROM nginx:alpine
   COPY --from=builder /app/dist /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/nginx.conf
   EXPOSE 80
   CMD ["nginx", "-g", "daemon off;"]
   ```

2. **Build and Run**
   ```bash
   docker build -t food-analyzer-frontend .
   docker run -p 80:80 food-analyzer-frontend
   ```

### 5. AWS S3 + CloudFront

1. **Build the project**
   ```bash
   npm run build:prod
   ```

2. **Upload to S3**
   ```bash
   aws s3 sync dist/ s3://your-bucket-name --delete
   ```

3. **Configure CloudFront** for SPA routing

## Environment Variables

Create a `.env` file for local development:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_NAME=AI Calorie Analyzer
VITE_APP_VERSION=1.0.0
```

For production, set these in your deployment platform:

- **VITE_API_BASE_URL**: Your backend API URL
- **VITE_APP_NAME**: Application name
- **VITE_APP_VERSION**: Version number

## Performance Optimizations

### 1. Bundle Analysis
```bash
npm install --save-dev vite-bundle-analyzer
```

Add to vite.config.ts:
```typescript
import { defineConfig } from 'vite'
import { analyzer } from 'vite-bundle-analyzer'

export default defineConfig({
  plugins: [
    // ... other plugins
    analyzer()
  ]
})
```

### 2. Code Splitting
The app already uses dynamic imports for optimal code splitting:
- Components are lazy-loaded
- Services are imported on-demand
- Charts library is loaded when needed

### 3. Image Optimization
- Images are automatically optimized during build
- WebP format is preferred when supported
- Lazy loading is implemented for images

### 4. Caching Strategy
- Static assets are cached with long expiration
- API responses are cached in localStorage
- Service worker can be added for offline support

## Monitoring and Analytics

### 1. Error Tracking
Add Sentry for error tracking:
```bash
npm install @sentry/react @sentry/tracing
```

### 2. Performance Monitoring
Add Web Vitals tracking:
```bash
npm install web-vitals
```

### 3. Analytics
Add Google Analytics or similar:
```bash
npm install gtag
```

## Security Considerations

1. **Content Security Policy**
   ```html
   <meta http-equiv="Content-Security-Policy" 
         content="default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline';">
   ```

2. **HTTPS Only**
   - Always deploy with HTTPS
   - Use HSTS headers

3. **Environment Variables**
   - Never commit sensitive data
   - Use platform-specific secret management

## Troubleshooting

### Common Issues

1. **Build Fails**
   - Check TypeScript errors: `npm run type-check`
   - Clear node_modules: `rm -rf node_modules && npm install`

2. **Routing Issues**
   - Configure server for SPA routing
   - Check base URL in vite.config.ts

3. **API Connection**
   - Verify CORS settings on backend
   - Check environment variables

4. **Performance Issues**
   - Analyze bundle size
   - Check for memory leaks
   - Optimize images and assets

### Debug Mode

Enable debug mode by setting:
```env
VITE_DEBUG=true
```

This will:
- Show detailed error messages
- Enable console logging
- Display performance metrics

## Maintenance

### Regular Updates
1. Update dependencies monthly
2. Monitor security vulnerabilities
3. Test on different browsers/devices
4. Review performance metrics

### Backup Strategy
1. Version control (Git)
2. Database backups (if applicable)
3. Asset backups
4. Configuration backups