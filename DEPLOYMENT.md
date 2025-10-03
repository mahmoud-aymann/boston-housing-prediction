# 🚀 Vercel Deployment Guide

## Deploy Boston Housing Prediction to Vercel

### Prerequisites
- GitHub repository with the project
- Vercel account (free)
- Python 3.8+ (for local testing)

### Method 1: Deploy from GitHub (Recommended)

1. **Go to [Vercel Dashboard](https://vercel.com/dashboard)**
2. **Click "New Project"**
3. **Import from GitHub:**
   - Select your `boston-housing-prediction` repository
   - Choose the `master` branch
4. **Configure Project:**
   - Framework Preset: `Other`
   - Root Directory: `./` (leave as default)
   - Build Command: Leave empty
   - Output Directory: Leave empty
5. **Environment Variables:** (None needed for this project)
6. **Click "Deploy"**

### Method 2: Deploy using Vercel CLI

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel:**
   ```bash
   vercel login
   ```

3. **Deploy from project directory:**
   ```bash
   vercel
   ```

4. **Follow the prompts:**
   - Set up and deploy? `Y`
   - Which scope? (Select your account)
   - Link to existing project? `N`
   - Project name: `boston-housing-prediction`
   - Directory: `./`
   - Override settings? `N`

### Method 3: One-Click Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/mahmoud-aymann/boston-housing-prediction)

### Project Structure for Vercel

```
boston-housing-prediction/
├── api/
│   └── index.py          # Vercel API endpoint
├── public/
│   └── index.html        # Frontend interface
├── boston_housing_model.pkl  # Trained model
├── model_info.pkl        # Model metadata
├── requirements.txt      # Python dependencies
├── vercel.json          # Vercel configuration
└── README.md
```

### API Endpoints

After deployment, your app will have these endpoints:

- **`/`** - Main interface
- **`/api/`** - API information
- **`/api/predict`** - Make predictions (POST)
- **`/api/info`** - Model information (GET)
- **`/api/sample`** - Sample predictions (GET)

### Testing the Deployment

1. **Visit your Vercel URL** (e.g., `https://boston-housing-prediction.vercel.app`)
2. **Test the prediction form**
3. **Test API endpoints:**
   ```bash
   curl -X POST https://your-app.vercel.app/api/predict \
     -H "Content-Type: application/json" \
     -d '{"rm": 6, "lstat": 15, "ptratio": 15}'
   ```

### Troubleshooting

**Common Issues:**

1. **Model files not found:**
   - Ensure `boston_housing_model.pkl` and `model_info.pkl` are in the root directory
   - Run `python train_model.py` locally first

2. **Import errors:**
   - Check `requirements.txt` has all dependencies
   - Ensure Python version compatibility

3. **API not responding:**
   - Check Vercel function logs
   - Verify file paths in `api/index.py`

### Environment Variables (Optional)

If you need to add environment variables:

1. Go to Vercel Dashboard
2. Select your project
3. Go to Settings → Environment Variables
4. Add any required variables

### Custom Domain (Optional)

1. Go to Vercel Dashboard
2. Select your project
3. Go to Settings → Domains
4. Add your custom domain

### Performance Optimization

- **Model Loading:** Model is loaded once when the function starts
- **Caching:** Vercel automatically caches static files
- **Cold Starts:** First request might be slower due to model loading

### Monitoring

- **Vercel Analytics:** Built-in performance monitoring
- **Function Logs:** Available in Vercel Dashboard
- **Error Tracking:** Automatic error reporting

### Cost

- **Free Tier:** 100GB bandwidth, 1000 serverless function invocations
- **Pro Tier:** $20/month for more resources
- **Enterprise:** Custom pricing for large scale

### Support

- **Vercel Documentation:** https://vercel.com/docs
- **Community:** https://github.com/vercel/vercel/discussions
- **Status:** https://vercel-status.com

---

**Your app will be live at:** `https://your-project-name.vercel.app` 🎉
