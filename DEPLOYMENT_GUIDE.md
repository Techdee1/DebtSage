# DebtSage API - Deployment Guide

## 🚀 Quick Deployment Options

Your DebtSage API is now on GitHub and can be accessed by your software engineer. Here are the deployment options:

---

## Option 1: GitHub Codespaces (Easiest - Recommended)

**Best for:** Quick access, no local setup needed

### Steps:
1. Go to: https://github.com/Techdee1/10Analytics
2. Click **Code** → **Codespaces** → **Create codespace on main**
3. Wait for environment to load (2-3 minutes)
4. In the terminal, run:
   ```bash
   cd /workspaces/10Analytics
   pip install -r requirements.txt
   python app/api.py
   ```
5. Codespace will show a port forwarding notification
6. Click "Open in Browser" or copy the forwarded URL
7. Share the public URL with your engineer

**URL Format:** `https://[codespace-name]-8000.app.github.dev`

**Pros:**
- ✅ No local installation needed
- ✅ Automatic HTTPS
- ✅ Can make public for collaboration
- ✅ 60 hours/month free

**Cons:**
- ⏰ Auto-stops after inactivity
- 🌐 Requires GitHub account

---

## Option 2: Render.com (Free Hosting)

**Best for:** Permanent public API, 24/7 availability

### Steps:

1. **Create Render Account:**
   - Go to https://render.com
   - Sign up with GitHub

2. **Create New Web Service:**
   - Click "New" → "Web Service"
   - Connect GitHub: `Techdee1/10Analytics`
   - Settings:
     - **Name:** `debtsage-api`
     - **Root Directory:** Leave blank
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
     - **Plan:** Free

3. **Deploy:**
   - Click "Create Web Service"
   - Wait 5-10 minutes for first build
   - You'll get a URL like: `https://debtsage-api.onrender.com`

4. **Share URL:**
   - API: `https://debtsage-api.onrender.com`
   - Docs: `https://debtsage-api.onrender.com/docs`
   - Health: `https://debtsage-api.onrender.com/health`

**Pros:**
- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ 24/7 availability
- ✅ Auto-deploys on git push
- ✅ Custom domain support

**Cons:**
- 🐌 Free tier sleeps after inactivity (15 min to wake)
- 💾 750 hours/month limit

---

## Option 3: Railway.app (Free)

**Best for:** Fast deployment, generous free tier

### Steps:

1. **Create Railway Account:**
   - Go to https://railway.app
   - Sign up with GitHub

2. **Deploy:**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select `Techdee1/10Analytics`
   - Railway auto-detects Python
   - Add start command: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`

3. **Share URL:**
   - Railway generates: `https://[project-name].up.railway.app`
   - Docs: `https://[project-name].up.railway.app/docs`

**Pros:**
- ✅ $5/month free credit
- ✅ No sleep on inactivity
- ✅ Fast deployment
- ✅ Good performance

**Cons:**
- 💰 Limited free usage

---

## Option 4: Heroku (Classic Option)

**Best for:** Enterprise-grade hosting

### Steps:

1. **Install Heroku CLI:**
   ```bash
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. **Login and Create App:**
   ```bash
   heroku login
   cd /workspaces/10Analytics
   heroku create debtsage-api
   ```

3. **Create Procfile:**
   ```bash
   echo "web: uvicorn app.api:app --host 0.0.0.0 --port \$PORT" > Procfile
   ```

4. **Deploy:**
   ```bash
   git add Procfile
   git commit -m "Add Procfile for Heroku"
   git push heroku main
   ```

5. **Share URL:**
   - `https://debtsage-api.herokuapp.com`
   - Docs: `https://debtsage-api.herokuapp.com/docs`

**Pros:**
- ✅ Reliable
- ✅ Good documentation
- ✅ Add-ons available

**Cons:**
- 💰 No free tier anymore

---

## Option 5: Local Development (Your Machine)

**Best for:** Testing locally, private access

### Steps:

1. **Clone Repository:**
   ```bash
   git clone https://github.com/Techdee1/10Analytics.git
   cd 10Analytics
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start API:**
   ```bash
   python app/api.py
   ```

4. **Access Locally:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

5. **Share with Engineer (Same Network):**
   - Find your IP: `hostname -I` or `ipconfig`
   - Share: `http://[YOUR_IP]:8000`

**To Make Accessible Outside Network (ngrok):**
```bash
# Install ngrok
brew install ngrok  # macOS
# or download from ngrok.com

# Start tunnel
ngrok http 8000

# Share the public URL (e.g., https://abc123.ngrok.io)
```

**Pros:**
- ✅ Full control
- ✅ Instant updates
- ✅ No deployment delay

**Cons:**
- 💻 Must keep computer running
- 🌐 Not accessible when offline

---

## Option 6: DigitalOcean App Platform

**Best for:** Production deployment

### Steps:

1. **Create DigitalOcean Account:**
   - Go to https://www.digitalocean.com

2. **Create App:**
   - Apps → Create App
   - Connect GitHub: `Techdee1/10Analytics`
   - Detected as Python app
   - Start command: `uvicorn app.api:app --host 0.0.0.0 --port 8080`
   - Plan: Basic ($5/month, or $200 free credit)

3. **Deploy:**
   - App URL: `https://debtsage-api-[hash].ondigitalocean.app`

**Pros:**
- ✅ Production-grade
- ✅ $200 free credit for new users
- ✅ Scalable

**Cons:**
- 💰 Paid after free tier

---

## 🎯 Recommended Quick Start

### For Immediate Access (5 minutes):

**Use GitHub Codespaces:**

1. Share this link with your engineer:
   ```
   https://github.com/Techdee1/10Analytics
   ```

2. Tell them to:
   - Click **Code** → **Codespaces** → **Create codespace**
   - Run in terminal:
     ```bash
     pip install -r requirements.txt
     python app/api.py
     ```
   - Access forwarded port URL

### For Permanent Public API (10 minutes):

**Use Render.com:**

1. You deploy once on Render.com (free)
2. Share permanent URL with engineer:
   ```
   https://debtsage-api.onrender.com
   ```
3. They can start integrating immediately

---

## 📝 What to Share with Your Engineer

Send them:

1. **Repository URL:**
   ```
   https://github.com/Techdee1/10Analytics
   ```

2. **API Documentation:**
   - Complete guide: `app/API_DOCUMENTATION.md`
   - Quick start: `app/API_QUICKSTART.md`
   - Interactive docs: `[YOUR_API_URL]/docs`

3. **Deployment URL:**
   - Once deployed, share: `https://[your-deployment-url]`
   - Health check: `https://[your-deployment-url]/health`
   - Docs: `https://[your-deployment-url]/docs`

4. **Quick Test:**
   ```bash
   curl https://[your-deployment-url]/health
   ```

---

## 🔒 Security Notes

### For Production:

1. **Update CORS in `app/api.py`:**
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=[
           "https://yourdomain.com",
           "https://dashboard.yourdomain.com"
       ],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Add API Key Authentication (Optional):**
   ```python
   from fastapi import Security, HTTPException
   from fastapi.security import APIKeyHeader
   
   API_KEY = "your-secret-key"
   api_key_header = APIKeyHeader(name="X-API-Key")
   
   async def verify_api_key(api_key: str = Security(api_key_header)):
       if api_key != API_KEY:
           raise HTTPException(status_code=403, detail="Invalid API Key")
   ```

3. **Use Environment Variables:**
   - Never commit secrets
   - Use `.env` file locally
   - Set environment variables in hosting platform

---

## 📊 Monitoring

Once deployed, monitor:

- **Health Endpoint:** `GET /health`
- **Response Times:** Should be <100ms for predictions
- **Error Rates:** Check logs on hosting platform

---

## 🆘 Troubleshooting

### API Won't Start:
```bash
# Check dependencies
pip install -r requirements.txt

# Verify models exist
ls models/

# Check data files
ls data/
```

### Port Already in Use:
```bash
# Use different port
uvicorn app.api:app --port 8001
```

### CORS Errors:
- Update `allow_origins` in `app/api.py`
- Or use `allow_origins=["*"]` for development

---

## 🎉 Success Checklist

✅ Repository pushed to GitHub  
✅ API documentation complete  
✅ Choose deployment method  
✅ Deploy API  
✅ Test health endpoint  
✅ Share URL with engineer  
✅ Engineer can access `/docs`  
✅ Engineer makes successful prediction  

---

**Your DebtSage API is ready to deploy!** 🚀

Choose a deployment method and share the URL with your software engineer. They can start building their dashboard immediately.

**Last Updated:** November 29, 2025
