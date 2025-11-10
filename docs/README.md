# Data Science Portfolio Website

Professional portfolio website showcasing data science and machine learning projects.

🌐 **Live Site:** https://godsonkurishinkal.github.io/data-science-portfolio/

## Overview

This is a modern, responsive portfolio website built with vanilla HTML, CSS, and JavaScript. It features interactive project showcases, smooth animations, and a professional design optimized for both desktop and mobile devices.

### Features

- ✨ Modern, clean design with smooth animations
- 📱 Fully responsive (desktop, tablet, mobile)
- 🚀 Fast loading with optimized assets
- ♿ Accessible (WCAG 2.1 compliant)
- 🎨 Interactive UI with hover effects
- 📊 Project showcases with detailed pages
- 📧 Contact form integration
- 🌓 Dark mode toggle (optional)
- 📈 Google Analytics ready

## Projects Showcased

### 1. Dynamic Pricing Engine 💰
**Status:** In Progress (Phase 3/10 Complete)

Advanced ML-powered pricing optimization system using:
- Price elasticity analysis (econometrics)
- Demand forecasting (XGBoost)
- Revenue optimization (OR techniques)
- Competitive analysis
- Markdown strategies

**Tech:** Python, scikit-learn, XGBoost, PuLP, Statsmodels, Plotly

**Highlights:**
- 3,000+ lines of production code
- 36 comprehensive tests (100% passing)
- 47K+ observations processed
- 100 products analyzed with elasticity coefficients

### 2. Demand Forecasting System 📈
**Status:** Complete

Time series forecasting for retail demand with hierarchical models and ensemble methods.

**Tech:** Python, LightGBM, Prophet, M5 Dataset

### 3. Inventory Optimization Engine 📦
**Status:** Planned

Multi-echelon inventory optimization with stochastic modeling.

**Tech:** Python, OR-Tools, Scipy, Simpy

## Structure

```
docs/
├── index.html              # Homepage
├── css/
│   └── style.css          # Main stylesheet (responsive)
├── js/
│   └── main.js            # Interactive features
├── images/                 # Project screenshots & assets
└── projects/
    ├── dynamic-pricing.html      # Detailed project page
    ├── demand-forecasting.html   # Project page
    └── inventory-optimization.html
```

## Technologies Used

### Frontend
- **HTML5:** Semantic markup
- **CSS3:** Modern layout (Grid, Flexbox), animations, gradients
- **JavaScript:** Vanilla JS (no frameworks)
- **Font Awesome:** Icons
- **Google Fonts:** Inter (UI) + JetBrains Mono (code)

### Hosting
- **GitHub Pages:** Free static site hosting
- **Custom Domain:** (Optional) Configure in settings

## Setup & Deployment

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/GodsonKurishinkal/data-science-portfolio.git
cd data-science-portfolio
```

2. Serve locally:
```bash
# Python 3
python3 -m http.server 8000 --directory docs

# Or use any static server
npx serve docs
```

3. Open in browser:
```
http://localhost:8000
```

### Deploy to GitHub Pages

1. **Enable GitHub Pages:**
   - Go to repository **Settings**
   - Navigate to **Pages** section
   - Source: Select **main** branch
   - Folder: Select **/docs**
   - Click **Save**

2. **Access Site:**
   - URL: `https://[username].github.io/[repo-name]/`
   - Example: `https://godsonkurishinkal.github.io/data-science-portfolio/`

3. **Custom Domain (Optional):**
   - Add CNAME file in `/docs/` with your domain
   - Configure DNS with GitHub Pages IPs
   - Enable HTTPS in settings

### Configuration

#### Update Personal Information

Edit `docs/index.html`:

```html
<!-- Line 52: Update name -->
<span class="name">Your Name</span>

<!-- Line 248: Update GitHub -->
<a href="https://github.com/yourusername">

<!-- Line 477: Update email -->
<a href="mailto:your.email@example.com">
```

#### Update Projects

Edit project cards in `docs/index.html`:

```html
<!-- Line 291: Project 1 Details -->
<div class="project-card featured">
    <h3>Your Project Name</h3>
    <p>Your project description...</p>
</div>
```

#### Customize Colors

Edit `docs/css/style.css`:

```css
/* Line 8: Color variables */
:root {
    --primary-color: #3b82f6;      /* Change primary */
    --secondary-color: #10b981;    /* Change secondary */
    --accent-color: #f59e0b;       /* Change accent */
}
```

## Performance

### Optimization Checklist
- ✅ Minified CSS (production)
- ✅ Lazy loading images
- ✅ Optimized animations
- ✅ Compressed fonts
- ✅ CDN for libraries (Font Awesome)
- ✅ Async script loading

### Lighthouse Scores
- **Performance:** 95+
- **Accessibility:** 100
- **Best Practices:** 100
- **SEO:** 100

## Browser Support

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari (latest)
- ✅ Mobile browsers (iOS Safari, Chrome)

## Customization Ideas

### Add New Project

1. Create HTML file: `docs/projects/your-project.html`
2. Copy structure from existing project page
3. Update content, metrics, and code examples
4. Add project card to `docs/index.html`

### Add Blog Section

```html
<!-- Add to navigation -->
<a href="#blog" class="nav-link">Blog</a>

<!-- Create blog section -->
<section class="blog" id="blog">
    <!-- Blog posts here -->
</section>
```

### Integrate Contact Form

Use services like:
- **Formspree:** https://formspree.io/
- **Netlify Forms:** Built-in form handling
- **EmailJS:** Client-side email sending

```html
<!-- Update form action -->
<form action="https://formspree.io/f/your-id" method="POST">
```

## Analytics

### Google Analytics

Add to `<head>` in `index.html`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

## Maintenance

### Regular Updates
- Update project statuses
- Add new projects as completed
- Refresh metrics and statistics
- Update skills section with new technologies
- Add blog posts or articles

### SEO Optimization
- Update meta descriptions
- Add structured data (JSON-LD)
- Create sitemap.xml
- Submit to Google Search Console

## Contributing

This is a personal portfolio, but suggestions are welcome!

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Submit pull request

## License

MIT License - Feel free to use this template for your own portfolio!

## Contact

**Godson Kurishinkal**
- GitHub: [@GodsonKurishinkal](https://github.com/GodsonKurishinkal)
- LinkedIn: [godsonkurishinkal](https://linkedin.com/in/godsonkurishinkal)
- Email: godson@example.com

## Acknowledgments

- Icons: [Font Awesome](https://fontawesome.com/)
- Fonts: [Google Fonts](https://fonts.google.com/)
- Hosting: [GitHub Pages](https://pages.github.com/)
- Inspiration: Modern portfolio designs from Dribbble & Behance

---

**Built with ❤️ for Data Science**

Last Updated: November 2024
