#!/usr/bin/env python3
"""
Script to update project pages to use the new minimal design system.
Updates CSS references, navigation, and adds animation scripts.
"""

import re
from pathlib import Path

# Project pages to update
PROJECT_PAGES = [
    'inventory-optimization.html',
    'supply-chain-network.html',
    'realtime-demand-sensing.html'
]

# New CSS references to inject
NEW_CSS = '''    
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>{EMOJI}</text></svg>">
    
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <link rel="stylesheet" href="../css/minimal.css">
    <link rel="stylesheet" href="../css/minimal-project.css">
</head>'''

# New navigation HTML
NEW_NAV = '''    <!-- Navigation -->
    <nav class="nav">
        <div class="nav-container container">
            <a href="../index.html" class="nav-brand">Godson Kurishinkal</a>
            <ul class="nav-menu">
                <li><a href="../index.html#home" class="nav-link">Home</a></li>
                <li><a href="../index.html#projects" class="nav-link">Projects</a></li>
                <li><a href="../index.html#skills" class="nav-link">Skills</a></li>
                <li><a href="../index.html#contact" class="nav-link">Contact</a></li>
            </ul>
        </div>
    </nav>'''

# New footer HTML
NEW_FOOTER = '''    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <p class="footer-text">© 2024 Godson Kurishinkal. All rights reserved.</p>
                <ul class="footer-links">
                    <li><a href="../index.html#home" class="footer-link">Home</a></li>
                    <li><a href="../index.html#projects" class="footer-link">Projects</a></li>
                    <li><a href="../index.html#contact" class="footer-link">Contact</a></li>
                </ul>
                <div class="social-links">
                    <a href="https://github.com/GodsonKurishinkal" target="_blank" class="social-link" aria-label="GitHub">
                        <i class="fab fa-github"></i>
                    </a>
                    <a href="https://linkedin.com/in/godsonkurishinkal" target="_blank" class="social-link" aria-label="LinkedIn">
                        <i class="fab fa-linkedin"></i>
                    </a>
                    <a href="mailto:godson.kurishinkal+github@gmail.com" class="social-link" aria-label="Email">
                        <i class="fas fa-envelope"></i>
                    </a>
                </div>
            </div>
        </div>
    </footer>

    <script>
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    const navHeight = document.querySelector('.nav').offsetHeight;
                    const targetPosition = target.offsetTop - navHeight;
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });

        // Nav scroll effect
        const nav = document.querySelector('.nav');
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                nav.classList.add('scrolled');
            } else {
                nav.classList.remove('scrolled');
            }
        });

        // Scroll reveal animation
        const revealElements = document.querySelectorAll('.feature-card, .tech-item, .metric-card, .timeline-item');
        
        const revealOnScroll = () => {
            revealElements.forEach(element => {
                const elementTop = element.getBoundingClientRect().top;
                const windowHeight = window.innerHeight;
                
                if (elementTop < windowHeight * 0.85) {
                    element.style.opacity = '1';
                    element.style.transform = 'translateY(0)';
                }
            });
        };

        // Initialize elements
        revealElements.forEach(element => {
            element.style.opacity = '0';
            element.style.transform = 'translateY(30px)';
            element.style.transition = 'opacity 0.8s cubic-bezier(0.4, 0, 0.2, 1), transform 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        });

        window.addEventListener('scroll', revealOnScroll);
        window.addEventListener('load', revealOnScroll);
        revealOnScroll();
    </script>
</body>
</html>'''

# Project-specific emojis
PROJECT_EMOJIS = {
    'inventory-optimization.html': '📦',
    'supply-chain-network.html': '🌐',
    'realtime-demand-sensing.html': '⚡'
}

def update_project_page(filepath):
    """Update a single project page with new design system."""
    print(f"Updating {filepath.name}...")
    
    # Read the file
    content = filepath.read_text(encoding='utf-8')
    
    # 1. Remove all inline styles (everything between <style> and </style>)
    content = re.sub(r'<style>.*?</style>', '', content, flags=re.DOTALL)
    
    # 2. Update CSS references
    # Replace old CSS links with new ones
    emoji = PROJECT_EMOJIS.get(filepath.name, '📊')
    new_css_with_emoji = NEW_CSS.replace('{EMOJI}', emoji)
    
    content = re.sub(
        r'<link rel="stylesheet".*?href="../css/(unified|projects|style|modern)\.css".*?>',
        '',
        content,
        flags=re.DOTALL
    )
    
    # Insert new CSS before </head>
    content = re.sub(r'</head>', new_css_with_emoji, content)
    
    # 3. Update navigation
    content = re.sub(
        r'<!-- Navigation -->.*?</nav>',
        NEW_NAV,
        content,
        flags=re.DOTALL
    )
    
    # 4. Update footer and add scripts
    content = re.sub(
        r'<!-- Footer -->.*?</html>',
        NEW_FOOTER,
        content,
        flags=re.DOTALL
    )
    
    # 5. Clean up multiple blank lines
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    # Write back
    filepath.write_text(content, encoding='utf-8')
    print(f"✓ Updated {filepath.name}")

def main():
    """Main function to update all project pages."""
    projects_dir = Path(__file__).parent / 'projects'
    
    print("=" * 60)
    print("Updating Project Pages to Minimal Design System")
    print("=" * 60)
    print()
    
    for page_name in PROJECT_PAGES:
        filepath = projects_dir / page_name
        if filepath.exists():
            try:
                update_project_page(filepath)
            except Exception as e:
                print(f"✗ Error updating {page_name}: {e}")
        else:
            print(f"✗ File not found: {page_name}")
    
    print()
    print("=" * 60)
    print("Update Complete!")
    print("=" * 60)
    print()
    print("Changes applied:")
    print("  ✓ Removed all inline styles")
    print("  ✓ Updated CSS references to minimal.css & minimal-project.css")
    print("  ✓ Replaced navigation with new minimal nav")
    print("  ✓ Replaced footer with new minimal footer")
    print("  ✓ Added scroll animations and nav effects")
    print()

if __name__ == '__main__':
    main()
