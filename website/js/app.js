/* LazyBridge — app.js */

// ── Tab switching ──────────────────────────────────────────────────────────
function initTabs(containerSelector, btnSelector, panelSelector) {
  document.querySelectorAll(containerSelector).forEach(container => {
    const btns   = container.querySelectorAll(btnSelector);
    const panels = container.querySelectorAll(panelSelector);

    btns.forEach((btn, i) => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        panels[i] && panels[i].classList.add('active');
      });
    });

    // activate first by default if none active
    if (!container.querySelector(btnSelector + '.active') && btns.length > 0) {
      btns[0].classList.add('active');
      panels[0] && panels[0].classList.add('active');
    }
  });
}

// ── Copy to clipboard ──────────────────────────────────────────────────────
function initCopyButtons() {
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const wrap = btn.closest('.code-block-wrap');
      const pre  = wrap && wrap.querySelector('pre');
      const text = pre ? pre.innerText : '';
      navigator.clipboard.writeText(text).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = orig;
          btn.classList.remove('copied');
        }, 2000);
      }).catch(() => {});
    });
  });

  // install pill copy
  document.querySelectorAll('.install-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      const cmd = pill.dataset.cmd || pill.querySelector('.cmd')?.textContent;
      if (!cmd) return;
      navigator.clipboard.writeText(cmd.trim()).then(() => {
        const orig = pill.querySelector('.copy-icon')?.innerHTML;
        if (pill.querySelector('.copy-icon')) {
          pill.querySelector('.copy-icon').innerHTML = '✓';
          setTimeout(() => {
            if (orig) pill.querySelector('.copy-icon').innerHTML = orig;
          }, 2000);
        }
      }).catch(() => {});
    });
  });
}

// ── Nav copy badge ─────────────────────────────────────────────────────────
function initNavInstall() {
  document.querySelectorAll('.nav-install').forEach(btn => {
    btn.addEventListener('click', () => {
      navigator.clipboard.writeText('pip install lazybridge').then(() => {
        const orig = btn.textContent;
        btn.textContent = '✓ Copied';
        setTimeout(() => { btn.textContent = orig; }, 2000);
      }).catch(() => {});
    });
  });
}

// ── Smooth scroll for anchor links ────────────────────────────────────────
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.getElementById(a.getAttribute('href').slice(1));
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

// ── Active nav highlight on scroll (docs) ─────────────────────────────────
function initDocsSidebar() {
  const sidebar = document.querySelector('.docs-sidebar');
  if (!sidebar) return;

  const sections = document.querySelectorAll('.docs-content section[id]');
  const links    = sidebar.querySelectorAll('a[href^="#"]');

  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        links.forEach(l => l.classList.remove('active'));
        const link = sidebar.querySelector(`a[href="#${entry.target.id}"]`);
        if (link) link.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });

  sections.forEach(s => obs.observe(s));
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTabs('.tabs-container', '.tab-btn', '.tab-panel');
  initTabs('.variants-container', '.variant-btn', '.variant-panel');
  initCopyButtons();
  initNavInstall();
  initSmoothScroll();
  initDocsSidebar();
});
